import logging
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange

from experiments.utils import count_parameters, ddp_setup, set_logger, set_seed

set_logger()


@torch.no_grad()
def evaluate(model, loader, device, num_batches=None):
    model.eval()
    loss = 0.0
    correct = 0.0
    total = 0.0
    predicted, gt = [], []
    for i, batch in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        batch = batch.to(device)
        inputs = (batch.weights, batch.biases)
        out = model(inputs)
        loss += F.cross_entropy(out, batch.label, reduction="sum")
        total += len(batch.label)
        pred = out.argmax(1)
        correct += pred.eq(batch.label).sum()
        predicted.extend(pred.cpu().numpy().tolist())
        gt.extend(batch.label.cpu().numpy().tolist())

    model.train()
    avg_loss = loss / total
    avg_acc = correct / total

    return dict(avg_loss=avg_loss, avg_acc=avg_acc, predicted=predicted, gt=gt)


def train(cfg, hydra_cfg):
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.seed is not None:
        set_seed(cfg.seed)

    rank = OmegaConf.select(cfg, "distributed.rank", default=0)
    ckpt_dir = Path(hydra_cfg.runtime.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb.name is None:
        model_name = cfg.model._target_.split(".")[-1]
        cfg.wandb.name = (
            f"{cfg.data.dataset_name}_clf_{model_name}"
            f"_bs_{cfg.batch_size}_seed_{cfg.seed}"
        )
    if rank == 0:
        wandb.init(
            **OmegaConf.to_container(cfg.wandb, resolve=True),
            settings=wandb.Settings(start_method="fork"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    train_set = hydra.utils.instantiate(cfg.data.train)
    val_set = hydra.utils.instantiate(cfg.data.val)
    test_set = hydra.utils.instantiate(cfg.data.test)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=cfg.batch_size,
        shuffle=not cfg.distributed,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=DistributedSampler(train_set) if cfg.distributed else None,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    if rank == 0:
        logging.info(
            f"train size {len(train_set)}, "
            f"val size {len(val_set)}, "
            f"test size {len(test_set)}"
        )

    point = train_set[0]
    weight_shapes = tuple(w.shape[:2] for w in point.weights)
    bias_shapes = tuple(b.shape[:1] for b in point.biases)

    layer_layout = [weight_shapes[0][0]] + [b[0] for b in bias_shapes]
    if rank == 0:
        logging.info(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")
    model_kwargs = dict()
    model_cls = cfg.model._target_.split(".")[-1]
    if model_cls == "DWSModelForClassification":
        model_kwargs["weight_shapes"] = weight_shapes
        model_kwargs["bias_shapes"] = bias_shapes
    else:
        model_kwargs["layer_layout"] = layer_layout
    model = hydra.utils.instantiate(cfg.model, **model_kwargs).to(device)

    if rank == 0:
        logging.info(f"number of parameters: {count_parameters(model)}")

    if cfg.compile:
        model = torch.compile(model, **cfg.compile_kwargs)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = hydra.utils.instantiate(cfg.optim, params=parameters)
    if hasattr(cfg, "scheduler"):
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1
    best_test_results, best_val_results = None, None
    test_acc, test_loss = -1.0, -1.0
    global_step = 0
    start_epoch = 0

    if cfg.load_ckpt:
        ckpt = torch.load(cfg.load_ckpt)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
        if "global_step" in ckpt:
            global_step = ckpt["global_step"]
        if rank == 0:
            logging.info(f"loaded checkpoint {cfg.load_ckpt}")

    epoch_iter = trange(start_epoch, cfg.n_epochs, disable=rank != 0)
    if cfg.distributed:
        model = DDP(
            model, device_ids=cfg.distributed.device_ids, find_unused_parameters=False
        )
    model.train()

    if rank == 0:
        ckpt_dir = Path(hydra_cfg.runtime.output_dir) / wandb.run.path.split("/")[-1]
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    scaler = GradScaler(**cfg.gradscaler)
    autocast_kwargs = dict(cfg.autocast)
    autocast_kwargs["dtype"] = getattr(torch, cfg.autocast.dtype, torch.float32)
    optimizer.zero_grad()
    for epoch in epoch_iter:
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            inputs = (batch.weights, batch.biases)
            label = batch.label

            with torch.autocast(**autocast_kwargs):
                out = model(inputs)
                loss = criterion(out, label) / cfg.num_accum

            scaler.scale(loss).backward()
            log = {
                "train/loss": loss.item() * cfg.num_accum,
                "global_step": global_step,
            }

            if ((i + 1) % cfg.num_accum == 0) or (i + 1 == len(train_loader)):
                if cfg.clip_grad:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters, cfg.clip_grad_max_norm
                    )
                    log["grad_norm"] = grad_norm
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None:
                    log["lr"] = scheduler.get_last_lr()[0]
                    scheduler.step()

            if rank == 0:
                wandb.log(log)
                epoch_iter.set_description(
                    f"[{epoch} {i+1}], train loss: {log['train/loss']:.3f}, test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}"
                )
            global_step += 1

            if (global_step + 1) % cfg.eval_every == 0 and rank == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "cfg": cfg,
                        "global_step": global_step,
                    },
                    ckpt_dir / "latest.ckpt",
                )

                val_loss_dict = evaluate(model, val_loader, device)
                test_loss_dict = evaluate(model, test_loader, device)
                val_loss = val_loss_dict["avg_loss"]
                val_acc = val_loss_dict["avg_acc"]
                test_loss = test_loss_dict["avg_loss"]
                test_acc = test_loss_dict["avg_acc"]

                best_val_criteria = val_acc >= best_val_acc

                if best_val_criteria:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "cfg": cfg,
                            "global_step": global_step,
                        },
                        ckpt_dir / "best_val.ckpt",
                    )
                    best_val_acc = val_acc
                    best_test_results = test_loss_dict
                    best_val_results = val_loss_dict

                log = {
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/best_loss": best_val_results["avg_loss"],
                    "val/best_acc": best_val_results["avg_acc"],
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "test/best_loss": best_test_results["avg_loss"],
                    "test/best_acc": best_test_results["avg_acc"],
                    "epoch": epoch,
                    "global_step": global_step,
                }

                wandb.log(log)


def train_ddp(rank, cfg, hydra_cfg):
    ddp_setup(rank, cfg.distributed.world_size)
    cfg.distributed.rank = rank
    train(cfg, hydra_cfg)
    destroy_process_group()


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    if cfg.distributed:
        mp.spawn(
            train_ddp,
            args=(cfg, hydra_cfg),
            nprocs=cfg.distributed.world_size,
            join=True,
        )
    else:
        train(cfg, hydra_cfg)


if __name__ == "__main__":
    main()

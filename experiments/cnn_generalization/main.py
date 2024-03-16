import logging
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from tqdm import trange
import wandb
import hydra
from omegaconf import OmegaConf

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch_geometric
from sklearn.metrics import r2_score
from scipy.stats import kendalltau

from experiments.utils import (
    count_parameters,
    set_logger,
    set_seed,
)
from experiments.inr_classification.main import ddp_setup
from nn.nfn.common.data import WeightSpaceFeatures, network_spec_from_wsfeat

set_logger()

warnings.filterwarnings("ignore", ".*TypedStorage is deprecated.*")
OmegaConf.register_new_resolver("prod", lambda x, y: x * y)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_batches=None, data_format="graph"):
    model.eval()
    pred, gt = [], []
    losses = []
    for i, batch in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break

        if data_format in ("graph", "stat"):
            inputs = batch.to(device)
        else:
            inputs = WeightSpaceFeatures(
                [wi.unsqueeze(1) for wi in batch.weights],
                [bi.unsqueeze(1) for bi in batch.biases],
            ).to(device)
        gt_accuracy = batch.y.to(device)

        pred_acc = F.sigmoid(model(inputs)).squeeze(-1)
        losses.append(criterion(pred_acc, gt_accuracy).item())
        pred.append(pred_acc.detach().cpu().numpy())
        gt.append(batch.y.cpu().numpy())

    model.train()
    avg_loss = np.mean(losses)
    gt = np.concatenate(gt)
    pred = np.concatenate(pred)
    rsq = r2_score(gt, pred)
    tau = kendalltau(gt, pred).statistic
    return dict(avg_loss=avg_loss, rsq=rsq, tau=tau, gt=gt, pred=pred)


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
        cfg.wandb.name = f"cnn_gen_{model_name}" f"_bs_{cfg.batch_size}_seed_{cfg.seed}"
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

    train_loader = torch_geometric.loader.DataLoader(
        dataset=train_set,
        batch_size=cfg.batch_size,
        shuffle=not cfg.distributed,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=DistributedSampler(train_set) if cfg.distributed else None,
    )
    val_loader = torch_geometric.loader.DataLoader(
        dataset=val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = torch_geometric.loader.DataLoader(
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

    model_args = []
    model_kwargs = dict()
    model_cls = cfg.model._target_.split(".")[-1]
    if model_cls == "DWSModelForClassification":
        model_kwargs["weight_shapes"] = None
        model_kwargs["bias_shapes"] = None
    elif model_cls in ("InvariantNFN", "StatNet"):
        data_sample = next(iter(train_loader))
        network_spec = network_spec_from_wsfeat(
            WeightSpaceFeatures(
                [wi.unsqueeze(1) for wi in data_sample.weights],
                [bi.unsqueeze(1) for bi in data_sample.biases],
            )
        )
        model_args.append(network_spec)
    model = hydra.utils.instantiate(cfg.model, *model_args, **model_kwargs).to(device)

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

    criterion = hydra.utils.instantiate(cfg.loss)
    best_val_tau = -float("inf")
    best_test_results, best_val_results = None, None
    test_loss = -1.0
    test_tau = -float("inf")
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
            if cfg.data.data_format in ("graph", "stat"):
                inputs = batch.to(device)
            else:
                inputs = WeightSpaceFeatures(
                    [wi.unsqueeze(1) for wi in batch.weights],
                    [bi.unsqueeze(1) for bi in batch.biases],
                ).to(device)
            gt_accuracy = batch.y.to(device)

            with torch.autocast(**autocast_kwargs):
                out = F.sigmoid(model(inputs)).squeeze(-1)
                loss = criterion(out, gt_accuracy) / cfg.num_accum

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
                    f"[{epoch} {i+1}], train loss: {log['train/loss']:.5f}, test_loss: {test_loss:.5f}, test_tau: {test_tau:.3f}"
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

                val_loss_dict = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    data_format=cfg.data.data_format,
                )
                test_loss_dict = evaluate(
                    model,
                    test_loader,
                    criterion,
                    device,
                    data_format=cfg.data.data_format,
                )

                val_loss = val_loss_dict["avg_loss"]
                val_rsq = val_loss_dict["rsq"]
                val_tau = val_loss_dict["tau"]

                test_loss = test_loss_dict["avg_loss"]
                test_rsq = test_loss_dict["rsq"]
                test_tau = test_loss_dict["tau"]

                best_val_criteria = val_tau >= best_val_tau

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
                    best_val_tau = val_tau
                    best_test_results = test_loss_dict
                    best_val_results = val_loss_dict

                plt.clf()
                plot = plt.scatter(val_loss_dict["gt"], val_loss_dict["pred"])
                plt.xlabel("Actual model accuracy")
                plt.ylabel("Predicted model accuracy")

                log = {
                    "val/loss": val_loss,
                    "val/rsq": val_rsq,
                    "val/kendall_tau": val_tau,
                    "val/best_loss": best_val_results["avg_loss"],
                    "val/best_tau": best_val_results["tau"],
                    "val/scatter": wandb.Image(plot),
                    "test/loss": test_loss,
                    "test/rsq": test_rsq,
                    "test/kendall_tau": test_tau,
                    "test/best_loss": best_test_results["avg_loss"],
                    "test/best_tau": best_test_results["tau"],
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

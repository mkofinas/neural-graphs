import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from einops import rearrange
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import trange

from experiments.data_nfn import SirenAndOriginalDataset
from experiments.utils import count_parameters, ddp_setup, set_logger, set_seed

set_logger()


def residual_param_update(
    weights, biases, delta_weights, delta_biases, data_format="dws_mnist"
):
    if data_format == "dws_mnist":
        new_weights = [weights[j] + delta_weights[j] for j in range(len(weights))]
        new_biases = [biases[j] + delta_biases[j] for j in range(len(weights))]
    else:
        new_weights = [
            weights[j].permute(0, 3, 2, 1) + delta_weights[j]
            for j in range(len(weights))
        ]
        new_biases = [
            biases[j].permute(0, 2, 1) + delta_biases[j] for j in range(len(weights))
        ]
    return new_weights, new_biases


def log_images(gt_images, pred_images, input_images, data_format="dws_mnist"):
    _gt_images = gt_images.detach().permute(0, 2, 3, 1).cpu().numpy()
    _pred_images = pred_images.detach().permute(0, 2, 3, 1).cpu().numpy()
    _input_images = input_images.detach().permute(0, 2, 3, 1).cpu().numpy()
    if data_format != "dws_mnist":
        _gt_images = 0.5 * _gt_images + 0.5
        _pred_images = 0.5 * _pred_images + 0.5
        _input_images = 0.5 * _input_images + 0.5
    _pred_images = np.clip(_pred_images, 0.0, 1.0)
    gt_img_plots = [wandb.Image(img) for img in gt_images]
    pred_img_plots = [wandb.Image(img) for img in pred_images]
    input_img_plots = [wandb.Image(img) for img in _input_images]
    return gt_img_plots, pred_img_plots, input_img_plots


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    inr_model,
    img_shape=(28, 28),
    data_format="dws_mnist",
    log_n_imgs=0,
    num_batches=None,
):
    model.eval()
    log_n_imgs = min(log_n_imgs, loader.batch_size)
    loss = 0.0
    imgs, preds, input_imgs = [], [], []
    losses = []
    for i, batch in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        params, img, input_image = batch
        img = img.to(device)
        if data_format == "dws_mnist":
            params = params.to(device)
            weights = params.weights
            biases = params.biases
            inputs = (params.weights, params.biases)
        else:
            weights, biases = params
            weights, biases = (
                [w.to(device) for w in weights],
                [b.to(device) for b in biases],
            )
            inputs = (
                [w.to(device).permute(0, 3, 2, 1) for w in weights],
                [b.to(device).permute(0, 2, 1) for b in biases],
            )

        delta_weights, delta_biases = model(inputs)
        new_weights, new_biases = residual_param_update(
            weights, biases, delta_weights, delta_biases, data_format
        )

        outs = inr_model(new_weights, new_biases)
        outs = rearrange(outs, "b (h w) c -> b c h w", h=img_shape[0])
        loss = ((outs - img) ** 2).mean(dim=(1, 2, 3))
        losses.append(loss.detach().cpu())

        if i == 0 and log_n_imgs > 0:
            gt_img_plots, pred_img_plots, input_img_plots = log_images(
                img[:log_n_imgs],
                outs[:log_n_imgs],
                input_image[:log_n_imgs],
                data_format,
            )
            imgs.extend(gt_img_plots)
            preds.extend(pred_img_plots)
            input_imgs.extend(input_img_plots)

    losses = torch.cat(losses)
    losses = losses.mean()

    model.train()
    return {
        "avg_loss": losses,
        "imgs/gt": imgs,
        "imgs/pred": preds,
        "imgs/input": input_imgs,
    }


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
            f"{cfg.data.dataset_name}_style_{model_name}"
            f"_bs_{cfg.batch_size}_seed_{cfg.seed}"
        )
    if rank == 0:
        wandb.init(
            **OmegaConf.to_container(cfg.wandb, resolve=True),
            settings=wandb.Settings(start_method="fork"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.data.target.endswith("SirenAndOriginalDataset"):
        data_tfm = transforms.Compose(
            [
                transforms.Lambda(np.array),
                hydra.utils.instantiate(cfg.data.style),
                transforms.ToTensor(),
                transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
            ]
        )

        dset = SirenAndOriginalDataset(
            cfg.data.siren_path, "randinit_smaller", cfg.data.img_path, data_tfm
        )

        train_set = torch.utils.data.Subset(dset, range(45_000))
        val_set = torch.utils.data.Subset(dset, range(45_000, 50_000))
        test_set = torch.utils.data.Subset(dset, range(50_000, 60_000))

        inr_model = hydra.utils.instantiate(cfg.data.batch_siren).to(device)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=cfg.batch_size,
            shuffle=not cfg.distributed,
            num_workers=cfg.num_workers,
            pin_memory=True,
            sampler=DistributedSampler(train_set) if cfg.distributed else None,
            drop_last=True,
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
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        out_of_domain_val_loader = None
        out_of_domain_test_loader = None
    elif cfg.data.data_format == "dws_mnist":
        train_set = hydra.utils.instantiate(cfg.data.train)
        val_set = hydra.utils.instantiate(cfg.data.val)
        test_set = hydra.utils.instantiate(cfg.data.test)
        out_of_domain_val_set = hydra.utils.instantiate(cfg.out_of_domain_data.val)
        out_of_domain_test_set = hydra.utils.instantiate(cfg.out_of_domain_data.test)

        inr_model = hydra.utils.instantiate(cfg.data.batch_siren).to(device)

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
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        out_of_domain_val_loader = torch.utils.data.DataLoader(
            dataset=out_of_domain_val_set,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        out_of_domain_test_loader = torch.utils.data.DataLoader(
            dataset=out_of_domain_test_set,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    if rank == 0:
        logging.info(
            f"train size {len(train_set)}, "
            f"val size {len(val_set)}, "
            f"test size {len(test_set)}"
        )

    if cfg.data.data_format == "dws_mnist":
        point = train_set[0][0]
        weight_shapes = tuple(w.shape[:2] for w in point.weights)
        bias_shapes = tuple(b.shape[:1] for b in point.biases)
    else:
        point = train_set[0][0]
        weight_shapes = tuple(w.transpose(-1, -2).shape[1:] for w in point[0])
        bias_shapes = tuple(b.shape[1:] for b in point[1])

    layer_layout = [weight_shapes[0][0]] + [b[0] for b in bias_shapes]

    if rank == 0:
        logging.info(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")

    model_kwargs = dict()
    model_cls = cfg.model._target_.split(".")[-1]
    # TODO: make defaults for MLP so that parameters for MLP and DWS are the same
    if model_cls == "MLPModel":
        model_kwargs["in_dim"] = sum([w.numel() for w in weight_shapes + bias_shapes])
    elif model_cls == "DWSModel":
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

    criterion = torch.nn.MSELoss()
    best_val_loss = float("inf")
    best_test_results, best_val_results = None, None
    test_loss = -1.0
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
            params, img, input_image = batch
            img = img.to(device)
            if cfg.data.data_format == "dws_mnist":
                params = params.to(device)
                weights = params.weights
                biases = params.biases
                inputs = (params.weights, params.biases)
            else:
                weights, biases = params
                weights, biases = (
                    [w.to(device) for w in weights],
                    [b.to(device) for b in biases],
                )
                inputs = (
                    [w.to(device).permute(0, 3, 2, 1) for w in weights],
                    [b.to(device).permute(0, 2, 1) for b in biases],
                )

            with torch.autocast(**autocast_kwargs):
                delta_weights, delta_biases = model(inputs)
                new_weights, new_biases = residual_param_update(
                    weights, biases, delta_weights, delta_biases, cfg.data.data_format
                )

                outs = inr_model(new_weights, new_biases)
                outs = rearrange(outs, "b (h w) c -> b c h w", h=cfg.data.img_shape[0])
                loss = criterion(outs, img) / cfg.num_accum

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
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
                if i == 0:
                    (
                        log["train/imgs/gt"],
                        log["train/imgs/pred"],
                        log["train/imgs/input"],
                    ) = log_images(
                        img[: cfg.log_n_imgs],
                        outs[: cfg.log_n_imgs],
                        input_image[: cfg.log_n_imgs],
                        cfg.data.data_format,
                    )
                wandb.log(log)

                epoch_iter.set_description(
                    f"[{epoch} {i+1}], train loss: {loss.item():.3f}, test_loss: {test_loss:.3f}"
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
                    device,
                    inr_model,
                    img_shape=cfg.data.img_shape,
                    data_format=cfg.data.data_format,
                    log_n_imgs=cfg.log_n_imgs,
                )
                test_loss_dict = evaluate(
                    model,
                    test_loader,
                    device,
                    inr_model,
                    img_shape=cfg.data.img_shape,
                    data_format=cfg.data.data_format,
                    log_n_imgs=cfg.log_n_imgs,
                )
                if out_of_domain_val_loader is not None:
                    out_of_domain_val_loss_dict = evaluate(
                        model,
                        out_of_domain_val_loader,
                        device,
                        inr_model,
                        img_shape=cfg.data.img_shape,
                        data_format=cfg.data.data_format,
                        log_n_imgs=cfg.log_n_imgs,
                    )
                    out_of_domain_test_loss_dict = evaluate(
                        model,
                        out_of_domain_test_loader,
                        device,
                        inr_model,
                        img_shape=cfg.data.img_shape,
                        data_format=cfg.data.data_format,
                        log_n_imgs=cfg.log_n_imgs,
                    )
                else:
                    out_of_domain_val_loss_dict = dict()
                    out_of_domain_test_loss_dict = dict()
                val_loss = val_loss_dict["avg_loss"]
                test_loss = test_loss_dict["avg_loss"]
                train_loss_dict = evaluate(
                    model,
                    train_loader,
                    device,
                    inr_model,
                    img_shape=cfg.data.img_shape,
                    data_format=cfg.data.data_format,
                    log_n_imgs=cfg.log_n_imgs,
                    num_batches=100,
                )

                best_val_criteria = val_loss < best_val_loss

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
                    best_test_results = test_loss_dict
                    best_val_results = val_loss_dict
                    best_val_loss = val_loss

                log = {
                    "train/avg_loss": train_loss_dict["avg_loss"],
                    "val/best_loss": best_val_results["avg_loss"],
                    "test/best_loss": best_test_results["avg_loss"],
                    **{f"val/{k}": v for k, v in val_loss_dict.items()},
                    **{f"test/{k}": v for k, v in test_loss_dict.items()},
                    **{
                        f"out_of_domain_val/{k}": v
                        for k, v in out_of_domain_val_loss_dict.items()
                    },
                    **{
                        f"out_of_domain_test/{k}": v
                        for k, v in out_of_domain_test_loss_dict.items()
                    },
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

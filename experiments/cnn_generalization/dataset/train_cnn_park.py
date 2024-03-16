from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import ray
from ray import tune
from ray import air
from ray.air.integrations.wandb import WandbLoggerCallback
import torch

from experiments.cnn_generalization.dataset import cnn_sampler
from experiments.cnn_generalization.dataset.cnn_trainer import NN_tune_trainable


def prepare_dataset(cfg: DictConfig):
    """
    partially from https://github.com/ModelZoos/ModelZooDataset/blob/main/code/zoo_generators/train_zoo_f_mnist_uniform.py
    """
    data_path = Path(cfg.efficient_dataset_path).expanduser().resolve()
    if not data_path.exists():
        Path(cfg.root).expanduser().resolve().mkdir(parents=True, exist_ok=True)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        val_and_trainset_raw = hydra.utils.instantiate(cfg.train)
        testset_raw = hydra.utils.instantiate(cfg.test)
        trainset_raw, valset_raw = torch.utils.data.random_split(
            val_and_trainset_raw,
            [len(val_and_trainset_raw) - 1, 1],
            generator=torch.Generator().manual_seed(cfg.dataset_seed),
        )

        # temp dataloaders
        trainloader_raw = torch.utils.data.DataLoader(
            dataset=trainset_raw, batch_size=len(trainset_raw), shuffle=True
        )
        valloader_raw = torch.utils.data.DataLoader(
            dataset=valset_raw, batch_size=len(valset_raw), shuffle=True
        )
        testloader_raw = torch.utils.data.DataLoader(
            dataset=testset_raw, batch_size=len(testset_raw), shuffle=True
        )
        # one forward pass
        assert (
            trainloader_raw.__len__() == 1
        ), "temp trainloader has more than one batch"
        for train_data, train_labels in trainloader_raw:
            pass
        assert valloader_raw.__len__() == 1, "temp valloader has more than one batch"
        for val_data, val_labels in valloader_raw:
            pass
        assert testloader_raw.__len__() == 1, "temp testloader has more than one batch"
        for test_data, test_labels in testloader_raw:
            pass

        trainset = torch.utils.data.TensorDataset(train_data, train_labels)
        valset = torch.utils.data.TensorDataset(val_data, val_labels)
        testset = torch.utils.data.TensorDataset(test_data, test_labels)

        # save dataset and seed in data directory
        dataset = {
            "trainset": trainset,
            "valset": valset,
            "testset": testset,
            "dataset_seed": cfg.dataset_seed,
        }
        torch.save(dataset, data_path)


@hydra.main(
    config_path="generate_cnn_park_config", config_name="base", version_base=None
)
def main(cfg: DictConfig):
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.manual_seed(cfg.seed)

    # Resolve the relative path now
    cfg.data.efficient_dataset_path = (
        Path(cfg.data.efficient_dataset_path).expanduser().resolve()
    )

    prepare_dataset(cfg.data)

    ray.init(
        num_cpus=cfg.cpus,
        num_gpus=cfg.gpus,
    )

    gpu_fraction = ((cfg.gpus * 100) // (cfg.cpus / cfg.cpu_per_trial)) / 100
    resources_per_trial = {"cpu": cfg.cpu_per_trial, "gpu": gpu_fraction}

    assert ray.is_initialized() == True

    # create tune config
    tune_config = OmegaConf.to_container(cfg, resolve=True)
    model_configs = []
    for _ in range(cfg.num_models):
        model_configs.append(cnn_sampler.sample_cnn_config(cfg.random_options))
    tune_config["model"] = tune.grid_search(model_configs)

    # run tune trainable experiment
    analysis = tune.run(
        NN_tune_trainable,
        name=cfg.name,
        stop={
            "training_iteration": cfg.num_epochs,
        },
        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=cfg.ckpt_freq),
        config=tune_config,
        local_dir=Path(cfg.data.root).expanduser().resolve().as_posix(),
        callbacks=[WandbLoggerCallback(**cfg.wandb)],
        reuse_actors=False,
        # resume="ERRORED_ONLY",  # resumes from previous run. if run should be done all over, set resume=False
        # resume="LOCAL",  # resumes from previous run. if run should be done all over, set resume=False
        resume=False,  # resumes from previous run. if run should be done all over, set resume=False
        resources_per_trial=resources_per_trial,
        verbose=3,
    )

    ray.shutdown()
    assert ray.is_initialized() == False


if __name__ == "__main__":
    main()

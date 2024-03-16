from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from ray.tune import Trainable

from experiments.cnn_generalization.dataset.cnn_sampler import CNN
from experiments.cnn_generalization.dataset.fast_tensor_dataloader import (
    FastTensorDataLoader,
)


class NN_tune_trainable(Trainable):
    def setup(self, cfg: dict):
        self.cfg = OmegaConf.create(cfg)

        dataset = torch.load(self.cfg.data.efficient_dataset_path)
        self.trainset = dataset["trainset"]
        self.testset = dataset["testset"]
        self.valset = dataset.get("valset", None)

        # instantiate Tensordatasets
        self.trainloader = FastTensorDataLoader(
            dataset=self.trainset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            # num_workers=self.cfg.num_workers,
        )
        self.testloader = FastTensorDataLoader(
            dataset=self.testset, batch_size=len(self.testset), shuffle=False
        )

        self.steps_per_epoch = len(self.trainloader)

        # init model
        self.model = CNN(self.cfg.model).to(self.cfg.device)
        self.optimizer = hydra.utils.instantiate(
            self.cfg.optimizer, params=self.model.parameters()
        )

        # run first test epoch and log results
        self._iteration = -1

    def step(self):
        # here, all manual writers are disabled. tune takes care of that
        # run one training epoch
        train(self.model, self.optimizer, self.trainloader, self.cfg.device, 1)
        # run one test epoch
        test_results = evaluate(self.model, self.testloader, self.cfg.device)

        result_dict = {
            **{"test/" + k: v for k, v in test_results.items()},
        }
        # if self.valset is not None:
        #     pass
        self.stats = result_dict

        return result_dict

    def save_checkpoint(self, tmp_checkpoint_dir):
        # define checkpoint path
        path = Path(tmp_checkpoint_dir) / "checkpoint.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg.model,
                **self.get_state(),
            },
            path,
        )

        # tune apparently expects to return the directory
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        # define checkpoint path
        path = Path(tmp_checkpoint_dir) / "checkpoint.pt"
        # save model state dict
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        # load optimizer
        try:
            # opt_dict = torch.load(path / "optimizer")
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            print(f"Could not load optimizer state_dict. (not found at path {path})")

    def reset_config(self, new_config):
        success = False
        try:
            print(
                "### warning: reuse actors / reset_config only if the dataset remains exactly the same. \n ### only dataloader and model are reconfiugred"
            )
            self.cfg = new_config

            # init model
            self.NN = CNN(self.cfg.model).to(self.cfg.device)

            # instanciate Tensordatasets
            self.trainloader = FastTensorDataLoader(
                dataset=self.trainset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
            )
            self.testloader = FastTensorDataLoader(
                dataset=self.testset, batch_size=len(self.testset), shuffle=False
            )

            # drop inital checkpoint
            self.save()

            # run first test epoch and log results
            self._iteration = -1

            # if we got to this point:
            success = True

        except Exception as e:
            print(e)

        return success


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
) -> None:
    model.train()
    model.to(device)
    for e in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()


def evaluate(
    model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device
) -> dict:
    model.eval()
    model.to(device)
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            loss += F.cross_entropy(output, target, reduction="sum").item()

    return {
        "acc": correct / len(test_loader.dataset),
        "loss": loss / len(test_loader.dataset),
    }

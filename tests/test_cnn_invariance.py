import hydra
import pytest
import torch
from omegaconf import OmegaConf

from experiments.utils import set_seed
from tests.utils import wb_to_batch, permute_weights_biases

set_seed(42)

OmegaConf.register_new_resolver("prod", lambda x, y: x * y)


@pytest.fixture
def model():
    with hydra.initialize(
        version_base=None, config_path="../experiments/cnn_generalization/configs"
    ):
        cfg = hydra.compose(config_name="base")
        cfg.data.flattening_method = None
        cfg.data._max_kernel_height = 5
        cfg.data._max_kernel_width = 5
        model = hydra.utils.instantiate(cfg.model)
    return model


def test_model_invariance(model):
    batch_size = 4
    layer_layout = [3, 16, 32, 10]
    dims = [25, 25, 1]
    pad = (12, 12)
    weights = tuple(
        torch.randn(batch_size, layer_layout[i], layer_layout[i + 1], dims[i])
        for i in range(len(layer_layout) - 1)
    )
    biases = tuple(
        torch.randn(batch_size, layer_layout[i], 1) for i in range(1, len(layer_layout))
    )

    batch = wb_to_batch(weights, biases, layer_layout, pad=pad)
    out = model(batch)

    # Generate random permutations
    permutations = [
        torch.randperm(layer_layout[i]) for i in range(1, len(layer_layout) - 1)
    ]

    perm_weights, perm_biases = permute_weights_biases(weights, biases, permutations)
    perm_batch = wb_to_batch(perm_weights, perm_biases, layer_layout, pad=pad)

    out_perm = model(perm_batch)

    assert torch.allclose(out, out_perm, atol=1e-5, rtol=0)

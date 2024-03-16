import hydra
import pytest
import torch

from experiments.utils import set_seed
from tests.utils import permute_weights_biases

set_seed(42)


@pytest.fixture
def model():
    with hydra.initialize(
        version_base=None, config_path="../experiments/inr_classification/configs"
    ):
        cfg = hydra.compose(config_name="base")
        cfg.data.stats = None
        model = hydra.utils.instantiate(cfg.model, layer_layout=(2, 32, 32, 32, 32, 3))
    return model


def test_model_invariance(model):
    batch_size = 4
    layer_layout = [2, 32, 32, 32, 32, 3]
    weights = tuple(
        torch.randn(batch_size, layer_layout[i], layer_layout[i + 1], 1)
        for i in range(len(layer_layout) - 1)
    )
    biases = tuple(
        torch.randn(batch_size, layer_layout[i], 1) for i in range(1, len(layer_layout))
    )
    out = model((weights, biases))

    # Generate random permutations
    permutations = [
        torch.randperm(layer_layout[i]) for i in range(1, len(layer_layout) - 1)
    ]

    perm_weights, perm_biases = permute_weights_biases(weights, biases, permutations)
    out_perm = model((perm_weights, perm_biases))

    assert torch.allclose(out, out_perm, atol=1e-5, rtol=0)
    # return out, out_perm

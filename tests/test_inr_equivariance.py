import hydra
import pytest
import torch

from experiments.utils import set_seed
from tests.utils import permute_weights_biases

set_seed(42)


@pytest.fixture
def model():
    with hydra.initialize(
        version_base=None, config_path="../experiments/style_editing/configs"
    ):
        cfg = hydra.compose(config_name="base")
        cfg.data.stats = {
            "weights_mean": None,
            "weights_std": None,
            "biases_mean": None,
            "biases_std": None,
        }
        model = hydra.utils.instantiate(cfg.model, layer_layout=(2, 32, 32, 32, 32, 3))
    return model


def test_model_equivariance(model):
    batch_size = 4
    layer_layout = [2, 32, 32, 32, 32, 3]
    weights = tuple(
        torch.randn(batch_size, layer_layout[i], layer_layout[i + 1], 1)
        for i in range(len(layer_layout) - 1)
    )
    biases = tuple(
        torch.randn(batch_size, layer_layout[i], 1) for i in range(1, len(layer_layout))
    )
    out_weights, out_biases = model((weights, biases))

    # Generate random permutations
    permutations = [
        torch.randperm(layer_layout[i]) for i in range(1, len(layer_layout) - 1)
    ]

    perm_weights, perm_biases = permute_weights_biases(weights, biases, permutations)
    out_perm_weights, out_perm_biases = model((perm_weights, perm_biases))
    perm_out_weights, perm_out_biases = permute_weights_biases(
        out_weights, out_biases, permutations
    )

    for i in range(len(out_weights)):
        assert torch.allclose(
            perm_out_weights[i], out_perm_weights[i], atol=1e-5, rtol=1e-8
        )
        assert torch.allclose(
            perm_out_biases[i], out_perm_biases[i], atol=1e-4, rtol=1e-1
        )

import torch
import torch.nn.functional as F


class ActivationEmbedding(torch.nn.Module):
    ACTIVATION_FN = [
        "none",
        "relu",
        "gelu",
        "silu",
        "tanh",
        "sigmoid",
        "leaky_relu",
    ]

    def __init__(self, embedding_dim):
        super().__init__()

        self.activation_idx = {k: i for i, k in enumerate(self.ACTIVATION_FN)}
        self.idx_activation = {i: k for i, k in enumerate(self.ACTIVATION_FN)}

        self.embedding = torch.nn.Embedding(len(self.ACTIVATION_FN), embedding_dim)

    def forward(self, activations, layer_layout, device):
        indices = torch.tensor(
            [self.activation_idx[act] for act in activations],
            device=device,
            dtype=torch.long,
        )
        emb = self.embedding(indices)
        emb = emb.repeat_interleave(
            torch.tensor(layer_layout[1:-1], device=device), dim=0
        )
        emb = F.pad(emb, (0, 0, layer_layout[0], layer_layout[-1]))
        return emb

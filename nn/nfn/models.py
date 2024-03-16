from typing import Optional, Type, Union

import torch
from einops.layers.torch import Rearrange
from torch import nn

from nn.nfn.common import NetworkSpec, WeightSpaceFeatures
from nn.nfn.layers import (
    ChannelDropout,
    GaussianFourierFeatureTransform,
    HNPLinear,
    HNPPool,
    IOSinusoidalEncoding,
    LearnedScale,
    NPLinear,
    NPPool,
    ParamLayerNorm,
    Pointwise,
    SimpleLayerNorm,
    StatFeaturizer,
    TupleOp,
)

MODE2LAYER = {
    "PT": Pointwise,
    "NP": NPLinear,
    "NP-PosEmb": lambda *args, **kwargs: NPLinear(*args, io_embed=True, **kwargs),
    "HNP": HNPLinear,
}

LN_DICT = {
    "param": ParamLayerNorm,
    "simple": SimpleLayerNorm,
}

POOL_DICT = {"HNP": HNPPool, "NP": NPPool}


class NormalizingModule(nn.Module):
    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.weights_mean = [
            -0.0001166215879493393,
            -3.2710825053072767e-06,
            7.234242366394028e-05,
        ]
        self.weights_std = [
            0.06279338896274567,
            0.01827024295926094,
            0.11813738197088242,
        ]
        self.biases_mean = [
            4.912401891488116e-06,
            -3.210141949239187e-05,
            -0.012279038317501545,
        ]
        self.biases_std = [
            0.021347912028431892,
            0.0109943225979805,
            0.09998151659965515,
        ]

    # def set_stats(self, mean_std_stats):
    #     if self.normalize:
    #         print("Setting stats")
    #         weight_stats, bias_stats = mean_std_stats
    #         for i, (w, b) in enumerate(zip(weight_stats, bias_stats)):
    #             mean_weights, std_weights = w
    #             mean_bias, std_bias = b
    #             # wherever std_weights < 1e-5, set to 1
    #             std_weights = torch.where(std_weights < 1e-5, torch.ones_like(std_weights), std_weights)
    #             std_bias = torch.where(std_bias < 1e-5, torch.ones_like(std_bias), std_bias)
    #             self.register_buffer(f"mean_weights_{i}", mean_weights)
    #             self.register_buffer(f"std_weights_{i}", std_weights)
    #             self.register_buffer(f"mean_bias_{i}", mean_bias)
    #             self.register_buffer(f"std_bias_{i}", std_bias)

    def _normalize(self, params):
        out_weights, out_bias = [], []
        for i, (w, b) in enumerate(params):
            # mean_weights_i, std_weights_i = getattr(self, f"mean_weights_{i}"), getattr(self, f"std_weights_{i}")
            # mean_bias_i, std_bias_i = getattr(self, f"mean_bias_{i}"), getattr(self, f"std_bias_{i}")
            out_weights.append((w - self.weights_mean[i]) / self.weights_std[i])
            out_bias.append((b - self.biases_mean[i]) / self.biases_std[i])
        return WeightSpaceFeatures(out_weights, out_bias)

    def preprocess(self, params):
        if self.normalize:
            params = self._normalize(params)
        return params


class MlpHead(nn.Module):
    def __init__(
        self,
        network_spec,
        in_channels,
        append_stats,
        num_out=1,
        h_size=1000,
        dropout=0.0,
        lnorm=False,
        pool_mode="HNP",
        sigmoid=False,
    ):
        super().__init__()
        head_layers = []
        pool_cls = POOL_DICT[pool_mode]
        head_layers.extend([pool_cls(network_spec), nn.Flatten(start_dim=-2)])
        num_pooled_outs = in_channels * pool_cls.get_num_outs(
            network_spec
        ) + StatFeaturizer.get_num_outs(network_spec) * int(append_stats)
        head_layers.append(nn.Linear(num_pooled_outs, h_size))
        for i in range(2):
            if lnorm:
                head_layers.append(nn.LayerNorm(h_size))
            head_layers.append(nn.ReLU())
            if dropout > 0:
                head_layers.append(nn.Dropout(p=dropout))
            head_layers.append(nn.Linear(h_size, h_size if i == 0 else num_out))
        if sigmoid:
            head_layers.append(nn.Sigmoid())
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        return self.head(x)


# InpEncTypes = Optional[Union[Type[GaussianFourierFeatureTransform], Type[Pointwise]]]
class InvariantNFN(NormalizingModule):
    """Invariant hypernetwork. Outputs a scalar."""

    def __init__(
        self,
        # network_spec: NetworkSpec,
        layer_layout,
        hchannels,
        mode="HNP",
        normalize=False,
        in_channels=1,
        d_out=10,
        dropout=0.0,
    ):
        super().__init__(normalize=normalize)
        layers = []
        prev_channels = in_channels

        # 1d -> hidden_dim
        inp_enc = GaussianFourierFeatureTransform(in_channels)
        layers.append(inp_enc)
        prev_channels = inp_enc.out_channels

        # pos emb
        pos_enc = IOSinusoidalEncoding(layer_layout)
        layers.append(pos_enc)
        prev_channels = pos_enc.num_out_chan(prev_channels)

        for num_channels in hchannels:
            layers.append(
                MODE2LAYER[mode](
                    layer_layout, in_channels=prev_channels, out_channels=num_channels
                )
            )
            # if lnorm is not None:
            #     layers.append(LN_DICT[lnorm](layer_layout, num_channels))
            layers.append(TupleOp(nn.ReLU()))
            # if feature_dropout > 0:
            #     layers.append(ChannelDropout(feature_dropout))
            prev_channels = num_channels
        self.nfnet_features = nn.Sequential(*layers)
        self.pool = POOL_DICT[mode]()
        prev_channels = prev_channels * 2 * (len(layer_layout) - 1)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(prev_channels, 1000),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1000, d_out),
        )
        self.rearrange_weights = Rearrange("b i o c -> b c o i")
        self.rearrange_bias = Rearrange("b i o -> b o i")

    def forward(self, params):
        params = (
            (self.rearrange_weights(w) for w in params[0]),
            (self.rearrange_bias(b) for b in params[1]),
        )
        params = WeightSpaceFeatures(*params)
        features = self.nfnet_features(self.preprocess(params))
        features = self.pool(features).flatten(-2)
        return self.head(features)


class StatNet(NormalizingModule):
    """Outputs a scalar."""

    def __init__(
        self,
        network_spec: NetworkSpec,
        h_size,
        dropout=0.0,
        sigmoid=False,
        normalize=False,
    ):
        super().__init__(normalize=normalize)
        activations = [nn.Sigmoid()] if sigmoid else []
        self.hypernetwork = nn.Sequential(
            StatFeaturizer(),
            nn.Flatten(start_dim=-2),
            nn.Linear(StatFeaturizer.get_num_outs(network_spec), h_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_size, 1),
            *activations
        )

    def forward(self, params):
        return self.hypernetwork(self.preprocess(params))


class TransferNet(nn.Module):
    def __init__(
        self,
        # network_spec,
        layer_layout,
        hidden_chan,
        hidden_layers,
        gfft,
        iosinemb,
        # inp_enc_cls: InpEncTypes=None,
        # pos_enc_cls: Optional[Type[IOSinusoidalEncoding]]=None,
        mode="full",
        # lnorm=False,
        out_scale=0.01,
        dropout=0,
    ):
        super().__init__()
        layers = []
        in_channels = 1
        inp_enc = GaussianFourierFeatureTransform(**gfft)
        layers.append(inp_enc)
        in_channels = inp_enc.out_channels

        # pos emb
        pos_enc = IOSinusoidalEncoding(layer_layout, **iosinemb)
        layers.append(pos_enc)
        in_channels = pos_enc.num_out_chan(in_channels)

        layer_cls = MODE2LAYER[mode]
        layers.append(
            layer_cls(layer_layout, in_channels=in_channels, out_channels=hidden_chan)
        )
        layers.append(TupleOp(nn.ReLU()))
        if dropout > 0:
            layers.append(TupleOp(nn.Dropout(dropout)))
        for _ in range(hidden_layers - 1):
            layers.append(
                layer_cls(
                    layer_layout, in_channels=hidden_chan, out_channels=hidden_chan
                )
            )
            layers.append(TupleOp(nn.ReLU()))
        layers.append(layer_cls(layer_layout, in_channels=hidden_chan, out_channels=1))
        layers.append(LearnedScale(layer_layout, out_scale))
        self.hnet = nn.Sequential(*layers)

        self.rearrange_weights = Rearrange("b i o c -> b c o i")
        self.rearrange_bias = Rearrange("b i o -> b o i")

    def forward(self, params):
        params = (
            (self.rearrange_weights(w) for w in params[0]),
            (self.rearrange_bias(b) for b in params[1]),
        )
        params = WeightSpaceFeatures(*params)
        out = self.hnet(params)
        return (
            [o.permute(0, 3, 2, 1) for o in out.weights],
            [o.permute(0, 2, 1) for o in out.biases],
        )

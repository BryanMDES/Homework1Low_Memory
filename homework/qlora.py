from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, group_size=group_size)
        self.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.linear_dtype = torch.float32

        self.lora_dim = lora_dim
        self.lora_alpha = float(lora_dim)  
        self.scaling = self.lora_alpha / self.lora_dim

        self.lora_A = torch.nn.Parameter(torch.empty(lora_dim, in_features, dtype=torch.float32))
        self.lora_B = torch.nn.Parameter(torch.empty(out_features, lora_dim, dtype=torch.float32))

        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in_dtype = x.dtype
        base = super().forward(x.to(self.linear_dtype))
        x32 = x.to(torch.float32)
        delta = (x32 @ self.lora_A.t()) @ self.lora_B.t()
        y = base + delta * self.scaling
        return y.to(x_in_dtype)

class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            self.model = torch.nn.Sequential(QLoRALinear(channels, channels, lora_dim=lora_dim, group_size=group_size),torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim=lora_dim, group_size=group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim=lora_dim, group_size=group_size),)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net

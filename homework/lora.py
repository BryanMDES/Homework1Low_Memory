from pathlib import Path

import torch
import torch.nn.functional as F

from .bignet import BigNet, LayerNorm  # noqa: F401
from .half_precision import HalfLinear

class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework

        Hint: You can use the HalfLinear class as a parent class (it makes load_state_dict easier, names match)
        Hint: Remember to initialize the weights of the lora layers
        Hint: Make sure the linear layers are not trainable, but the LoRA layers are
        """
        super().__init__(in_features, out_features, bias)

        # TOD
        # Keep the LoRA layers in float32
        self.lora_dim = int(lora_dim)
        self.scale = 1.0
        self.lora_a = torch.nn.Linear(in_features, self.lora_dim, bias=False, dtype=torch.float32)
        self.lora_b = torch.nn.Linear(self.lora_dim, out_features, bias=False, dtype=torch.float32)
        torch.nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.lora_b.weight)
        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO:
        base = super().forward(x)
        x32 = x.to(torch.float32)
        delta = self.lora_b(self.lora_a(x32)) * self.scale

        return base + delta.to(x.dtype)

def _replace_linears_with_lora(module: torch.nn.Module, lora_dim: int) -> None:
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear) and not isinstance(child, HalfLinear):
            new = LoRALinear(child.in_features, child.out_features, lora_dim=lora_dim, bias=(child.bias is not None))
            setattr(module, name, new)
        elif isinstance(child, HalfLinear) and not isinstance(child, LoRALinear):
            new = LoRALinear(child.in_features, child.out_features, lora_dim=lora_dim, bias=(child.bias is not None))
            new.weight.data.copy_(child.weight.data)
            if (child.bias is not None) and (new.bias is not None):
                new.bias.data.copy_(child.bias.data)
            setattr(module, name, new)
        else:
            _replace_linears_with_lora(child, lora_dim)


class LoraBigNet(BigNet):
    """
    BigNet with LoRA adapters on every linear layer.
    Base weights are frozen/half via HalfLinear parent.
    """
    def __init__(self, lora_dim: int = 32):
        super().__init__()                 # builds the real BigNet (keys match checkpoint)
        _replace_linears_with_lora(self, lora_dim)

        # keep LayerNorm in fp32 for stability
        for m in self.modules():
            if isinstance(m, LayerNorm):
                m.to(torch.float32)
                m.requires_grad_(False)

def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net

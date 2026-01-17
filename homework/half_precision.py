from pathlib import Path

import torch
import torch.nn.functional as F

#from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .bignet import BigNet, LayerNorm

class HalfLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """
        Implement a half-precision Linear Layer.
        Feel free to use the torch.nn.Linear class as a parent class (it makes load_state_dict easier, names match).
        Feel free to set self.requires_grad_ to False, we will not backpropagate through this layer.
        """
        super().__init__(in_features, out_features, bias=bias)

        self.weight.data = self.weight.data.to(torch.float16)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(torch.float16)

        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hint: Use the .to method to cast a tensor to a different dtype (i.e. torch.float16 or x.dtype)
        # The input and output should be of x.dtype = torch.float32
        #Hacer
        x16 = x.to(torch.float16)
        w = self.weight.detach()
        b = self.bias.detach() if self.bias is not None else None
        y16 = F.linear(x16, w, b) 
        return y16.to(x.dtype) 

def _replace_linears(module: torch.nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear) and not isinstance(child, HalfLinear):
            new = HalfLinear(child.in_features, child.out_features, bias=(child.bias is not None))
            setattr(module, name, new)
        else:
            _replace_linears(child) 


class HalfBigNet(BigNet):
    """
    A BigNet where all weights are in half precision. Make sure that the normalization uses full
    precision though to avoid numerical instability.
    """
    def __init__(self):
        super().__init__()
        _replace_linears(self)

        for m in self.modules():
            if isinstance(m, LayerNorm):
                m.to(torch.float32)

def load(path: Path | None) -> HalfBigNet:
    # You should not need to change anything here
    # PyTorch can load float32 states into float16 models
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net

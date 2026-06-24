import torch
from torch import nn, Tensor

class Swiglu(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
    ): 
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nonlinearity = nn.SiLU()
        self.projection = nn.Linear(in_features, out_features*2, bias=False)
        with torch.no_grad():
            self.projection.weight[:out_features].zero_()
        
    def forward(self, x: Tensor) -> Tensor:
        a, b = self.projection(x).chunk(2, dim=-1)
        return self.nonlinearity(a) * b
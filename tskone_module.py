import torch
from torch import Tensor
import torch.nn as nn

class TSKONE(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self):
        super(TSKONE, self).__init__()

    def forward(self, input: Tensor, config: Tensor) -> Tensor:
        return tskone(input, config)

def tskone(input: Tensor, config: Tensor) -> Tensor:
    
    # tskone calculation, input in mT, output in dM
     
    return torch.where(config<2,-0.03828571*torch.square(input)+0.44128571*input+0.003,
                         -0.04857143*input**2+0.53471429*input-0.42914286)

class TSKONEout(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self):
        super(TSKONEout, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return tskone_out(input)

def tskone_out(input: Tensor) -> Tensor:
    
    # tskone calculation, input in mT, output in dM
    
    # if torch.sum(torch.abs(input)>4.5):
    #     print('out')
    return torch.abs(-0.02035931*torch.square(input) + 0.37077425*input)
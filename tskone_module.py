import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple
import math
import pandas as pd

class TSKONE(nn.Module):

    def __init__(self):
        super(TSKONE, self).__init__()

    def forward(self, input: Tensor, config: Tensor) -> Tensor:
        return tskone(input, config)

class TSKONEout(nn.Module):

    def __init__(self):
        super(TSKONEout, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return tskone_out(input)

class CTEXTgen(nn.Module):

    def __init__(self):
        super(CTEXTgen, self).__init__()
    
    def forward(self,input:Tensor,ctextdet:Tensor) -> Tensor:
        return ctext(input,ctextdet)

def tskone(input: Tensor, config: Tensor) -> Tensor:
    
    # tskone calculation, input in mT, output in dM
     
    return torch.where(config<2,-0.03828571*torch.square(input)+0.44128571*input+0.003,
                         -0.04857143*input**2+0.53471429*input-0.42914286)

def tskone_out(input: Tensor) -> Tensor:
    
    # tskone calculation, input in mT, output in dM
    
    return torch.abs(-0.02035931*torch.square(input) + 0.37077425*input)

def ctext(input: Tensor, ctextdet: Tensor) -> Tuple[Tensor,Tensor]:

    x = input
    context1 = torch.ones_like(x)
    context2 = 2*torch.ones_like(x)
    context = torch.where(ctextdet < 1,context1,context2)
    
    return x, context
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

    return torch.where(config<0.5,-0.03828571*torch.square(input)+0.44128571*input+0.003,
                         -0.04857143*input**2+0.53471429*input-0.42914286) - 1

def tskone_out(input: Tensor) -> Tensor:
    
    # tskone calculation, input in mT, output in dM
    x = torch.abs(input)
    return -3.46620416e-3*torch.pow(x,3) - 2.18491891e-12*torch.pow(x,2) + 3.44760852e-1*x

def ctext(input: Tensor, ctextdet: Tensor) -> Tuple[Tensor,Tensor]:

    x = input
    context1 = torch.ones_like(x)
    context2 = 2*torch.ones_like(x)
    context = torch.where(ctextdet < 1,context1,context2)
    
    return x, context
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from utils.Tensorhelp import *
import math




expo=3
MPO1=MPO(expo)
MPO2=MPO(expo)

MPO1.permutationMPO(2,5)
print(MPO1.r)
I=np.eye(2**expo)
T=getTensor_forMP0(I,expo)

MPO2.permutationMPO()
A=MPO1.reTensor()
print(A)
MPO1.MPOMP0(MPO2)

A=MPO1.reTensor()
print(A)
print(MPO1.r)
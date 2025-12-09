import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from utils.Tensorhelp import *
from utils.read_data import read_amira

expo=18

path=r"N:\Code\MPS\MPS_Paths\data\2D_VortexFlow_ETH\0500.am"
path2=r"N:\Code\MPS\MPS_Paths\data\2D_VortexFlow_ETH\hit.1024.02241.h5"
data, info=read_amira(path)

x=np.linspace(0,1,512)
y=x
X,Y=np.meshgrid(x,y)
print(np.shape(X))
t=500
u0=data[t,:,:,0]
v0=data[t,:,:,1]
low=np.zeros(expo,dtype=int)
high=np.ones(expo,dtype=int)
U0=np.sqrt(u0**2+v0**2)

T=getTensor_forMPS(U0,expo)
MPS1=MPS(expo)
MPS1.truncated_l(T,1E-3)
MPS1.quicksort(low,high)
A=MPS1.reTensor()
A=np.reshape(A,(2**9,2**9))

plt.figure()
plt.pcolormesh(X,Y,A)
plt.show()

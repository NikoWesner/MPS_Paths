import numpy as np
from utils.Tensorhelp import *
import matplotlib.pyplot as plt
from Optimization import *

def sinconst_alpha(x,y,alpha):

    fun= np.sin(np.cos(alpha)*x+np.sin(alpha)*y)
    return fun
def sin_alpha(x,y,alpha,f):
    fun=np.cos(alpha)*np.sin(f*x)+np.sin(alpha)*np.sin(f*y)
    return fun


def cost_function(matrix,expo):
    # Example: minimize sum of squared differences between neighbors
    return np.abs(np.linalg.det(matrix))*(2**expo)

def swap_random_entries(matrix):
    a, b = np.random.randint(0, matrix.size, 2)
    m = matrix.flatten()
    m[a], m[b] = m[b], m[a]
    return m.reshape(matrix.shape)

expo=8
eexpo=2**expo

alpha=np.pi/4
e=1E-6

x=np.linspace(0,2*np.pi,eexpo)
y=x

X,Y=np.meshgrid(x,y)
# fun=sinconst_alpha(X,Y,alpha=alpha)
for i in range(1,10):
    alpha=np.pi/i

    fun3=(sin_alpha(X,Y,alpha,1)+sin_alpha(X,Y,alpha,4)+sin_alpha(X,Y,alpha,8))/np.sqrt(3)


    plt.figure()
    plt.pcolormesh(X,Y,fun3,cmap='viridis')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    fun=fun3.flatten()


    fun=np.reshape(fun,(eexpo,eexpo))
    fun2=fun

    MPS1=MPS(2*expo)
    MPS2=MPS(2*expo)

    MPS_Size=2*np.ones(2*expo, dtype=int)
    T_assemble=np.reshape(fun2,MPS_Size)
    permuter=np.zeros((2*expo),dtype=int)
    for i in range(expo):
        permuter[2*i]=i
        permuter[2*i+1]=expo+i
    permuter=permuter.tolist()
    permuter=tuple(permuter)
    T=T_assemble.transpose(permuter)

    e=1E-5
    MPS1.truncated_l(T,e)
    MPS2.truncated_l(T_assemble,e)
    print(f"Chi{MPS1.r}")
    print(f"Chi{MPS2.r}")
    print(f"Chi1:{np.sum(MPS1.r)}")
    print(f"Chi2:{np.sum(MPS2.r)}")



# plt.show()




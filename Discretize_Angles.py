import numpy as np
import matplotlib.pyplot as plt
from utils.Tensorhelp import *

def sinconst_alpha(x,y,alpha):

    fun= np.sin(np.cos(alpha)*x+np.sin(alpha)*y)
    return fun

def find_mesh_indices(X, Y, point):
    """
    Find the indices (iy, ix) in the meshgrid (X, Y) 
    that are closest to the given point [x0, y0].
    
    Parameters:
        X, Y : 2D arrays from np.meshgrid(x, y)
        point : list or tuple [x0, y0]
        
    Returns:
        (iy, ix) : tuple of indices
    """
    x0, y0 = point

    # Find closest x and y in the meshgrid
    ix = np.argmin(np.abs(X[0, :] - x0))   # X is constant along rows
    iy = np.argmin(np.abs(Y[:, 0] - y0))   # Y is constant along columns

    return ix, iy

expo=4
eexpo=2**expo
alpha=np.pi/4

xend=2*np.pi
yend=xend

x=np.linspace(0,xend,eexpo)
y=x

X,Y=np.meshgrid(x,y)
T=sinconst_alpha(X,Y,alpha=alpha)

scale=np.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2)


n=np.array([np.cos(alpha),np.sin(alpha)])*scale
t=np.array([-np.sin(alpha),np.cos(alpha)])*scale


X0=0.0
Y0=0.0

x0=np.array([X0,Y0])

T_assemble=np.array([T[0,0]])

i_nx=0
i_ny=0
rounding=1E-5
path=[0,0]
for i in range(eexpo**2-1):
    x_updated=x0+t
    if x_updated[0]<-rounding or x_updated[1]>yend+rounding:
        i_nx+=1
        if i_nx<eexpo:
            x_updated[0]=X[0,i_nx]
            x_updated[1]=0.0
        else:
            if i_ny<eexpo-1:
                i_ny+=1
                x_updated[0]=X[0,eexpo-1]
                x_updated[1]=Y[i_ny,0]
    ix,iy=find_mesh_indices(X,Y,x_updated)
    T_assemble=np.append(T_assemble,T[ix,iy])
    x0=x_updated
    path=np.append(path,[ix,iy],axis=0)
print(T_assemble)
print(T)
print(np.shape(T_assemble))
print(f'scale:{scale}')

plt.figure()
plt.pcolormesh(x,y,T,cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')

MPS1=MPS(2*expo)

MPS_Size=2*np.ones(2*expo, dtype=int)
T_assemble=np.reshape(T_assemble,MPS_Size)
e=1E-4

MPS1.truncated_l(T_assemble,e)
print(MPS1.r)
xp = path[0::2]
yp = path[1::2]

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.5, eexpo + 0.5)
ax.set_ylim(-0.5, eexpo + 0.5)
ax.set_xticks(np.arange(0, eexpo + 1, 1))
ax.set_yticks(np.arange(0, eexpo + 1, 1))
ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

# Draw the path
ax.plot(xp, yp, 'o-', color='red', linewidth=2, markersize=6, label='Path')

# Label points
for i, (xi, yi) in enumerate(zip(xp, yp)):
    ax.text(xi + 0.1, yi + 0.1, f'{i+1}', color='blue')

ax.set_xlabel('X index')
ax.set_ylabel('Y index')
ax.set_title('Mesh with Path')
ax.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()







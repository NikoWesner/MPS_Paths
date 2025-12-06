import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math

def LinearDependency(v1,v2):
    p1=np.linalg.norm(v1,2)
    p2=np.linalg.norm(v2,2)
    dot=np.dot(v1,v2)

    costheta=dot/(p1*p2)

    return np.abs(1-costheta)

A=np.array([[1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16]])
A=np.reshape(A,(4,4))

print(A)
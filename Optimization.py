import numpy as np

def cost_function(matrix,expo):
    # Example: minimize sum of squared differences between neighbors
    return np.abs(np.linalg.det(matrix))*(2**expo)
def cost_function2(matrix,e):
    m,n=np.shape(matrix)
    p=min(m,n)
    s=np.zeros(p)
    r=np.zeros(n)
    c=np.zeros(m)
    sig=s
    a=np.diag(matrix)
    for i in range(n):
        r[i]=np.sum(np.abs(matrix[i,:]))-np.abs(a[i])
    for j in range(m):
        c[j]=np.sum(np.abs(matrix[:,j]))-np.abs(a[j])
    s=np.maximum(r,c)
    sig=a+s
    u=np.maximum(np.sqrt(a**2+a*r+c**2*0.25)+c/2,np.sqrt(a**2+a*c+0.25*r**2)+r/2)
    sig=u
    e0=np.min(sig)
    return -np.sum(sig<e),e0
def cost_function3(matrix,e):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return -np.sum(S<e)


def swap_random_entries(matrix):
    a, b = np.random.randint(0, matrix.size, 2)
    m = matrix.flatten()
    m[a], m[b] = m[b], m[a]
    return m.reshape(matrix.shape)


def Optimization1(matrix,e):
    print('initializing Optimization')
    p=np.linalg.norm(matrix,'fro')
    matrix=matrix/p
    best = matrix.copy()
    best_cost = cost_function3(matrix,e)
    T = 1.0  # initial temperature
    allcosts=[]
    for i in range(10000):
        candidate = swap_random_entries(matrix=matrix)
        new_cost = cost_function3(candidate,e)
        allcosts.append(new_cost)
        print(i,new_cost)
        if new_cost < best_cost:
            matrix = candidate
            best_cost = new_cost
            best = candidate.copy()
        T *= 0.999  # cooling
    print('finished Optimization')
    return best*p, best_cost

import numpy as np
from SVDTest import MPS1
from utils.Tensorhelp import *
from collections import defaultdict
from utils.Optimization import applyOperator_minMaxTT
from utils.read_data import read_amira
import matplotlib.pyplot as plt

def find_perm_sqrt_with_internal_swap(p1_input):
    n = len(p1_input)
    # We copy the input so we can modify the cycles internally
    p1_modified = list(p1_input)
    visited = [False] * n
    required_swaps = []
    
    # 1. First pass: Identify cycles and find the "lonely" even ones
    cycles = []
    for i in range(n):
        if not visited[i]:
            curr, cycle = i, []
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = p1_input[curr]
            cycles.append(cycle)
            
    length_map = defaultdict(list)
    for c in cycles:
        length_map[len(c)].append(c)
        
    # 2. THE FIX: Break lonely even cycles into two odd cycles
    for length, group in length_map.items():
        if length % 2 == 0 and len(group) % 2 != 0:
            lonely_cycle = group.pop()
            # We swap the targets of the first and last elements in the cycle
            # This mathematically splits the cycle into an (L-1) cycle and a (1) cycle
            idx_a, idx_b = lonely_cycle[0], lonely_cycle[-1]
            
            p1_modified[idx_a], p1_modified[idx_b] = p1_modified[idx_b], p1_modified[idx_a]
            
            # This is the single swap needed to restore the even cycle later
            required_swaps.append((idx_a, idx_b))

    # 3. Second pass: Decompose the modified P1 (all cycles now rootable)
    visited = [False] * n
    final_cycles = []
    for i in range(n):
        if not visited[i]:
            curr, cycle = i, []
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = p1_modified[curr]
            final_cycles.append(cycle)

    # 4. Solve for P2
    final_map = defaultdict(list)
    for c in final_cycles:
        final_map[len(c)].append(c)

    p2 = [0] * n
    for length, group in final_map.items():
        if length % 2 != 0:
            step = (length + 1) // 2
            for cycle in group:
                for i in range(length):
                    p2[cycle[i]] = cycle[(i + step) % length]
        else:
            # Pair the even cycles that were not lonely
            for i in range(0, len(group), 2):
                c1, c2 = group[i], group[i+1]
                for j in range(length):
                    p2[c1[j]] = c2[j]
                    p2[c2[j]] = c1[(j + 1) % length]

    return p2, required_swaps
def permutation_vector_to_matrix(p_vector):
    """
    Converts a permutation vector to a Permutation Matrix P.
    p_vector: list or array where p[i] = destination of index i
    """
    n = len(p_vector)
    # Create an n x n identity matrix
    # Then reorder its rows (or columns) based on the vector
    P = np.zeros((n, n), dtype=int)
    
    # Each row i gets a 1 at column p_vector[i]
    P[np.arange(n), p_vector] = 1
    
    return P
def decompose_to_independent_layers(p_vector):
    n = len(p_vector)
    visited = [False] * n
    layer1_swaps = []
    layer2_swaps = []

    # 1. Decompose into cycles first
    for i in range(n):
        if not visited[i]:
            curr, cycle = i, []
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = p_vector[curr]
            
            # 2. Decompose each cycle into two layers of independent swaps
            L = len(cycle)
            if L < 2: continue
            
            # Layer 1: Swaps (cycle[i], cycle[L-1-i])
            for k in range(L // 2):
                layer1_swaps.append((cycle[k], cycle[L - 1 - k]))
            
            # Layer 2: Swaps (cycle[i], cycle[L-i]) starting from index 1
            # This "un-rotates" the first swap to achieve the permutation
            for k in range(1, (L + 1) // 2):
                layer2_swaps.append((cycle[k], cycle[L - k]))

    return layer1_swaps, layer2_swaps
def block_sort(arr, x):
    n = len(arr)
    if n % x != 0:
        raise ValueError("Array length must be divisible by block size x.")
    
    # Create the destination array and the permutation vector
    sorted_arr = np.zeros_like(arr)
    p_vector = np.zeros(n, dtype=int)
    
    for i in range(0, n, x):
        # 1. Extract the current block
        block = arr[i : i + x]
        
        # 2. Get the sorting indices for THIS block
        # argsort gives the positions that would sort the block
        local_indices = np.argsort(block)
        
        # 3. Fill the sorted array
        sorted_arr[i : i + x] = block[local_indices]
        
        # 4. Map the local permutation to global indices
        # i + local_indices are where the values CAME from
        # np.arange(i, i+x) are where they are GOING
        p_vector[i + local_indices] = np.arange(i, i + x)
            
    # Calculate the inverse permutation
    p_inverse = [0] * n
    for i, val in enumerate(p_vector):
        p_inverse[val] = i
        
    return sorted_arr, p_vector, p_inverse
def create_mandelbrot_fractal(width=512, height=512, max_iter=100, x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5):
    """
    Generates a Mandelbrot fractal on a grid of specified dimensions.

    :param width: The width of the grid (image).
    :param height: The height of the grid (image).
    :param max_iter: The maximum number of iterations before a point is considered stable (in the set).
    :param x_min: The minimum real coordinate for the viewing window.
    :param x_max: The maximum real coordinate for the viewing window.
    :param y_min: The minimum imaginary coordinate for the viewing window.
    :param y_max: The maximum imaginary coordinate for the viewing window.
    :return: A 2D NumPy array representing the fractal image data (iteration counts).
    """

    # 1. Create the grid of complex numbers
    # Create arrays for the real (x) and imaginary (y) parts
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    
    # Use broadcasting to create the complex grid C = x + iy
    # Transpose y for correct shape (width, height) complex plane
    C = x[None, :] + 1j * y[:, None] 
    
    # Initialize Z (the iterative variable) and M (the output iteration count map)
    Z = np.zeros(C.shape, dtype=np.complex64)
    M = np.full(C.shape, max_iter, dtype=np.int32)

    # 2. Iterate the Mandelbrot function: Z(n+1) = Z(n)^2 + C
    for i in range(max_iter):
        # Find points that are not yet divergent (magnitude squared < 4)
        # Using Z.real**2 + Z.imag**2 < 4 avoids calculating the expensive sqrt
        not_escaped = (Z.real**2 + Z.imag**2) < 4.0
        
        # Stop iterating if all points have escaped
        if not np.any(not_escaped):
            break

        # Only update the Z values for points that haven't escaped
        Z[not_escaped] = Z[not_escaped]**2 + C[not_escaped]
        
        # Update the iteration count M for points that *just* escaped in this step
        escaped = np.logical_and(not_escaped, (Z.real**2 + Z.imag**2) >= 4.0)
        M[escaped] = i
        
    return M

expo=18
e=1E-8

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

U0=np.sqrt(u0**2+v0**2)
U0=U0.flatten()
p=np.argsort(U0)

M=create_mandelbrot_fractal()
M=M.flatten()
MPS1=MPS(expo)
s1=[]
# for i in range(5,19):
#     T,p,pinv=block_sort(M,2**i)
#     T1=getTensor_forMPS(T,expo)
#     MPS1.truncated_l(T1,1E-4)
#     print(MPS1.r)
#     s=np.sum(np.array(MPS1.r))

#     s1=np.append(s1,s)
# print("s1",s)
# plt.figure()
# plt.plot(np.arange(5,19),np.log2(s1))
# plt.title("TT-ranks, blocksearch")
# plt.xlabel("exponent of blocksize")
# plt.ylabel("added TT-ranks")


expo=18
e=1E-8

MPO1=MPO(expo)
MPO1.eye()
MPO2=MPO(expo)
used_pairs = set()
results = []
n_iterations = 12
r1=[]
all_data=np.zeros((14,expo-1))
for l in range(5,19):    
    limit = 2 **l
    for i in range(n_iterations):
        while True:
            # Generate two random integers
            p, j = np.random.randint(0, limit, size=2)
            
            # Check conditions:
            # 1. p != j
            # 2. (p, j) hasn't occurred before
            # 3. (j, p) hasn't occurred before (optional: include if order doesn't matter)
            if p != j and (p, j) not in used_pairs:
                used_pairs.add((p, j))
                # used_pairs.add((j, p)) # Uncomment if you treat (1,2) the same as (2,1)
                results.append((p, j))
                break

    for idx, (p, j) in enumerate(results):
        MPO2.unit_permutationMPO(p,j)
        MPO1.MPO_add(MPO2)
        if idx%4==0:
            MPO1.reTruncate(e)
    MPO1.reTruncate(e)
    all_data[l-5,:]=MPO1.r
    print(MPO1.r)
    r1=np.append(r1,np.sum(MPO1.r))
print(all_data)
plt.figure()
plt.plot(np.arange(5,19),np.log2(r1))
plt.title("combi")
plt.xlabel("exponent of permutation range")
plt.ylabel("added up TT-ranks")

plt.figure(figsize=(10, 6))
# shading='auto' or 'nearest' handles the grid alignment
mesh = plt.pcolormesh(np.arange(expo-1),np.arange(5,19), all_data, shading='auto', cmap='viridis')
plt.colorbar(mesh, label='Rank')
plt.xlabel('core')
plt.ylabel('exponent of permutaiton range')
plt.show()
# l1, l2 = decompose_to_independent_layers(p)

# print("l1",np.shape(l1))
# print("l2",np.shape(l2))

# # print(f"Layer 1 (Independent Swaps): {l1}")
# # print(f"Layer 2 (Independent Swaps): {l2}")


# MPO1=MPO(expo)
# MPO1.eye()
# MPO2=MPO(expo)
# for idx, (p, j) in enumerate(l1):
#     MPO2.unit_permutationMPO(p,j)
#     MPO1.MPO_add(MPO2)
#     if idx%4==0:
#         MPO1.reTruncate(e)
#         print(MPO1.r)
#     print(idx)
# MPO1.reTruncate(e)
# print(MPO1.r)
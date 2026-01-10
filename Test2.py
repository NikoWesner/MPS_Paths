import numpy as np
from utils.Tensorhelp import *
from utils.Analysis import minmaxMPS
from utils.read_data import read_amira, read_h5_data
RED = "\033[91m"
RESET = "\033[0m"

def color_matrix(M, value=1):
    """
    Prints matrix M with all entries equal to `value` colored red.
    """
    M = np.asarray(M)
    rows, cols = M.shape
    
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            x = M[i, j]
            if x == value:
                row_str += f"{RED}{x}{RESET} "
            else:
                row_str += f"{x} "
        print(row_str)



def permutation_matrix_from_vectors(y, x):
    """
    Compute permutation matrix P such that y = P @ x.
    Both x and y must be 1D arrays containing the same elements.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(x)
    P = np.zeros((n, n), dtype=int)

    # Build a mapping from value -> list of indices (handles duplicates)
    from collections import defaultdict
    pos = defaultdict(list)
    for i, val in enumerate(x):
        pos[val].append(i)

    # Fill permutation matrix
    for i, val in enumerate(y):
        if not pos[val]:
            raise ValueError(f"Value {val} in y is not found in x")
        j = pos[val].pop(0)     # take matching index from x
        P[i, j] = 1

    return P 


def permuted_bitstrings(n_bits, order):
    """
    Generate all bitstrings of length n_bits and reorder their bit positions.

    Args:
        n_bits : number of bits
        order  : a permutation list of indices, e.g. [0,2,1]

    Returns:
        bits_permuted : array of shape (2^n_bits, n_bits)
        dec_permuted  : decimal values of the permuted bitstrings
    """
    # 1. all decimals
    dec = np.arange(2**n_bits)

    # 2. convert to bitstrings
    bits = np.array([list(f"{d:0{n_bits}b}") for d in dec], dtype=int)

    # 3. apply permutation to columns
    bits_permuted = bits[:, order]

    # 4. convert back to decimal
    dec_permuted = bits_permuted.dot(1 << np.arange(n_bits-1, -1, -1))

    return bits_permuted, dec_permuted

def permuter(expo):
    permuter=np.zeros((2*expo),dtype=int)
    for i in range(expo):
        permuter[2*i]=i
        permuter[2*i+1]=expo+i
    permuter=permuter.tolist()
    return permuter

import numpy as np

def extract_coefficients(M):
    """
    Extract coefficients from a matrix composed of 2×2 identity blocks.
    M must be of shape (2R, 2C).

    Returns:
        A matrix C of shape (R, C) where each entry C[i,j] is the scalar
        multiplying the 2×2 identity block in M.
    """
    M = np.asarray(M)
    rows, cols = M.shape

    if rows % 2 != 0 or cols % 2 != 0:
        raise ValueError("Matrix dimensions must be multiples of 2")

    R, C = rows // 2, cols // 2
    coeffs = np.zeros((R, C))

    for i in range(R):
        for j in range(C):
            block = M[2*i:2*i+2, 2*j:2*j+2]

            # Extract scalar: since block = a*I, we can read block[0,0]
            coeffs[i, j] = block[0, 0]

    return coeffs
def match_values_double(T):
    """
    For each value in t1, find the closest available value in t20.
    Each value in t20 is used at most once.

    Returns:
        t2: array of chosen values from t20 (same shape as t1)
        idx_used: indices in t20 that were chosen
    """
    size=np.size(T)
    T=np.reshape(T,size)
    perm = np.argsort(T)
    perm=perm.flatten()
    # sorted array
    T=T[perm]
    T=T[perm]

    # compute inverse permutation
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))

    # reconstruct original
    n = len(T)
    half = n // 2
    t1=T[0::2]
    t2=T[1::2]

    # t2[0:1000]=t1[0:1000]
    # t1=T[0:half]
    # t2=T[half:]
    return t1,t2,inv
def minmaxMPS2(T,expo,e):
    size=np.size(T)
    T=np.reshape(T,(size))
    t1,t2,inv=match_values_double(T)
    T_sorted=np.append(t1,t2)
    T=getTensor_forMPS(T_sorted,2*expo)
    MPS2=MPS(2*expo)
    MPS2.truncated_l(T,e)
    MPS2.getSize()
    r=MPS2.r
    s=MPS2.size
    return MPS2, r,s,inv
import numpy as np
from collections import defaultdict

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

# --- Usage & Verification ---
# p1 = [1, 2, 0, 4, 3, 6, 5, 8, 9, 10, 7, 11]
# p2, correction_swaps = find_perm_sqrt_with_internal_swap(p1)

# print(f"P2: {p2}")
# print(f"Swaps to apply at the very end: {correction_swaps}")

# # Final verification
# p2_arr = np.array(p2)
# res = p2_arr[p2_arr] # P2 squared
# print(res)
# for i, j in correction_swaps:
#     res[i], res[j] = res[j], res[i] # Apply the single fix

# print("Matches original P1?", np.array_equal(res, p1))

expo=9
e=1E-8

MPO1=MPO(expo)
MPO1.eye()
MPO2=MPO(expo)
for i in range(12):
    p,j=np.random.randint(0,2*expo,size=2)
    MPO2.permutationMPO(p,j)
    MPO1.MPOMPO(MPO2)
    if i%2==0:
        MPO1.reTruncate(e)
        print(MPO1.r)

MPO1.reTruncate(e)
print(MPO1.r)
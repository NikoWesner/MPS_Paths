import numpy as np

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


np.set_printoptions(threshold=8*8*8*8)


expo=4

x=np.arange(4**expo)
yb,y=permuted_bitstrings(2*expo,permuter(expo))
print(y)
P= permutation_matrix_from_vectors(y,x)
M=extract_coefficients(P)

color_matrix(M[0:32,0:32])
print(np.shape(M))

G1=np.zeros((2,2,4))
G2=np.zeros((4,2,2,4))
G3=np.zeros((4,2,2,4))
G4=np.zeros((4,2,2))

uno=np.array([[1,0],[0,0]])
due=np.array([[0,1],[0,0]])
tre=np.array([[0,0],[1,0]])
qua=np.array([[0,0],[0,1]])

G1[:,:,0]=uno
G1[:,:,1]=due
G1[:,:,2]=tre
G1[:,:,3]=qua

G2[0,:,:,0]=uno
G2[1,:,:,2]=uno

G2[0,:,:,1]=tre
G2[1,:,:,3]=tre


G2[2,:,:,0]=due
G2[3,:,:,2]=due


G2[3,:,:,3]=qua
G2[2,:,:,1]=qua




G3[0,:,:,0]=uno
G3[1,:,:,1]=uno

G3[2,:,:,0]=tre
G3[3,:,:,1]=tre


G3[0,:,:,2]=due
G3[1,:,:,3]=due


G3[2,:,:,2]=qua
G3[3,:,:,3]=qua




G4[0,:,:]=uno
G4[1,:,:]=due
G4[2,:,:]=tre
G4[3,:,:]=qua

T=np.tensordot(G1,G2,axes=(-1,0))
T=np.tensordot(T,G2,axes=(-1,0))
T=np.tensordot(T,G3,axes=(-1,0))
T=np.tensordot(T,G3,axes=(-1,0))
T=np.tensordot(T,G4,axes=(-1,0))

T=T.transpose((0,2,4,6,8,10,1,3,5,7,9,11))
T=T.reshape(64,64)


# T1=np.zeros((2,2,4))
# T2=np.zeros((4,2,2))
# T3=np.zeros((4,2,2,4))
# T1[:,:,0]=uno
# T1[:,:,1]=due
# T2[0,:,:]=uno
# T2[1,:,:]=due
# T3[0,:,:,0]=uno
# T3[1,:,:,1]=due

# T=np.tensordot(T1,T3,axes=(-1,0))
# T=np.tensordot(T,T2,axes=(-1,0))
# T=T.transpose()
# T=T.reshape(8,8)
# print(T)
print("Here")

color_matrix(T[0:32,0:32],value=1)
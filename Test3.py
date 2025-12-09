import numpy as np

def assemble_pattern_arrays_revised(N):
    """
    Generates two NumPy arrays of length 2^(N-1) * 2^N based on specific patterns.

    Args:
        N (int): The factor used in array length calculation (2^(N-1) * 2^N).
                 For the example 256*512, N=9.

    Returns:
        tuple: A tuple containing (array_one, array_two).
    """

    block_length = 2**N  # 2^N
    num_blocks = 2**(N - 1)  # 2^(N-1)
    total_length = num_blocks * block_length

    # --- Array One Assembly (Left Shifted) ---
    # Pattern: [1, 2, ..., 2^N], then [2, 3, ..., 2^N, 1], and so on.
    
    # Base array: [1, 2, ..., 2^N]
    base_array = np.arange(1, block_length + 1, dtype=np.int32)
    
    array_one = np.empty(total_length, dtype=np.int32)

    for i in range(num_blocks):
        # The shift amount is 'i' positions to the **left**.
        # In NumPy, a negative shift value in np.roll performs a left shift.
        shifted_array = np.roll(base_array, shift=-i)
        
        # Place the shifted block into the final array
        start_index = i * block_length
        end_index = (i + 1) * block_length
        array_one[start_index:end_index] = shifted_array

    # --- Array Two Assembly (Unchanged) ---
    # Pattern: [1], [2, 2], [3, 3, 3], ..., [n, n, ..., n] (n times), 
    #          followed by fill-in value 2^N.
    
    array_two_parts = []
    current_length = 0
    fill_value = block_length
    
    for n in range(1, total_length + 2):
        # Length of the current sequence of 'n's
        block = np.full(n, n, dtype=np.int32)
        
        if current_length + n > total_length:
            break
            
        array_two_parts.append(block)
        current_length += n
        
    # Concatenate the parts generated so far
    array_two = np.concatenate(array_two_parts)
    
    # Fill the remainder with the maximum possible value, 2^N (block_length)
    remainder_length = total_length - len(array_two)

    if remainder_length > 0:
        final_array_two = np.concatenate([array_two, 
                                          np.full(remainder_length, fill_value, dtype=np.int32)])
    else:
        final_array_two = array_two
        
    return array_one, final_array_two




N = 9
v1, v2 = assemble_pattern_arrays_revised(N)

v3=v1[::-1]



A1=np.zeros((2,2**(N-1)*2**N))
A1[0,:]=v1
A1[1,:]=v3
A2=np.zeros((2,2**(N-1)*2**N))
A2[0,:]=v2
A2[1,:]=v2



p1=np.linalg.norm(A1,"fro")
A1=A1/p1
p2=np.linalg.norm(A2,"fro")
A2=A2/p2

U,S1,Vt=np.linalg.svd(A1,full_matrices=False)
U,S2,Vt=np.linalg.svd(A2,full_matrices=False)
print("First")
print(S1)
print(S2)


A1=np.reshape(A1,(2**N,2**N))
A2=np.reshape(A2,(2**N,2**N))

U,S1,Vt=np.linalg.svd(A1,full_matrices=False)
U,S2,Vt=np.linalg.svd(A2,full_matrices=False)
print("Middle")
print(S1)
print(S2)
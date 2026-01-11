import numpy as np
import matplotlib.pyplot as plt
from utils.read_data import read_amira,read_h5_data
from matplotlib.animation import FuncAnimation
from utils.Tensorhelp import *
from Optimization import Optimization1
from hilbertcurve.hilbertcurve import HilbertCurve
from matplotlib.colors import LogNorm, PowerNorm
import math
import matplotlib.cm as cm
from utils.Analysis import fetch4plots,sort_by_pairs,variation_of_lasttwo,minmaxMPS
from scipy.sparse import diags
from collections import defaultdict, Counter

def create_tridiagonal(n, low, mid, high):
    """
    Creates an n x n tridiagonal matrix with constant values.
    low: value for the sub-diagonal
    mid: value for the main diagonal
    high: value for the super-diagonal
    """
    # diags takes a list of arrays and their offsets
    # [low, mid, high] at offsets [-1, 0, 1]
    # np.full(length, value) creates the constant value arrays
    offsets = [-1, 0, 1]
    data = [
        np.full(n - 1, low), 
        np.full(n, mid), 
        np.full(n - 1, high)
    ]
    
    # Returning as a dense array for your small-scale tests,
    # but .toarray() can be removed if you want a sparse object.
    return diags(data, offsets).toarray()

def create_random_permutation(n):
    """
    Creates an n x n random permutation matrix.
    """
    # Create an identity matrix
    I = np.eye(n)
    # Generate a random permutation of indices
    p = np.random.permutation(n)
    # Reorder the rows (or columns) based on the permutation
    return I[p]

def sinconst_alpha(x,y,alpha):

    fun= np.sin(np.cos(alpha)*x+np.sin(alpha)*y)
    return fun
def sinconst_k(x,y,k):
    fun=0
    for i in range(1,k+1):
        fun=fun+np.sin(i*(X+Y))*1/i**2
    return fun
def linear(x,y):

    return X+Y
def randomfunction(x,y):
    return x+y
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
def calculate_2d_holder_exponent(data_2d, num_scales=5):
    """
    Estimates the global Hölder Exponent (alpha) of a 2D function (e.g., a 512x512 grid) 
    using the Structure Function Method.

    The method analyzes the scaling relationship between the average squared difference 
    (structure function) of pixel values and the distance (scale) between them:
    E[|data(x+r) - data(x)|^2] ~ r^(2*H) where H is the Hurst exponent.
    For functions, the scaling exponent alpha is H.

    :param data_2d: A 2D NumPy array (e.g., 512x512) representing the function values.
    :param num_scales: The number of different 'r' distances (scales) to sample.
    :return: The estimated global scaling exponent (alpha), which corresponds to the 
             Hölder exponent for continuous functions.
    """
    
    # 1. Input Validation and Setup
    if not isinstance(data_2d, np.ndarray) or data_2d.ndim != 2:
        raise ValueError("Input data_2d must be a 2D NumPy array.")
    
    height, width = data_2d.shape
    
    # Maximum scale to sample (e.g., a quarter of the minimum dimension)
    max_r = min(height, width) // 4
    
    # Select scales (r) for analysis, spaced logarithmically
    scales = np.unique(np.logspace(0, np.log10(max_r), num_scales, dtype=int))
    
    # Filter out scales that are too small (r=0) or too large
    scales = scales[scales > 0]
    
    # Arrays to store the calculated structure function values
    structure_function_values = []
    
    # 2. Calculate the 2nd-order Structure Function
    for r in scales:
        # Calculate the square of the difference for horizontal and vertical shifts
        
        # Horizontal shift (r, 0)
        diff_h = (data_2d[:, r:] - data_2d[:, :-r])**2
        
        # Vertical shift (0, r)
        diff_v = (data_2d[r:, :] - data_2d[:-r, :])**2
        
        # Calculate the mean of the squared differences (Structure Function S_2(r))
        # The mean is taken over all possible pairs at distance 'r'
        S_2_r = np.mean(np.concatenate([diff_h.flatten(), diff_v.flatten()]))
        
        if S_2_r > 0:
            structure_function_values.append(S_2_r)
        
    # Filter out any scales that resulted in zero mean difference (unlikely for fractals)
    scales = scales[:len(structure_function_values)]
    
    if len(scales) < 2:
        print("Warning: Not enough unique scales found for meaningful regression.")
        return 0.0

    # 3. Log-Log Regression to Find the Exponent
    # The relationship is log(S_2(r)) = (2*alpha) * log(r) + log(C)
    
    log_r = np.log(scales)
    log_S_2 = np.log(structure_function_values)

    # Perform linear regression: fit Y = slope * X + intercept
    # The slope is 2 * alpha
    try:
        slope, intercept = np.polyfit(log_r, log_S_2, 1)
        
        # The Hurst/Hölder exponent (alpha) is half the slope
        holder_exponent = slope / 2.0
        
        return holder_exponent
    
    except np.linalg.LinAlgError:
        print("Error during linear regression. Check input data variability.")
        return 0.0
def synthesize_2d_holder_function(target_alpha, size=512):
    """
    Synthesizes a 2D function (Fractional Gaussian Field) of size x size 
    with a target global Hölder Exponent (alpha).

    Uses the Fourier Filtering Method based on the relationship between
    the scaling exponent (alpha, or Hurst exponent H) and the power spectrum exponent (beta).

    :param target_alpha: The desired Hölder exponent (alpha). Must be in the range (0, 2).
                         - alpha = 0.5: White noise (rough, no memory).
                         - alpha = 1.0: Brown noise/fBm (smooth, high memory).
                         - alpha > 1.0: Very smooth/differentiable.
    :param size: The dimension of the square output grid (e.g., 512).
    :return: A 2D NumPy array of size x size representing the synthesized function.
    """
    
    N = size
    
    # 1. Define the Spectral Exponent (beta) based on the Hölder Exponent (alpha)
    # For Fractional Brownian Motion (fBm), the relationship is:
    # Power Spectrum S(k) ~ 1 / k^beta
    # where beta = 2 * alpha + D - 2
    # and D is the topological dimension (D=2 for a 2D grid).
    #
    # Simplifying for D=2: beta = 2 * alpha + 2 - 2 = 2 * alpha
    beta = 2 * target_alpha
    
    # 2. Create the Frequency Grid (k)
    
    # Generate spatial frequencies (k_x and k_y)
    # Use rfftfreq to handle the symmetry of the real FFT, but we'll use the full grid later
    # for cleaner geometry.
    freq_x = np.fft.fftfreq(N)
    freq_y = np.fft.fftfreq(N)
    
    # Use meshgrid to create 2D arrays of the frequency coordinates
    k_x, k_y = np.meshgrid(freq_x, freq_y)
    
    # Calculate the radial distance k = |k| = sqrt(k_x^2 + k_y^2)
    # The zero-frequency component (k=0) must be handled to prevent division by zero.
    k = np.sqrt(k_x**2 + k_y**2)
    
    # Set the zero-frequency component (at (0, 0)) to a very small, non-zero value
    # or exclude it from the power law calculation. Setting it to 1.0 is safe for the power law denominator.
    k[0, 0] = 1.0 
    
    # 3. Create the Power Spectrum Filter (P(k))
    # The filter P(k) is the magnitude of the desired spectral coefficients.
    # P(k) = 1 / k^(beta / 2)
    P_k = 1.0 / (k**(beta / 2.0))
    
    # Restore the zero-frequency component (DC component) to zero after creating the filter
    # This ensures the resulting field has zero mean.
    P_k[0, 0] = 0.0

    # 4. Generate Random Noise in the Frequency Domain
    # Create a 2D array of complex Gaussian random numbers.
    # The real and imaginary parts must be independent standard Gaussians.
    noise_fft = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    
    # 5. Apply the Filter and Inverse Fourier Transform
    
    # Apply the spectral filter: F_filtered = F_noise * P_k
    filtered_fft = noise_fft * P_k
    
    # Perform the inverse 2D Fourier Transform
    f_x_y = np.fft.ifft2(filtered_fft)
    
    # Since the input (P_k) was chosen symmetrically (real-valued power spectrum)
    # the inverse transform should result in a real-valued function.
    # We take the real part, discarding the tiny imaginary parts due to numerical error.
    synthesized_function = np.real(f_x_y)

    # Normalize the output to the range [0, 1] for typical visualization/use
    synthesized_function -= synthesized_function.min()
    if synthesized_function.max() > 0:
        synthesized_function /= synthesized_function.max()
        
    return synthesized_function
def compute_and_visualize_fft2d(f,name="function"):
    """
    Computes the 2D FFT of a function f and visualizes 
    the original function and its magnitude spectrum.
    
    Args:
        f (numpy.ndarray): The 2D input array (function).
    """
    # --- 1. Compute the 2D FFT ---
    # The fft2 function computes the unshifted 2D FFT
    F = np.fft.fft2(f)

    # --- 2. Shift the Zero-Frequency Component (DC component) ---
    # The zero-frequency component is typically in the corner (0,0).
    # fftshift moves it to the center, which is standard for visualization.
    F_shifted = np.fft.fftshift(F)

    # --- 3. Compute the Magnitude Spectrum ---
    # The magnitude spectrum is the absolute value of the complex numbers.
    # The log scale is often used to visualize the large dynamic range of 
    # the frequency components.
    magnitude_spectrum = 20 * np.log(np.abs(F_shifted) + 1e-9) 
    # Adding a small epsilon (1e-9) to avoid log(0)
    
    # --- 4. Visualization ---
    plt.figure(figsize=(12, 5))

    # Original Function (Spatial Domain)
    plt.subplot(1, 2, 1)
    plt.imshow(f, cmap='gray')
    plt.title(f'Original 2D Function {name}')
    plt.colorbar(label='Amplitude')
    plt.axis('off')

    # Magnitude Spectrum (Frequency Domain)
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='magma')
    plt.title('Magnitude Spectrum $|F(\omega_x, \omega_y)|$ (Log Scale)')
    plt.colorbar(label='Log Magnitude')
    plt.axis('off')

    plt.tight_layout()
def find_perm_sqrt(p1):
    n = len(p1)
    visited = [False] * n
    cycles = []

    # 1. Decompose P1 into cycles
    for i in range(n):
        if not visited[i]:
            curr = i
            cycle = []
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = p1[curr]
            cycles.append(cycle)

    # 2. Group cycles by their lengths
    from collections import defaultdict
    length_map = defaultdict(list)
    for cycle in cycles:
        length_map[len(cycle)].append(cycle)

    p2 = [0] * n
    print("Even length cycles (length: count):", 
        {k: len(v) for k, v in length_map.items() if k % 2 == 0})
    # 3. Process cycles to find the root
    for length, group in length_map.items():
        if length % 2 != 0:
            # Odd cycle: Square root is P1^((length+1)/2)
            # This effectively "un-skips" the elements
            step = (length + 1) // 2
            for cycle in group:
                for i in range(length):
                    p2[cycle[i]] = cycle[(i + step) % length]
        else:
            # Even cycle: Must pair two cycles of the same length
            if len(group) % 2 != 0:
                raise ValueError(f"No square root: odd number of cycles of even length {length}")
            
            for i in range(0, len(group), 2):
                c1, c2 = group[i], group[i+1]
                # Interleave two cycles of length L into one cycle of length 2L
                for j in range(length):
                    p2[c1[j]] = c2[j]
                    p2[c2[j]] = c1[(j + 1) % length]

    return p2
def find_perm_sqrt_cheating(p1):
    n = len(p1)
    visited = [False] * n
    cycles = []

    # 1. Decompose into cycles
    for i in range(n):
        if not visited[i]:
            curr, cycle = i, []
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = p1[curr]
            cycles.append(cycle)

    from collections import defaultdict
    length_map = defaultdict(list)
    for cycle in cycles:
        length_map[len(cycle)].append(cycle)

    p2 = [0] * n

    # 2. The Cheat: Handle Even cycles
    for length in list(length_map.keys()):
        if length % 2 == 0:
            group = length_map[length]
            
            # If we have an odd number of even cycles, "steal" one from elsewhere
            if len(group) % 2 != 0:
                # OPTION: Try to find another cycle of the same length to pair with
                # CHEAT: If no pair exists, we simply force it to stay as is 
                # (This creates a tiny error in P2*P2 but prevents a crash)
                lonely_cycle = group.pop()
                for i in range(length):
                    p2[lonely_cycle[i]] = lonely_cycle[i] # Force to Identity
            
            # Pair up the remaining even cycles
            for i in range(0, len(group), 2):
                c1, c2 = group[i], group[i+1]
                for j in range(length):
                    p2[c1[j]] = c2[j]
                    p2[c2[j]] = c1[(j + 1) % length]
                    
        else:
            # Odd cycles work as normal
            step = (length + 1) // 2
            for cycle in length_map[length]:
                for i in range(length):
                    p2[cycle[i]] = cycle[(i + step) % length]

    return p2
def find_perm_sqrt_with_correction(p1):
    n = len(p1)
    visited = [False] * n
    cycles = []

    # 1. Decompose into cycles
    for i in range(n):
        if not visited[i]:
            curr, cycle = i, []
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = p1[curr]
            cycles.append(cycle)

    from collections import defaultdict
    length_map = defaultdict(list)
    for cycle in cycles:
        length_map[len(cycle)].append(cycle)

    p2 = list(range(n)) # Start with Identity
    correction = list(range(n)) # Elements that need fixing

    # 2. Process Cycles
    for length, group in length_map.items():
        if length % 2 != 0:
            # Odd cycles: Perfect square root
            step = (length + 1) // 2
            for cycle in group:
                for i in range(length):
                    p2[cycle[i]] = cycle[(i + step) % length]
        else:
            # Even cycles: Pair them if possible
            while len(group) >= 2:
                c1 = group.pop()
                c2 = group.pop()
                for j in range(length):
                    p2[c1[j]] = c2[j]
                    p2[c2[j]] = c1[(j + 1) % length]
            
            # If one is left over, it's "Unsolvable"
            if len(group) == 1:
                lonely_cycle = group.pop()
                # We leave p2 as identity for these indices
                # and put the original mapping into 'correction'
                for i in range(length):
                    # Elements stay still in p2, but we note the needed swap
                    correction[lonely_cycle[i]] = p1[lonely_cycle[i]]

    return p2, correction
def find_perm_sqrt_efficient_cheat(p1):
    n = len(p1)
    visited = [False] * n
    p2 = list(range(n))
    required_swaps = []
    
    # 1. Decompose into cycles
    for i in range(n):
        if not visited[i]:
            curr, cycle = i, []
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = p1[curr]
            
            L = len(cycle)
            if L % 2 != 0:
                # ODD: Perfect square root (0 swaps)
                step = (L + 1) // 2
                for j in range(L):
                    p2[cycle[j]] = cycle[(j + step) % L]
            else:
                # EVEN: The "One-Swap" Cheat
                # We can't solve it perfectly, but we can solve L-1 elements
                # and leave a single swap (cycle[0], cycle[L//2]) for the end.
                
                # Use a modified step for even length
                step = L // 2
                for j in range(L):
                    p2[cycle[j]] = cycle[(j + step) % L]
                
                # This specific P2 squared results in p1 PLUS a swap
                # between the elements halfway across the cycle.
                required_swaps.append((cycle[0], cycle[step]))

    return p2, required_swaps
def verify_solution(p1, p2, swaps):
    """
    p1: Original permutation list
    p2: The calculated square-root permutation list
    swaps: List of tuples (i, j)
    """
    n = len(p1)
    # Convert to numpy for fast vectorized application
    p2_arr = np.array(p2)
    
    # 1. Apply P2 twice: data = p2[p2[identity]]
    # This simulates the matrix multiplication B * B
    identity = np.arange(n)
    after_p2_twice = p2_arr[p2_arr[identity]]
    
    # 2. Apply the simple swaps to the result
    # We work on a copy to avoid mutating the intermediate result
    final_result = after_p2_twice.copy()
    for i, j in swaps:
        # Simple swap logic
        final_result[i], final_result[j] = final_result[j], final_result[i]
    
    # 3. Compare with P1
    is_correct = np.array_equal(final_result, p1)
    
    # Calculate error if not correct
    if not is_correct:
        diff_count = np.sum(final_result != p1)
        print(f"❌ Verification Failed! {diff_count} elements do not match.")
    else:
        print("✅ Verification Successful! (P2^2 + Swaps) == P1")
        
    return is_correct
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


path=r"N:\Code\MPS\MPS_Paths\data\2D_VortexFlow_ETH\0500.am"
path2=r"N:\Code\MPS\MPS_Paths\data\2D_VortexFlow_ETH\hit.1024.02241.h5"
data, info=read_amira(path)
# data2=read_h5_data(path2)

# print(np.shape(data2["u"]))


x=np.linspace(0,1,512)
y=x
X,Y=np.meshgrid(x,y)
print(np.shape(X))
t=500
u0=data[t,:,:,0]
v0=data[t,:,:,1]

U0=np.sqrt(u0**2+v0**2)
# fun=sinconst_k(X,Y,150)
# fun2=randomfunction(X,Y)
# fun3=synthesize_2d_holder_function(1.2)
# fun4=sinconst_alpha(X,Y,0)
# M=create_mandelbrot_fractal()





# plt.figure()
# plt.pcolormesh(X,Y,M)
# plt.title("SinSum")
# plt.figure()
# plt.pcolormesh(X,Y,fun2)
# plt.title("X+Y")
# plt.figure()
# plt.pcolormesh(X,Y,fun3)
# plt.title("Synth")
# plt.figure()
# plt.pcolormesh(X,Y,fun4)
# plt.title("Sin45")
# plt.figure()
# plt.pcolormesh(X,Y,U0)
# plt.title("Velocity")
# plt.figure()
# plt.pcolormesh(X,Y,M)
# plt.title("Mandelbrot")

# alpha1=calculate_2d_holder_exponent(U0)
# alpha2=calculate_2d_holder_exponent(fun)
# alpha3=calculate_2d_holder_exponent(fun2)
# alpha4=calculate_2d_holder_exponent(fun3)
# print(alpha1,alpha2,alpha3)


# fetch4plots(U0,expo,e,"Velocity")
# fetch4plots(fun,expo,e,"SinSum")
# fetch4plots(fun2,expo,e,"X+Y")
# fetch4plots(fun3,expo,e,"Synth")
# fetch4plots(fun4,expo,e,"Sin45")
# fetch4plots(M,expo,e,"Mandelbrot")
p1=U0.flatten().argsort()
p2, correction_swaps = find_perm_sqrt_with_internal_swap(p1)

# # print(f"P2: {p2}")
# print(f"Swaps to apply at the very end: {correction_swaps}")
# # Final verification
# p2_arr = np.array(p2)
# res = p2_arr[p2_arr] # P2 squared
# print(res)
# for i, j in correction_swaps:
#     res[i], res[j] = res[j], res[i] # Apply the single fix

# print("Matches original P1?", np.array_equal(res, p1))
expo=12
e=1E-3
T=np.random.random((2**expo,1)).flatten()
A=create_tridiagonal(2**expo, -1, 2, -1)
MPO1=MPO(expo)
MPS1=MPS(expo)
T1=getTensor_forMPS(T,expo)
A=getTensor_forMP0(A,expo)
MPS1.truncated_l(T1,e)
MPO1.truncated_l(A,e)

MPS2,inv=minmaxMPS(T1,expo,e)
T2=np.sort(T)
T3=getTensor_forMPS(T2,expo)
MPS3=MPS(expo)
MPS3.truncated_l(T3,e)


print(MPS1.r)
print(MPS2.r)
print(MPS3.r)

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
from utils.Analysis import fetch4plots,sort_by_pairs,variation_of_lasttwo
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
fun=sinconst_k(X,Y,150)
fun2=randomfunction(X,Y)
fun3=synthesize_2d_holder_function(1.2)
fun4=sinconst_alpha(X,Y,0)
M=create_mandelbrot_fractal()

expo=9
eexpo=2**expo
e=2E-3





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

MPS1=MPS(2*expo)
# T1=variation_of_lasttwo(U0)
T1=U0
T1=getTensor_forMPS(T1,2*expo)
print(np.shape(T1))
MPS1.truncated_l(T1,e)
MPS1.getSize()
T1=MPS1.reTensor()
T1=T1.reshape(2**expo*2**expo,1)
T1[-2]=0
T1[-1]=0
U0_flat=U0.reshape(2**expo*2**expo,1)
U0_flat[-2]=0
U0_flat[-1]=0
print(np.linalg.norm(T1-U0_flat)/np.linalg.norm(U0_flat),"Relative Error after variation_of_lasttwo")
T1=T1.reshape(2**expo,2**expo)
plt.figure()
plt.pcolormesh(X,Y,T1)
plt.title("MPS Velocity Approximation")
print(MPS1.r,MPS1.size)



plt.show()
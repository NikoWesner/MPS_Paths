import numpy as np
import matplotlib.pyplot as plt
from utils.read_data import read_amira
from matplotlib.animation import FuncAnimation
from utils.Tensorhelp import *
from Optimization import Optimization1
from hilbertcurve.hilbertcurve import HilbertCurve
from matplotlib.colors import LogNorm, PowerNorm
import math
import matplotlib.cm as cm


def LinearDependency(v1,v2):
    p1=np.linalg.norm(v1,2)
    p2=np.linalg.norm(v2,2)
    dot=np.dot(v1,v2)
    costheta=dot/(p1*p2)
    return 1-np.abs(costheta)
def permuter(expo):
    permuter=np.zeros((2*expo),dtype=int)
    for i in range(expo):
        permuter[2*i]=i
        permuter[2*i+1]=expo+i
    permuter=permuter.tolist()
    return permuter
def match_values(T):
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
    T=np.sort(T)

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
def interleave_array(arr):
    arr = np.asarray(arr)
    n = len(arr)
    assert n % 2 == 0, "Array length must be even"
    
    half = n // 2
    first = arr[:half]
    second = arr[half:]
    
    # Interleave first and second
    interleaved = np.empty(n, dtype=arr.dtype)
    interleaved[0::2] = first
    interleaved[1::2] = second
    
    return interleaved
def index_to_coords(index, N):
    """
    Convert 1D index to 2D coordinates in 2^N x 2^N grid
    """
    size = 2**N
    i = index // size
    j = index % size
    return i, j
def HilbertMPS(T,expo,e):
    size=np.size(T)
    T=np.reshape(T,(2**expo,2**expo))
    N=2**(expo*2)
    hilbert_curve = HilbertCurve(expo, 2)
    points=hilbert_curve.points_from_distances(list(range(N)))
    T_hilbert=[]
    for [x,y] in points:
        T_hilbert.append(T[x,y])
    T_hilbert=np.array(T_hilbert)
    MPSH=MPS(expo*2)
    T_hilbert=getTensor_forMPS(T_hilbert,expo*2)
    MPSH.truncated_l(T_hilbert,e)
    MPSH.getSize()
    r=MPSH.r
    s=MPSH.size

    return MPSH,r,s
def minmaxMPS(T,expo,e):
    size=np.size(T)
    T=np.reshape(T,(size))
    t1,t2,inv=match_values(T)
    T_sorted=np.append(t1,t2)
    T=getTensor_forMPS(T_sorted,2*expo)
    MPS2=MPS(2*expo)
    MPS2.truncated_l(T,e)
    MPS2.getSize()
    r=MPS2.r
    s=MPS2.size
    d1=LinearDependency(t1,t2)
    return MPS2, r,s,inv,d1
def XYMPS(T,expo,e):
    size=np.size(T)
    T=np.reshape(T,(size))
    T=getTensor_forMPS(T,2*expo)
    MPS1=MPS(2*expo)
    MPS1.truncated_l(T,e)
    MPS1.getSize()
    r=MPS1.r
    s=MPS1.size
    return MPS1,r,s
def GourianovMPS(T,expo,e):
    size=np.size(T)
    T=np.reshape(T,(size))
    T=getTensor_forMPS(T,2*expo)
    permuter1=permuter(expo)
    T=T.transpose(permuter1)
    MPSG=MPS(2*expo)
    MPSG.truncated_l(T,e)
    MPSG.getSize()
    r=MPSG.r
    s=MPSG.size
    return MPSG,r,s
def plotGourianov(data,title):
    NUM_ROWS = len(data)

    # DYNAMIC CALCULATION: MAX_COLS is the max length of any inner array.
    MAX_COLS = max(len(row) for row in data) 
    print(f"Calculated MAX_COLS: {MAX_COLS}") # Output: 4

    DENSE_MATRIX = np.full((NUM_ROWS, MAX_COLS), np.nan)
    EPSILON = 1e-6 

    # Populate the dense matrix
    for i, row_values in enumerate(data):
        # Ensure values are positive for log color scale
        row_values_for_color = np.array([val if val > 0 else EPSILON for val in row_values])
        
        num_segments = len(row_values)
        
        # Fill the first 'num_segments' columns
        DENSE_MATRIX[i, :num_segments] = row_values_for_color

    # --- 3. Colorbar and Normalization Setup ---

    # Log10 scale for color, based on LaTeX point meta min/max
    LOG_DATA = np.log10(DENSE_MATRIX)
    VMIN = -10 # point meta min
    VMAX = 0   # point meta max

    cmap = plt.get_cmap('jet') 

    # --- 4. Dynamic X-axis Calculation ---

    # Determine the maximum power of 2 needed to cover MAX_COLS
    if MAX_COLS == 0:
        max_power = 0
    else:
        # We need to cover MAX_COLS, so the upper tick should be >= MAX_COLS
        max_power = int(np.ceil(np.log2(MAX_COLS)))

    plot_max_x = 2**max_power
    # Ensure the X-axis range includes all MAX_COLS indices (1 to MAX_COLS)
    # The X-axis extends up to 2^max_power, e.g., if MAX_COLS=4, max_power=2, plot_max_x=4

    # X-axis ticks will be powers of 2 up to max_power
    xticks = [2**i for i in range(max_power + 1)] 
    xticklabels = [f'$2^{int(math.log2(t))}$' for t in xticks]

    # --- 5. Plotting using imshow (Matrix Plot equivalent) ---

    fig, ax = plt.subplots(figsize=(5, 8))

    # Use imshow to plot the matrix (equivalent to pgfplots matrix plot*)
    # The extent must now use the dynamic MAX_COLS and plot_max_x
    im = ax.imshow(
        LOG_DATA,
        cmap=cmap,
        vmin=VMIN,
        vmax=VMAX,
        aspect='auto',
        interpolation='none',
        # Extent defines the log-scaled area. The data is mapped to 1...MAX_COLS
        # We set the extent to [0.5, MAX_COLS + 0.5] for the indices to center the cells.
        # The X-axis scaling in Step 6 will handle the logarithmic visual representation.
        extent=[0.5, MAX_COLS + 0.5, NUM_ROWS + 0.5, 0.5] 
    )

    # --- 6. Customizing the Plot (Matching LaTeX Axes) ---

    # X-axis (Logarithmic base 2)
    ax.set_xscale('log', base=2)

    # Set the visible X-axis limits to the dynamically calculated power of 2
    ax.set_xlim(0.75, plot_max_x) # xmin=0.75

    # Set ticks to the dynamic powers of 2
    ax.set_xticks(xticks) 
    ax.set_xticklabels(xticklabels) 
    ax.set_xlabel('$\chi$')

    # Y-axis (Reversed, 1-based labels)
    yticks = np.arange(1, NUM_ROWS + 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylabel('$k$-th bond index')
    ax.set_ylim(NUM_ROWS + 0.5, 0.5) # y dir=reverse (Row 1 at top)

    # Grid style
    for val in xticks:
        ax.axvline(val, color='gray', linestyle='-', linewidth=1.0, alpha=0.3, zorder=0)

    ax.set_title(title)
    plt.tight_layout()

    # Colorbar (Matching log-scale labels)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Log Value ($10^n$)')

    tick_vals = np.array([-30, -25, -20, -15, -10, -5, 0])
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f'$10^{{{int(t)}}}$' for t in tick_vals])

    # plt.savefig("2d_pgfplots_heatmap_dynamic_max.png")
def getAllMPS_PlotFormat(fun,expo):
    #fun1
    fun1=getTensor_forMPS(fun,2*expo)
    #fun2
    size=np.size(fun)
    T=np.reshape(fun,(size))
    t1,t2,inv=match_values(T)
    T_sorted=np.append(t1,t2)
    fun2=getTensor_forMPS(T_sorted,2*expo)
    #funG
    T=np.reshape(T,(size))
    T=getTensor_forMPS(T,2*expo)
    permuter1=permuter(expo)
    funG=T.transpose(permuter1)
    #funH
    size=np.size(T)
    T=np.reshape(T,(2**expo,2**expo))
    N=2**(expo*2)
    hilbert_curve = HilbertCurve(expo, 2)
    points=hilbert_curve.points_from_distances(list(range(N)))
    T_hilbert=[]
    for [x,y] in points:
        T_hilbert.append(T[x,y])
    T_hilbert=np.array(T_hilbert)
    MPSH=MPS(expo*2)
    funH=getTensor_forMPS(T_hilbert,expo*2)

    return fun1,fun2,funG,funH
def plotGourianov_ax(ax, data, title="Log-Scaled Heatmap", log_thresholds=[-2,-3,-4,-5,-6]):
    """
    Plots the data as a log-scaled heatmap onto a given Axes object (ax), 
    including a set of lines tracing multiple logarithmic threshold values.
    
    Args:
        ax (matplotlib.axes.Axes): The target axes object for plotting.
        data (list of lists): The data, where inner lists represent row segments.
        title (str): The title for the subplot.
        log_thresholds (list of float, optional): A list of log10 values (e.g., [-2, -5])
                                                  to draw boundary lines at.
        
    Returns:
        matplotlib.image.AxesImage: The image object, required for the colorbar, 
                                    or None if plotting failed.
    """


    # --- 1. Data Preprocessing ---
    NUM_ROWS = len(data)
    MAX_COLS = max(len(row) for row in data)
    DENSE_MATRIX = np.full((NUM_ROWS, MAX_COLS), np.nan)
    EPSILON = 1e-11 

    for i, row_values in enumerate(data):
        row_values_for_color = np.array([val if val > 0 else EPSILON for val in row_values])
        DENSE_MATRIX[i, :len(row_values)] = row_values_for_color

    # --- 2. Normalization & Plotting Setup ---
    LOG_DATA = np.log10(DENSE_MATRIX)
    VMIN = -10
    VMAX = 0   
    cmap = plt.get_cmap('jet') 

    # --- 3. X-axis & Plotting ---
    max_power = int(np.ceil(np.log2(MAX_COLS))) if MAX_COLS > 0 else 0
    plot_max_x = 2**max_power
    if plot_max_x == 0: plot_max_x = 1

    xticks = [2**i for i in range(max_power + 1)] 
    xticklabels = [f'$2^{int(math.log2(t))}$' for t in xticks]

    im = ax.imshow(
        LOG_DATA,
        cmap=cmap,
        vmin=VMIN,
        vmax=VMAX,
        aspect='auto',
        interpolation='none',
        extent=[0.5, MAX_COLS + 0.5, NUM_ROWS + 0.5, 0.5] 
    )

    # --- 4. Customizing the Plot (Axes, Grid, Title) ---
    ax.set_xscale('log', base=2)
    ax.set_xlim(0.75, plot_max_x)
    ax.set_xticks(xticks) 
    ax.set_xticklabels(xticklabels) 
    
    yticks = np.arange(1, NUM_ROWS + 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylim(NUM_ROWS + 0.5, 0.5) 

    for val in xticks:
        ax.axvline(val, color='gray', linestyle='-', linewidth=1.0, alpha=0.3, zorder=0)

    ax.set_title(title, fontsize=12)
    
    # --- 5. Multiple Threshold Line Logic ---
    if log_thresholds:
        # Define a list of colors/styles for the thresholds
        line_styles = ['-', '--', ':', '-.'] # Solid, dashed, dotted, dash-dot
        colors = cm.Dark2(np.linspace(0, 1, len(log_thresholds)))
        
        # Sort thresholds so the lines are plotted consistently (largest value first, drawn underneath)
        log_thresholds.sort(reverse=True) 

        for idx, log_th in enumerate(log_thresholds):
            threshold = 10**log_th
            X_coords = []
            Y_coords = []
            
            for i, row_values in enumerate(data):
                # Default boundary is past the last segment
                boundary_index = len(row_values) + 0.5
                
                for k, value in enumerate(row_values):
                    # Check for the first segment whose value is <= threshold
                    if value <= threshold:
                        boundary_index = k + 0.5 
                        break
                
                X_coords.append(boundary_index)
                Y_coords.append(i + 1)

            ax.plot(
                X_coords, 
                Y_coords, 
                color=colors[idx], 
                linestyle=line_styles[idx % len(line_styles)],
                linewidth=2,
                label=f'$10^{{{int(log_th)}}}$',
                zorder=3 + idx
            )
        
        # Add a legend to distinguish the threshold lines
        ax.legend(title='Threshold', loc='lower left', fontsize=8, framealpha=0.7)

    return im
def getGouianovplots(MPS1,MPS2,MPSG,MPSH,fun,expo,name):
    
    fun1,fun2,funG,funH=getAllMPS_PlotFormat(fun,expo)
    data1=MPS1.GourianovPlot(fun1)
    data2=MPS2.GourianovPlot(fun2)
    dataG=MPSG.GourianovPlot(funG)
    dataH=MPSH.GourianovPlot(funH)
    all_data = [data1, data2, dataG, dataH]
    titles = ["XY", "minmax", "Multigrid", "Hilbert"]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.subplots_adjust(right=0.85)

    im_objects = []
    fig.suptitle(name, fontsize=16)
    for i in range(2):
        for j in range(2):
            data_index = i * 2 + j
            ax = axs[i, j]
            im = plotGourianov_ax(ax, all_data[data_index], titles[data_index])
            if im is not None:
                im_objects.append(im)
            
            # Post-processing for label visibility
            if i == 1:
                ax.set_xlabel('$\chi$')
            else:
                ax.set_xticklabels([])
                ax.set_xlabel('')

            if j == 0:
                ax.set_ylabel('$k$-th bond index')
            else:
                ax.set_yticklabels([])
                ax.set_ylabel('')

    # --- Add a single common colorbar ---
    if im_objects:
        cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
        # The image object from the last successful plot is used for the colorbar
        cbar = fig.colorbar(im_objects[-1], cax=cbar_ax, orientation='vertical')
        cbar.set_label('Log Value ($10^n$)')

        tick_vals = np.array([-10, -8, -6, -4, -2, 0]) # Adjusted ticks for VMIN=-10
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f'$10^{{{int(t)}}}$' for t in tick_vals])

    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
def fetch4plots(fun,expo,e,name):
    print("----------------------------")
    print(name)
    MPS1,r1,s1=XYMPS(fun,expo,e)
    MPSG,rg,sg=GourianovMPS(fun,expo,e)
    MPSH,rh,sh=HilbertMPS(fun,expo,e)
    MPS2,r2,s2,inv,d1=minmaxMPS(fun,expo,e)
    print(f"XY:{r1},{s1}")
    print(f"Gourianov:{rg},{sg}")
    print(f"Hilbert:{rh},{sh}")
    print(f"minmax:{r2},{s2}")
    getGouianovplots(MPS1,MPS2,MPSG,MPSH,fun,expo,name)
    print(f"Lineardependency:{d1}")
    print("----------------------------")   
    return


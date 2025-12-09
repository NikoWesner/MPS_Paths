import numpy as np
import re
import h5py
from typing import Dict,Union,Any

def read_amira(path):
    """
    Reads AmiraMesh (.am) files (ASCII or BINARY), scalar or vector fields.

    Returns:
        data (np.ndarray): Array of shape (nz, ny, nx, ncomp) or (ny, nx) for 2D scalar
        info (dict): Header metadata
    """

    # Read entire file in binary
    with open(path, "rb") as f:
        raw = f.read()

    # Locate the @1 marker (start of data block)
    marker = b"@1"
    idx = raw.find(marker)
    if idx < 0:
        raise ValueError("Could not find @1 marker in AmiraMesh file.")

    # Split header and raw data
    header_bytes = raw[:idx]
    data_bytes   = raw[idx + len(marker):]

    # Decode header safely
    header = header_bytes.decode("ascii", errors="ignore")

    # --- Extract dimensions ---
    dims = None
    m = re.search(r"define\s+Lattice\s+(\d+)\s+(\d+)\s+(\d+)", header)
    if m:
        dims = list(map(int, m.groups()))
    else:
        raise ValueError("No Lattice dimensions found in header.")

    nx, ny, nz = dims

    # --- Extract number of components (scalar vs vector) ---
    ncomp = 1
    m = re.search(r"float\[(\d+)\]", header)
    if m:
        ncomp = int(m.group(1))

    # --- Check ASCII or BINARY ---
    header_lower = header.lower()
    is_ascii = "ascii" in header_lower

    # ============================
    #     Load numeric data
    # ============================
    if is_ascii:
        # ASCII numeric block
        text = data_bytes.decode("ascii", errors="ignore")
        arr = np.fromstring(text, sep=" ")
    else:
        # Binary numeric block — AmiraMesh uses float32 by default
        arr = np.frombuffer(data_bytes, dtype=np.float32)

    # --- Reshape data ---
    expected_size = nx * ny * nz * ncomp
    if arr.size < expected_size:
        raise ValueError(f"Data too small. Expected {expected_size}, got {arr.size}")

    arr = arr[:expected_size]  # ignore trailing bytes
    arr = arr.reshape((nz, ny, nx, ncomp))

    # Remove last dimension if scalar
    if ncomp == 1:
        arr = arr[..., 0]

    # Flatten single-slice 2D data
    if nz == 1:
        arr = arr[0]

    # Collect metadata
    info = {
        "nx": nx, "ny": ny, "nz": nz,
        "ncomp": ncomp,
        "is_ascii": is_ascii,
        "header": header
    }

    return arr, info

def read_h5_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Reads all datasets from an HDF5 file and returns them in a dictionary.

    Args:
        file_path (str): The path to the .h5 or .hdf5 file.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are the dataset names 
                                (paths) and values are the corresponding 
                                NumPy arrays.
    """
    data = {}
    
    try:
        # Open the HDF5 file in read mode ('r')
        with h5py.File(file_path, 'r') as f:
            
            # Helper function to recursively traverse groups and find datasets
            def get_datasets(name, obj):
                # Check if the object is a dataset
                if isinstance(obj, h5py.Dataset):
                    # Read the dataset content into a NumPy array
                    data[name] = obj[()]
            
            # Visit all items in the file, calling get_datasets for each one
            f.visititems(get_datasets)
            
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the HDF5 file: {e}")
        return {}
        
    return data
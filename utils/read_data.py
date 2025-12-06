import numpy as np
import re

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


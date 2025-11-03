import matplotlib.pyplot as plt
import numpy as np

# Example data: path [x1, y1, x2, y2, ...]
data = [1, 0, 2, 1, 3, 1, 4, 2, 5, 3]

# Convert to coordinate pairs
x = data[0::2]
y = data[1::2]

# Define mesh extents (based on data range)
x_max = max(x) + 1
y_max = max(y) + 1

# Create the mesh grid
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.5, x_max + 0.5)
ax.set_ylim(-0.5, y_max + 0.5)

# Draw grid lines (mesh)
ax.set_xticks(np.arange(0, x_max + 1, 1))
ax.set_yticks(np.arange(0, y_max + 1, 1))
ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

# Draw the path
ax.plot(x, y, 'o-', color='red', linewidth=2, markersize=6, label='Path')

# Label points
for i, (xi, yi) in enumerate(zip(x, y)):
    ax.text(xi + 0.1, yi + 0.1, f'{i+1}', color='blue')

ax.set_xlabel('X index')
ax.set_ylabel('Y index')
ax.set_title('Mesh with Path')
ax.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

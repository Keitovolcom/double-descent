import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Data for the two vertical lines
# -----------------------------
# -----------------------------
# Data for the two vertical lines
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Base vertical lines (z = 0)
# -----------------------------
y_vals = np.linspace(2.5, 0.5, 100)

# line1: x=1, z=0
x1 = np.full_like(y_vals, 1.0)
z1 = np.zeros_like(y_vals)

# line2: x=2, z=0
x2 = np.full_like(y_vals, 2.5)
z2 = np.zeros_like(y_vals)

# -----------------------------
# Create 3D plot
# -----------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot base lines
ax.plot(x1, y_vals, z1, label='line1 (z=0)', linewidth=2, color='gray')
ax.plot(x2, y_vals, z2, label='line2 (z=0)', linewidth=2, color='gray')

# -----------------------------
# Points on each line
# -----------------------------
y_points = [2.00, 1.00]
colors = ['red', 'blue']
def z_curve_line1(y):
    return np.cos(np.pi * (y - 1.0))

# -----------------------------
# line2: half‑period from y=0.5→1.75, second half 1.75→2.50
# -----------------------------
split_start = 0.5      # start of first half‑period
split_mid   = 1.75     # mid-point (completes first half)
split_end   = 2.5      # end of second half
segment_len1 = split_mid - split_start   # 1.25
segment_len2 = split_end - split_mid     # 0.75

for y_p, c in zip(y_points, colors):
    ax.scatter(1.0, y_p, 0.0, color=c, s=50)  # line1
    ax.scatter(2.5, y_p, 0.0, color=c, s=50)  # line2

# -----------------------------
# Sine (cosine) curves
# -----------------------------
y_curve = np.linspace(0.5, 2.5, 400)
z_curve = np.cos(np.pi * (y_curve - 1.0))

# curve on line1 (x=1)
ax.plot(np.full_like(y_curve, 1.0), y_curve, z_curve,
        color='black', linewidth=2, label='sine_curve line1')

# curve on line2 (x=2)
# ax.plot(np.full_like(y_curve, 2.0), y_curve, z_curve,
#         color='magenta', linewidth=2, label='sine_curve line2')

# -----------------------------
# Horizontal line on line2 at z = 1
# -----------------------------
y_z1 = np.linspace(0.5, 2.5, 2)
ax.plot(np.full_like(y_z1, 2.5), y_z1, np.ones_like(y_z1),
        color='black', linestyle='-', linewidth=2, label='line2 z=1')

# -----------------------------
# Axes labels and limits
# -----------------------------
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim(0, 3)
ax.set_ylim(0.5, 2.5)
ax.set_zlim(-1.5, 1.5)


plt.savefig("test.svg")
print("test.svg")

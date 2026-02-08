import numpy as np
import matplotlib.pyplot as plt

# --- データの生成 ---
x, y = np.meshgrid(range(31), range(31), indexing="ij")
z0 = np.sin(x / 5) + np.cos(y / 5) + np.random.rand(31, 31) * 0.2
z = z0.copy()

alpha = 0.1
lam = 0.8
tol = 1e-4

# --- 反復計算 ---
for i in range(10000):
    laplacian = np.zeros_like(z)
    laplacian[1:-1, 1:-1] = (
        z[:-2, 1:-1] + z[2:, 1:-1] + z[1:-1, :-2] + z[1:-1, 2:] - 4 * z[1:-1, 1:-1]
    )
    grad = lam * laplacian - (1 - lam) * (z - z0)
    z += alpha * grad
    if np.linalg.norm(grad) < tol:
        print(f"Converged at {i}")
        break

# --- 可視化 (XY平面での比較) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Original Data (Noisy)
im1 = ax1.imshow(z0.T, origin='lower', extent=[0, 30, 0, 30], cmap='viridis')
cp1 = ax1.contour(x, y, z0, colors='white', alpha=0.5)
ax1.clabel(cp1, inline=True, fontsize=8)
ax1.set_title("Original (Noisy) - XY Plane")
fig.colorbar(im1, ax=ax1, shrink=0.8)

# Solved Data (Smooth)
im2 = ax2.imshow(z.T, origin='lower', extent=[0, 30, 0, 30], cmap='viridis')
cp2 = ax2.contour(x, y, z, colors='white', alpha=0.5)
ax2.clabel(cp2, inline=True, fontsize=8)
ax2.set_title("Solved (Smooth) - XY Plane")
fig.colorbar(im2, ax=ax2, shrink=0.8)

plt.tight_layout()
plt.show()

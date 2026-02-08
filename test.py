import numpy as np
import matplotlib.pyplot as plt

# grid
x, y = np.meshgrid(range(31), range(31), indexing="ij")

# original data
z0 = np.sin(x / 5) + np.cos(y / 5) + np.random.rand(31, 31) * 0.2
z = z0.copy()

alpha = 0.1
lam = 0.8
tol = 1e-4

for i in range(10000):
    laplacian = np.zeros_like(z)
    laplacian[1:-1, 1:-1] = (
        z[:-2, 1:-1] +
        z[2:, 1:-1] +
        z[1:-1, :-2] +
        z[1:-1, 2:] -
        4 * z[1:-1, 1:-1]
    )

    grad = lam * laplacian - (1 - lam) * (z - z0)
    z += alpha * grad

    if np.linalg.norm(grad) < tol:
        print("Converged at", i)
        break



fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(x, y, z0)
ax1.set_title("Original (noisy)")

ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(x, y, z)
ax2.set_title("Solved (equilibrium)")

plt.show()

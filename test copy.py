import numpy as np
import matplotlib.pyplot as plt

# create x, y grid
x, y = np.meshgrid(range(31), range(31), indexing="ij")

# random z values
z = np.random.rand(31, 31)

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x, y, z)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

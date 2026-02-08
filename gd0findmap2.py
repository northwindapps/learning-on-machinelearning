import numpy as np
import matplotlib.pyplot as plt

# 1. 少し複雑な地形（ノイズ多め）の生成
grid_size = 40
x_grid, y_grid = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
z0 = np.sin(x_grid / 4) + np.cos(y_grid / 4) + np.random.rand(grid_size, grid_size) * 0.5
z = z0.copy()

# 適度な平滑化（ノイズを残しつつ道筋を作る）
alpha_smooth, lam = 0.1, 0.7
for _ in range(300):
    laplacian = np.zeros_like(z)
    laplacian[1:-1, 1:-1] = (z[:-2, 1:-1] + z[2:, 1:-1] + z[1:-1, :-2] + z[1:-1, 2:] - 4 * z[1:-1, 1:-1])
    z += alpha_smooth * (lam * laplacian - (1 - lam) * (z - z0))

# 勾配の計算
gz, gx = np.gradient(z)

# 2. 複数のランダム地点からスタート
num_agents = 8
starts = np.random.uniform(2, grid_size-3, (num_agents, 2))
learning_rate = 1.5
n_steps = 60

plt.figure(figsize=(10, 8))
plt.contourf(x_grid, y_grid, z, levels=25, cmap='terrain', alpha=0.8)
plt.colorbar(label='Potential / Height')

# 各エージェントの移動
for i in range(num_agents):
    curr = starts[i].copy()
    path = [curr.copy()]
    
    for _ in range(n_steps):
        ix, iy = np.clip(curr.astype(int), 0, grid_size-2)
        grad = np.array([gz[ix, iy], gx[ix, iy]])
        
        curr -= learning_rate * grad # 勾配降下
        path.append(curr.copy())
        if np.linalg.norm(grad) < 0.01: break # 収束

    path = np.array(path)
    line, = plt.plot(path[:, 0], path[:, 1], '-', linewidth=2, alpha=0.7)
    plt.plot(path[0, 0], path[0, 1], 'o', color=line.get_color(), markersize=4)
    plt.plot(path[-1, 0], path[-1, 1], 'x', color=line.get_color(), markersize=10, markeredgewidth=3)

plt.title("Multi-Agent Discovery: Finding Local Minima (Descent=0)")
plt.xlim(0, grid_size-1); plt.ylim(0, grid_size-1)
plt.show()

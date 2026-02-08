import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. 地形のセットアップ
grid_size = 31
x_grid, y_grid = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
z0 = np.sin(x_grid / 5) + np.cos(y_grid / 5) + np.random.rand(grid_size, grid_size) * 0.2
z = z0.copy()

# 平滑化
for _ in range(500):
    laplacian = np.zeros_like(z)
    laplacian[1:-1, 1:-1] = (z[:-2, 1:-1] + z[2:, 1:-1] + z[1:-1, :-2] + z[1:-1, 2:] - 4 * z[1:-1, 1:-1])
    z += 0.1 * (0.8 * laplacian - 0.2 * (z - z0))

gz, gx = np.gradient(z)

# 2. エージェントの設定
num_agents = 10
# 軌跡を保存するためのリスト (各エージェントごとに (x, y) を保持)
agent_positions = np.random.uniform(2, 28, (num_agents, 2))
paths = [ [p.copy()] for p in agent_positions ]

# 3. グラフの初期設定
fig, ax = plt.subplots(figsize=(8, 7))
contour = ax.contourf(x_grid, y_grid, z, levels=20, cmap='magma')
fig.colorbar(contour)
lines = [ax.plot([], [], 'o-', lw=2, markersize=4, alpha=0.8)[0] for _ in range(num_agents)]
ax.set_title("Real-time Descent to Zero-Point")

def update(frame):
    global agent_positions
    for i in range(num_agents):
        curr = agent_positions[i]
        
        # 座標の勾配を取得
        ix, iy = np.clip(curr.astype(int), 0, grid_size-2)
        grad = np.array([gz[ix, iy], gx[ix, iy]])
        
        # 勾配降下 (Descent)
        if np.linalg.norm(grad) > 1e-4:
            curr -= 1.5 * grad # learning_rate = 1.5
            paths[i].append(curr.copy())
        
        # 描画データの更新
        p_array = np.array(paths[i])
        lines[i].set_data(p_array[:, 0], p_array[:, 1])
    
    return lines

# アニメーションの実行 (frames=繰り返し回数, interval=更新速度ms)
ani = FuncAnimation(fig, update, frames=60, interval=100, blit=True)

plt.show()

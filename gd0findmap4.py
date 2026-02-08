import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. 地形のセットアップ（少し複雑な地形に設定）
grid_size = 31
x_grid, y_grid = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
z0 = np.sin(x_grid / 4) + np.cos(y_grid / 4) + np.random.rand(grid_size, grid_size) * 0.4
z = z0.copy()

# 平滑化
for _ in range(300):
    laplacian = np.zeros_like(z)
    laplacian[1:-1, 1:-1] = (z[:-2, 1:-1] + z[2:, 1:-1] + z[1:-1, :-2] + z[1:-1, 2:] - 4 * z[1:-1, 1:-1])
    z += 0.1 * (0.8 * laplacian - 0.2 * (z - z0))

# 2. 粒子群最適化（PSO）のパラメータ
num_particles = 15
pos = np.random.uniform(2, 28, (num_particles, 2)) # 現在地
vel = np.random.uniform(-0.5, 0.5, (num_particles, 2)) # 速度（慣性）
pbest_pos = pos.copy() # 個人のベストポジション
# 個人ベストの値を初期化
pbest_val = np.array([z[int(p[0]), int(p[1])] for p in pos])
gbest_pos = pbest_pos[np.argmin(pbest_val)] # 群れ全体のベストポジション

# 物理定数
w = 0.5  # 慣性重み（前進し続ける力）
c1 = 0.8 # 自己ベストへの引力
c2 = 0.9 # 群れベストへの引力

# 3. アニメーション設定
fig, ax = plt.subplots(figsize=(8, 7))
ax.contourf(x_grid, y_grid, z, levels=25, cmap='viridis', alpha=0.9)
paths = [ [p.copy()] for p in pos ]
lines = [ax.plot([], [], 'o-', lw=1, markersize=3, alpha=0.6, color='white')[0] for _ in range(num_particles)]
gbest_marker, = ax.plot([], [], 'r*', markersize=15, label='Global Best')

def update(frame):
    global pos, vel, pbest_pos, pbest_val, gbest_pos
    
    for i in range(num_particles):
        # 1. 速度の更新 (慣性 + 自己ベスト + 群れベスト)
        r1, r2 = np.random.rand(), np.random.rand()
        vel[i] = (w * vel[i] + 
                  c1 * r1 * (pbest_pos[i] - pos[i]) + 
                  c2 * r2 * (gbest_pos - pos[i]))
        
        # 2. 位置の更新
        pos[i] += vel[i]
        pos[i] = np.clip(pos[i], 0, grid_size-1)
        
        # 3. ベストスコアの更新
        ix, iy = pos[i].astype(int)
        current_val = z[ix, iy]
        if current_val < pbest_val[i]:
            pbest_val[i] = current_val
            pbest_pos[i] = pos[i].copy()
            
        # 4. 軌跡の保存と描画
        paths[i].append(pos[i].copy())
        p_array = np.array(paths[i])
        lines[i].set_data(p_array[:, 0], p_array[:, 1])

    # 群れ全体のベストを更新
    gbest_pos = pbest_pos[np.argmin(pbest_val)]
    gbest_marker.set_data([gbest_pos[0]], [gbest_pos[1]])
    
    return lines + [gbest_marker]

ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.legend()
plt.title("Swarm Intelligence: Particles Communicating to Find the Deepest Valley")
plt.show()

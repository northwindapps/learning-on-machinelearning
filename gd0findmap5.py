import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. パラメータ設定
grid_size = 31
num_particles = 20
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")

# PSOの物理定数
w, c1, c2 = 0.4, 0.5, 0.7

# 粒子の初期化
pos = np.random.uniform(5, 25, (num_particles, 2))
vel = np.random.uniform(-0.5, 0.5, (num_particles, 2))
pbest_pos = pos.copy()
pbest_val = np.full(num_particles, np.inf)

# 2. 地形を生成する関数 (時間 t によって変化)
def get_terrain(t):
    # 山と谷がゆっくり波打つように移動
    z = np.sin(x / 5 + t * 0.1) + np.cos(y / 5 - t * 0.05)
    # 動く「深い穴」を1つ追加
    hole_x = 15 + 10 * np.sin(t * 0.07)
    hole_y = 15 + 10 * np.cos(t * 0.07)
    z -= 2.0 * np.exp(-((x - hole_x)**2 + (y - hole_y)**2) / 10)
    return z

# 3. アニメーションの設定
fig, ax = plt.subplots(figsize=(8, 7))
points, = ax.plot(pos[:, 0], pos[:, 1], 'wo', markersize=4, markeredgecolor='k', alpha=0.8)
gbest_marker, = ax.plot([], [], 'r*', markersize=15, label='Current Best')
ax.set_title("Swarm Intelligence: Tracking a Moving Target")

# 背景（地形）の初期描画
contour = [ax.contourf(x, y, get_terrain(0), levels=20, cmap='ocean')]

def update(frame):
    global pos, vel, pbest_pos, pbest_val, contour
    
    t = frame
    z = get_terrain(t)
    
    # 以前のコンターを消去して更新
    for c in contour[0].collections:
        c.remove()
    contour[0] = ax.contourf(x, y, z, levels=20, cmap='ocean', zorder=-1)
    
    # 群れ全体のベストを再評価（地形が動くため、過去のベストは無効になる可能性がある）
    current_vals = []
    for i in range(num_particles):
        ix, iy = np.clip(pos[i].astype(int), 0, grid_size-1)
        current_val = z[ix, iy]
        current_vals.append(current_val)
        
        # 自己ベスト更新 (動的環境では少し忘却を入れるのが一般的ですが、今回は単純化)
        if current_val < pbest_val[i]:
            pbest_val[i] = current_val
            pbest_pos[i] = pos[i].copy()
            
    gbest_idx = np.argmin(current_vals)
    gbest_pos = pos[gbest_idx]
    
    # 粒子の移動
    for i in range(num_particles):
        r1, r2 = np.random.rand(2)
        vel[i] = (w * vel[i] + 
                  c1 * r1 * (pbest_pos[i] - pos[i]) + 
                  c2 * r2 * (gbest_pos - pos[i]))
        pos[i] += vel[i]
        pos[i] = np.clip(pos[i], 0, grid_size-1)

    points.set_data(pos[:, 0], pos[:, 1])
    gbest_marker.set_data([gbest_pos[0]], [gbest_pos[1]])
    
    return points, gbest_marker

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
plt.show()

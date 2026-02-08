import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. 不連続で「分断された」地形の生成
grid_size = 50
x, y = np.meshgrid(np.linspace(0, 10, grid_size), np.linspace(0, 10, grid_size))

def get_rugged_terrain(x, y):
    # ベースとなる大きなうねり
    z = np.sin(x*0.5) + np.cos(y*0.5)
    
    # 1. 階段状の段差 (Quantization) -> 勾配がいたるところで0になる
    z = np.round(z * 4) / 4
    
    # 2. ランダムな「割れ目」と「壁」 (Discontinuity)
    np.random.seed(42)
    for _ in range(15):
        rx, ry = np.random.uniform(0, 10, 2)
        # 鋭い溝を追加
        z -= 1.5 * np.exp(-((x - rx)**2 + (y - ry)**2) / 0.1)
    
    return z

z_rugged = get_rugged_terrain(x, y)

# 2. エージェントの設定
num_particles = 20
pos = np.random.uniform(1, 9, (num_particles, 2))
vel = np.zeros((num_particles, 2))
pbest_pos = pos.copy()
pbest_val = np.array([z_rugged[int(p[1]*4.9), int(p[0]*4.9)] for p in pos])
gbest_pos = pbest_pos[np.argmin(pbest_val)]

# PSOパラメータ (少し「ジャンプ力」を強めに設定)
w, c1, c2 = 0.6, 0.8, 0.9

# 3. 可視化
fig, ax = plt.subplots(figsize=(9, 7))
# 'prism'や'terrain'など、段差が目立つカラーマップを使用
cont = ax.contourf(x, y, z_rugged, levels=30, cmap='nipy_spectral')
fig.colorbar(cont, label="Loss Value (Discontinuous)")

dots, = ax.plot(pos[:, 0], pos[:, 1], 'wo', markeredgecolor='k', markersize=5)
gbest_dot, = ax.plot([], [], 'r*', markersize=15, label='Global Best')

def update(frame):
    global pos, vel, pbest_pos, pbest_val, gbest_pos
    
    for i in range(num_particles):
        # 速度更新 (慣性 + 相互作用)
        r1, r2 = np.random.rand(2)
        vel[i] = w * vel[i] + c1*r1*(pbest_pos[i]-pos[i]) + c2*r2*(gbest_pos-pos[i])
        
        # 崖を飛び越えるための「ノイズ（突然変異）」を少し加える
        if np.random.rand() < 0.05:
            vel[i] += np.random.normal(0, 0.5, 2)
            
        pos[i] += vel[i]
        pos[i] = np.clip(pos[i], 0.1, 9.9)
        
        # 現在地の値をサンプリング
        ix, iy = int(pos[i, 1] * 4.9), int(pos[i, 0] * 4.9)
        val = z_rugged[ix, iy]
        
        if val < pbest_val[i]:
            pbest_val[i] = val
            pbest_pos[i] = pos[i].copy()
            
    gbest_pos = pbest_pos[np.argmin(pbest_val)]
    
    dots.set_data(pos[:, 0], pos[:, 1])
    gbest_dot.set_data([gbest_pos[0]], [gbest_pos[1]])
    return dots, gbest_dot

ani = FuncAnimation(fig, update, frames=150, interval=50, blit=True)
plt.title("Optimization on Discontinuous & Rugged Landscape")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. フラクタル（カオス）地形の生成
grid_size = 100
x_range = np.linspace(0, 10, grid_size)
y_range = np.linspace(0, 10, grid_size)
x, y = np.meshgrid(x_range, y_range)

def get_fractal_terrain(x, y, smooth_mode=False):
    z = np.zeros_like(x)
    # 複数の周波数を重ね合わせてフラクタル構造を作る (Octaves)
    for i in range(1, 5):
        freq = 2**i
        amp = 0.5**i
        z += amp * (np.sin(x * freq) * np.cos(y * freq))
    
    # ノイズの追加
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, z.shape)
    z += noise
    
    # 崖（不連続性）を作る
    z = np.where(np.abs(z) < 0.1, z - 0.5, z) 
    
    if smooth_mode:
        # ResNet的な効果：ガウスぼかしで地形を滑らかにする（スキップ接続のメタファー）
        from scipy.ndimage import gaussian_filter
        z = gaussian_filter(z, sigma=2.0)
        
    return z

# 最初は「カオスモード」で開始
z_chaos = get_fractal_terrain(x, y, smooth_mode=False)

# 2. PSO（群知能）の設定
num_particles = 30
pos = np.random.uniform(1, 9, (num_particles, 2))
vel = np.random.uniform(-0.1, 0.1, (num_particles, 2))
pbest_pos = pos.copy()
pbest_val = np.array([z_chaos[int(p[1]*9.9), int(p[0]*9.9)] for p in pos])
gbest_pos = pbest_pos[np.argmin(pbest_val)]

# 3. 可視化の設定
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左：現在の地形とエージェント
cont = ax1.contourf(x, y, z_chaos, levels=40, cmap='magma')
dots, = ax1.plot(pos[:, 0], pos[:, 1], 'wo', markeredgecolor='k', markersize=4)
gbest_marker, = ax1.plot([], [], 'r*', markersize=15)
ax1.set_title("Chaos Landscape (Fractal & Disconnected)")

# 右：ベストスコアの推移
history = []
line_hist, = ax2.plot([], [], 'b-')
ax2.set_xlim(0, 100); ax2.set_ylim(-1.5, 0.5)
ax2.set_title("Global Minimum Value Discovery")
ax2.set_xlabel("Steps"); ax2.set_ylabel("Loss")

def update(frame):
    global pos, vel, pbest_pos, pbest_val, gbest_pos
    
    # 途中で「滑らかモード」に切り替える実験も可能（手動設定）
    # z = get_fractal_terrain(x, y, smooth_mode=(frame > 50)) 
    z = z_chaos 

    for i in range(num_particles):
        r1, r2 = np.random.rand(2)
        vel[i] = 0.5 * vel[i] + 0.8 * r1 * (pbest_pos[i] - pos[i]) + 0.9 * r2 * (gbest_pos - pos[i])
        pos[i] += vel[i]
        pos[i] = np.clip(pos[i], 0.1, 9.9)
        
        ix, iy = np.clip((pos[i] * 9.9).astype(int), 0, grid_size-1)
        val = z[iy, ix]
        
        if val < pbest_val[i]:
            pbest_val[i] = val
            pbest_pos[i] = pos[i].copy()
            
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx]
    
    # 更新
    dots.set_data(pos[:, 0], pos[:, 1])
    gbest_marker.set_data([gbest_pos[0]], [gbest_pos[1]])
    
    history.append(pbest_val[gbest_idx])
    line_hist.set_data(range(len(history)), history)
    
    return dots, gbest_marker, line_hist

ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.tight_layout()
plt.show()

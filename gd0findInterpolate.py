import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# 1. 粗いデータの生成 (31x31)
size = 31
x_raw = np.arange(size)
y_raw = np.arange(size)
z0 = np.sin(x_raw[:, None] / 5) + np.cos(y_raw[None, :] / 5) + np.random.rand(size, size) * 0.2
z_data = z0.copy()

# 平滑化（前回同様のラプラス緩和）
for _ in range(500):
    laplacian = np.zeros_like(z_data)
    laplacian[1:-1, 1:-1] = (z_data[:-2, 1:-1] + z_data[2:, 1:-1] + z_data[1:-1, :-2] + z_data[1:-1, 2:] - 4 * z_data[1:-1, 1:-1])
    z_data += 0.1 * (0.8 * laplacian - 0.2 * (z_data - z0))

# 2. 【魔法のステップ】 連続関数化 (2D Interpolation)
# これにより、z_func([x, y]) で「インデックスの間」の値を計算できるようになります
z_func = RegularGridInterpolator((x_raw, y_raw), z_data, method='cubic')

def get_grad(pos, delta=0.01):
    """数値微分で任意の座標の勾配を計算する"""
    x, y = pos
    dz_dx = (z_func([x + delta, y]) - z_func([x - delta, y])) / (2 * delta)
    dz_dy = (z_func([x, y + delta]) - z_func([x, y - delta])) / (2 * delta)
    return np.array([dz_dx[0], dz_dy[0]])

# 3. エージェントの探索 (浮動小数点の座標でスムーズに移動)
start_pos = np.array([np.random.uniform(2, 28), np.random.uniform(2, 28)])
curr_pos = start_pos.copy()
path = [curr_pos.copy()]

lr = 1.0
for _ in range(100):
    grad = get_grad(curr_pos)
    curr_pos -= lr * grad
    curr_pos = np.clip(curr_pos, 0.1, 29.9)
    path.append(curr_pos.copy())
    if np.linalg.norm(grad) < 1e-4: break

path = np.array(path)

# 4. 可視化
# 背景は細かく描画 (100x100) して滑らかさを確認
x_fine, y_fine = np.meshgrid(np.linspace(0, 30, 100), np.linspace(0, 30, 100), indexing='ij')
z_fine = z_func(np.array([x_fine.ravel(), y_fine.ravel()]).T).reshape(100, 100)

plt.figure(figsize=(8, 6))
plt.contourf(np.linspace(0, 30, 100), np.linspace(0, 30, 100), z_fine.T, levels=30, cmap='viridis')
plt.colorbar(label='Loss (Interpolated)')
plt.plot(path[:, 0], path[:, 1], 'r.-', markersize=3, label='Smooth Path')
plt.plot(path[0,0], path[0,1], 'wo', label='Start')
plt.plot(path[-1,0], path[-1,1], 'r*', markersize=15, label='End')
plt.title("Optimization on Continuous (Interpolated) Function")
plt.show()

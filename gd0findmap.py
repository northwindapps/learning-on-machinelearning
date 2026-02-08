import numpy as np
import matplotlib.pyplot as plt

# 1. データの生成と平滑化 (前回の処理)
x_grid, y_grid = np.meshgrid(np.arange(31), np.arange(31), indexing="ij")
z0 = np.sin(x_grid / 5) + np.cos(y_grid / 5) + np.random.rand(31, 31) * 0.2
z = z0.copy()

# 平滑化処理 (ラプラス緩和)
lam, alpha_smooth = 0.8, 0.1
for _ in range(500):
    laplacian = np.zeros_like(z)
    laplacian[1:-1, 1:-1] = (z[:-2, 1:-1] + z[2:, 1:-1] + z[1:-1, :-2] + z[1:-1, 2:] - 4 * z[1:-1, 1:-1])
    z += alpha_smooth * (lam * laplacian - (1 - lam) * (z - z0))

# 2. 勾配 (Gradient) の計算
# np.gradient を使うと各点での傾き (dz/dx, dz/dy) が得られます
gz, gx = np.gradient(z) 

# 3. ランダムな点から開始して「勾配0」を探す (勾配降下法)
start_pos = np.array([np.random.uniform(0, 30), np.random.uniform(0, 30)])
current_pos = start_pos.copy()
path = [current_pos.copy()]

learning_rate = 2.0
n_steps = 100

for _ in range(n_steps):
    # 現在の座標の勾配を線形補間で取得
    # (簡易的に整数インデックスに丸めて参照)
    ix, iy = int(current_pos[0]), int(current_pos[1])
    ix = np.clip(ix, 0, 29)
    iy = np.clip(iy, 0, 29)
    
    grad_x = gz[ix, iy]
    grad_y = gx[ix, iy]
    
    # 勾配を下る方向に移動
    current_pos[0] -= learning_rate * grad_x
    current_pos[1] -= learning_rate * grad_y
    
    # 範囲内に収める
    current_pos = np.clip(current_pos, 0, 30)
    path.append(current_pos.copy())
    
    # 勾配がほぼ0になったら停止
    if np.hypot(grad_x, grad_y) < 1e-4:
        break

path = np.array(path)

# 4. 可視化 (XY平面)
plt.figure(figsize=(8, 6))
cp = plt.contourf(x_grid, y_grid, z, levels=20, cmap='viridis')
plt.colorbar(cp, label='Height (z)')

# パスの描画
plt.plot(path[:, 0], path[:, 1], 'r.-', label='Descent Path')
plt.plot(path[0, 0], path[0, 1], 'wo', label='Start')
plt.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='End (Grad≈0)')

plt.title("Path to the points where descent is 0")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

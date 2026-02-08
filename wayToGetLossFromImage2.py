import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from scipy.special import softmax

# 1. データ準備 (8x8ピクセルの手書き数字)
digits = load_digits()
X = digits.data / 16.0
y = digits.target
num_classes = 10

# 2. 地形生成用のランダム方向ベクトル (高次元を2次元に投影)
np.random.seed(42)
input_dim = X.shape[1]
W_base = np.random.randn(input_dim, num_classes) * 0.1
dir_x = np.random.randn(input_dim, num_classes) * 0.2
dir_y = np.random.randn(input_dim, num_classes) * 0.2

def get_loss(alpha, beta, indices=None):
    """特定の画像インデックスにおけるLossを計算"""
    W = W_base + alpha * dir_x + beta * dir_y
    X_sub = X[indices] if indices is not None else X
    y_sub = y[indices] if indices is not None else y
    
    logits = np.dot(X_sub, W)
    probs = softmax(logits, axis=1)
    # Cross Entropy Loss
    loss = -np.mean(np.log(probs[np.arange(len(y_sub)), y_sub] + 1e-10))
    return loss

# 3. 地形データの計算 (全データ vs 特定の数字)
res = 25
range_val = 10
alphas = np.linspace(-range_val, range_val, res)
betas = np.linspace(-range_val, range_val, res)
A, B = np.meshgrid(alphas, betas)

# 全データの地形
Z_all = np.array([get_loss(a, b) for a, b in zip(A.ravel(), B.ravel())]).reshape(res, res)

# 特定の数字（例：'8'）だけの地形への影響
digit_to_inspect = 8
idx_spec = np.where(y == digit_to_inspect)[0]
Z_spec = np.array([get_loss(a, b, idx_spec) for a, b in zip(A.ravel(), B.ravel())]).reshape(res, res)

# 4. エージェント（学習プロセス）のシミュレーション
path = []
curr_p = np.array([np.random.uniform(-8, 8), np.random.uniform(-8, 8)])
lr = 2.0
for _ in range(50):
    path.append(curr_p.copy())
    # 数値微分で勾配を取得
    d = 0.1
    g_x = (get_loss(curr_p[0]+d, curr_p[1]) - get_loss(curr_p[0]-d, curr_p[1])) / (2*d)
    g_y = (get_loss(curr_p[0], curr_p[1]+d) - get_loss(curr_p[0], curr_p[1]-d)) / (2*d)
    curr_p -= lr * np.array([g_x, g_y])

path = np.array(path)

# 5. 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左：全データの Loss Landscape と学習パス
cp1 = ax1.contourf(A, B, Z_all, levels=30, cmap='viridis')
ax1.plot(path[:, 0], path[:, 1], 'r.-', label='Learning Path (All Data)')
ax1.plot(path[0,0], path[0,1], 'wo', label='Start')
ax1.plot(path[-1,0], path[-1,1], 'r*', markersize=15, label='Optimized')
ax1.set_title("Total Loss Landscape (All Digits 0-9)")
fig.colorbar(cp1, ax=ax1)

# 右：特定の数字 (8) だけが見ている世界
cp2 = ax2.contourf(A, B, Z_spec, levels=30, cmap='magma')
ax2.plot(path[:, 0], path[:, 1], 'w--', alpha=0.5, label='Path from All Data')
ax2.set_title(f"Loss Landscape for Digit '{digit_to_inspect}' ONLY")
fig.colorbar(cp2, ax=ax2)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder

# 1. 小さな画像データのロード (8x8ピクセルの手書き数字)
digits = load_digits()
X = digits.data / 16.0  # 正規化 (0.0 - 1.0)
y = digits.target.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

# 2. 超簡易ネットワーク (重み W1, W2 だけを変化させる)
# 本来はもっと複雑ですが、可視化のために2つの「方向」に絞ります
np.random.seed(42)
base_W = np.random.randn(X.shape[1], 10) * 0.1 # ベースとなる重み
dir1 = np.random.randn(X.shape[1], 10) * 0.1 # 横軸の方向
dir2 = np.random.randn(X.shape[1], 10) * 0.1 # 縦軸の方向

def compute_loss(alpha, beta):
    # 重みを特定の方向にずらす: W = W_base + alpha*dir1 + beta*dir2
    W = base_W + alpha * dir1 + beta * dir2
    # ソフトマックス関数で「予想」を計算
    logits = np.dot(X, W)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    # Cross Entropy Loss (画像認識で一般的な損失関数)
    loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-10), axis=1))
    return loss

# 3. 地形（Loss Landscape）の計算
res = 25
alphas = np.linspace(-10, 10, res)
betas = np.linspace(-10, 10, res)
A, B = np.meshgrid(alphas, betas)
Z = np.array([compute_loss(a, b) for a, b in zip(np.ravel(A), np.ravel(B))]).reshape(res, res)

# 4. 可視化
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(A, B, Z, cmap='terrain', antialiased=True, alpha=0.8)
ax.set_title("Real Image Data Loss Landscape (MNIST-like)")
ax.set_xlabel("Weight Parameter 1 (Direction Alpha)")
ax.set_ylabel("Weight Parameter 2 (Direction Beta)")
ax.set_zlabel("Cross Entropy Loss")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

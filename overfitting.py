import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import load_digits
from scipy.special import softmax

# 1. データ準備
digits = load_digits()
X = digits.data / 16.0
y = digits.target
num_classes = 10

# 2. 投影用のランダム方向
np.random.seed(42)
input_dim = X.shape[1] # 64
W_base = np.random.randn(input_dim, num_classes) * 0.1
dir_x = np.random.randn(input_dim, num_classes) * 0.2
dir_y = np.random.randn(input_dim, num_classes) * 0.2

res = 20
alphas = np.linspace(-10, 10, res)
betas = np.linspace(-10, 10, res)
A, B = np.meshgrid(alphas, betas)

def get_loss_grid(indices=None):
    Z = np.zeros((res, res))
    X_sub = X[indices] if indices is not None else X
    y_sub = y[indices] if indices is not None else y
    for i in range(res):
        for j in range(res):
            W = W_base + alphas[i] * dir_x + betas[j] * dir_y
            logits = np.dot(X_sub, W)
            probs = softmax(logits, axis=1)
            Z[j, i] = -np.mean(np.log(probs[np.arange(len(y_sub)), y_sub] + 1e-10))
    return Z

print("Calculating landscapes... please wait.")
Z_base = get_loss_grid()
Z_target = get_loss_grid([10]) # インデックスをリストで指定
print("Done!")

# 3. アニメーション設定
fig, ax = plt.subplots(figsize=(8, 7))
# カラーバー用にダミー描画
temp_cont = ax.contourf(A, B, Z_base, levels=30, cmap='viridis')
fig.colorbar(temp_cont, label='Loss Value')

def update(frame):
    # 古い等高線を消去
    for coll in ax.collections:
        coll.remove()
        
    weight = (np.sin(frame * 0.03) + 1) / 2 
    Z_blended = (1 - weight) * Z_base + weight * Z_target
    
    new_contour = ax.contourf(A, B, Z_blended, levels=30, cmap='viridis')
    ax.set_title(f"Image Influence: {weight*100:.1f}%\nSlow-motion Warp Animation")
    # collections を返さず、空のリストを返す（blit=Falseなので問題ありません）
    return []

# blit=False に設定して、描画を確実に反映させます
ani = FuncAnimation(fig, update, frames=200, interval=200, blit=False)

plt.show()

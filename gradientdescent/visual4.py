import numpy as np
import matplotlib.pyplot as plt

# ===================== 解决matplotlib中文显示问题 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# Mac系统替换为：plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# Linux系统替换为：plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# ==========================================================================

# 损失函数：L(k) = (k-2)^2，真实值k=2
def L(k):
    return (k - 2) ** 2

# 梯度：dL/dk = 2(k-2)
def grad_L(k):
    return 2 * (k - 2)

# 梯度下降实现
def gradient_descent(init_k, eta, epochs):
    k = init_k
    history = [k]
    for _ in range(epochs):
        k = k - eta * grad_L(k)
        history.append(k)
    return history

# 模拟
init_k = 0  # 初始值
eta = 0.1  # 学习率
epochs = 20  # 迭代20次
k_history = gradient_descent(init_k, eta, epochs)

# 绘图
plt.figure(figsize=(10, 6))
k_range = np.linspace(-1, 5, 100)
plt.plot(k_range, L(k_range), label='$L(k) = (k-2)^2$', color='blue')
plt.scatter(k_history, [L(k) for k in k_history], color='red', s=50, zorder=5)
plt.plot(k_history, [L(k) for k in k_history], 'o-', color='red', label='迭代路径')
plt.scatter(2, 0, color='green', s=100, marker='*', label='全局最小值 (k=2)')
plt.xlabel('参数 k')
plt.ylabel('损失函数 L(k)')
plt.title('凸函数梯度下降路径（逐步逼近全局最小值）')
plt.legend()
plt.grid(True)
plt.show()
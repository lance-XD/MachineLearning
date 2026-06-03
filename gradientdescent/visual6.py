import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===================== 解决matplotlib中文显示问题 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# Mac系统替换为：plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# Linux系统替换为：plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ==========================================================================

# 1. 生成模拟数据（真实关系：y = 3x + 5，加入少量噪声）
np.random.seed(42)  # 固定随机种子，结果可复现
n = 100  # 100个样本
x = np.linspace(0, 10, n)  # x从0到10
y_true = 3 * x + 5  # 真实值
y = y_true + np.random.normal(0, 1, n)  # 加入高斯噪声


# 2. 定义损失函数（均方误差MSE）
def mse_loss(k, b, x, y):
    y_pred = k * x + b
    return np.mean((y - y_pred) ** 2)


# 3. 定义梯度计算函数
def compute_gradient(k, b, x, y):
    n = len(x)
    y_pred = k * x + b
    # 对k和b的偏导数
    dk = -2 / n * np.sum(x * (y - y_pred))
    db = -2 / n * np.sum(y - y_pred)
    return dk, db


# 4. 梯度下降主函数
def gradient_descent(x, y, init_k, init_b, eta, epochs):
    k = init_k
    b = init_b
    loss_history = []  # 记录损失函数变化
    k_history = []  # 记录k的变化
    b_history = []  # 记录b的变化

    for _ in range(epochs):
        # 计算梯度
        dk, db = compute_gradient(k, b, x, y)
        # 更新参数
        k = k - eta * dk
        b = b - eta * db
        # 记录历史
        loss_history.append(mse_loss(k, b, x, y))
        k_history.append(k)
        b_history.append(b)

    return k, b, loss_history, k_history, b_history


# 5. 超参数设置
init_k = 0  # 初始k
init_b = 0  # 初始b
eta = 0.01  # 学习率
epochs = 1000  # 迭代次数

# 6. 执行梯度下降
k_final, b_final, loss_history, k_history, b_history = gradient_descent(x, y, init_k, init_b, eta, epochs)

# 7. 结果可视化
plt.figure(figsize=(15, 10))

# 子图1：数据与拟合直线
plt.subplot(2, 2, 1)
plt.scatter(x, y, label='原始数据', color='lightblue', alpha=0.7)
plt.plot(x, k_final * x + b_final, label=f'拟合直线: y={k_final:.2f}x+{b_final:.2f}', color='red', linewidth=2)
plt.plot(x, y_true, '--', label='真实直线: y=3x+5', color='green', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('线性回归拟合结果')
plt.legend()
plt.grid(True)

# 子图2：损失函数变化曲线
plt.subplot(2, 2, 2)
plt.plot(loss_history, color='purple', linewidth=2)
plt.xlabel('迭代次数')
plt.ylabel('损失函数值 (MSE)')
plt.title('损失函数下降过程')
plt.grid(True)

# 子图3：参数k和b的变化过程
plt.subplot(2, 2, 3)
plt.plot(k_history, label='参数k的变化', color='orange', linewidth=2)
plt.plot(b_history, label='参数b的变化', color='brown', linewidth=2)
plt.xlabel('迭代次数')
plt.ylabel('参数值')
plt.title('参数迭代更新过程')
plt.legend()
plt.grid(True)

# 子图4：3D损失函数曲面与迭代路径
plt.subplot(2, 2, 4, projection='3d')
# 生成网格数据
k_range = np.linspace(0, 6, 50)
b_range = np.linspace(0, 10, 50)
K, B = np.meshgrid(k_range, b_range)
Z = np.array([[mse_loss(k, b, x, y) for k in k_range] for b in b_range])

# 绘制损失函数曲面
surf = plt.gca().plot_surface(K, B, Z, cmap='viridis', alpha=0.7)
# 绘制迭代路径
path_z = [mse_loss(k, b, x, y) for k, b in zip(k_history, b_history)]
plt.gca().plot(k_history, b_history, path_z, 'r-', marker='o', markersize=4, label='梯度下降路径')
plt.gca().scatter([k_final], [b_final], [mse_loss(k_final, b_final, x, y)], color='red', s=100, marker='*',
                  label='最优参数')
plt.gca().set_xlabel('k')
plt.gca().set_ylabel('b')
plt.gca().set_zlabel('损失值')
plt.title('损失函数曲面与梯度下降路径')
plt.legend()

plt.tight_layout()
plt.show()

# 8. 输出最终结果
print(f"最终拟合参数：k = {k_final:.4f}, b = {b_final:.4f}")
print(f"真实参数：k = 3, b = 5")
print(f"最终损失函数值：{mse_loss(k_final, b_final, x, y):.4f}")
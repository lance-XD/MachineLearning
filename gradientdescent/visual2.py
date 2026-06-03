import numpy as np
import matplotlib.pyplot as plt

# ===================== 解决matplotlib中文显示问题 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# Mac系统替换为：plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# Linux系统替换为：plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# ==========================================================================

# 定义函数和梯度（y = x² 是凸函数，最小值在x=0处）
def f(x):
    return x ** 2

def grad_f(x):
    return 2 * x

# 生成函数曲线数据（避免变量名冲突，改用x_range）
x_range = np.linspace(-5, 5, 100)
y_range = f(x_range)

# 模拟学习率过大导致震荡不收敛的情况
def gradient_descent_big_eta():
    x = 4.0  # 初始值
    eta = 0.95  # 调整为“接近收敛但略大”的学习率，体现震荡（η=1.1会直接发散）
    x_history = [x]
    y_history = [f(x)]
    max_iter = 20  # 增加迭代次数，清晰展示震荡过程

    for i in range(max_iter):
        grad = grad_f(x)
        x = x - eta * grad  # 梯度下降核心公式
        x_history.append(x)
        y_history.append(f(x))

    # 绘图（修复变量名冲突，优化可视化）
    plt.figure(figsize=(12, 7))
    # 绘制函数曲线
    plt.plot(x_range, y_range, label='$y = x^2$', color='lightblue', linewidth=2)
    # 绘制迭代路径（点+连线）
    plt.plot(x_history, y_history, 'o-', color='red', linewidth=1.5, markersize=6, label='迭代路径 (η过大)')
    plt.scatter(x_history, y_history, color='orange', s=80, zorder=5)
    # 标注最小值点
    plt.scatter(0, 0, color='green', s=150, marker='*', label='最小值 (0,0)', zorder=10)

    # 图表美化
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('学习率过大：梯度下降震荡不收敛', fontsize=14)
    plt.xlim(-5, 5)  # 限制x轴范围，聚焦震荡区域
    plt.ylim(0, 20)  # 限制y轴范围，避免发散出图
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.show()

    # 打印迭代值，直观展示震荡过程
    print("迭代过程中x的取值：")
    for i, val in enumerate(x_history):
        print(f"第{i}次迭代：x = {val:.4f}, y = {f(val):.4f}")

gradient_descent_big_eta()
import numpy as np
import matplotlib.pyplot as plt

# ===================== 解决matplotlib中文显示问题 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# Mac系统替换为：plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# Linux系统替换为：plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# ==========================================================================

# 定义函数和梯度
def f(x):
    return x ** 2

def grad_f(x):
    return 2 * x

# 生成数据
x = np.linspace(-5, 5, 100)
y = f(x)

# 学习率（步长）
lr = 0.1

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='$y = x^2$', color='blue')

# 标记点 x=3 和 x=-2
x1, x2 = 3, -2
y1, y2 = f(x1), f(x2)
plt.scatter([x1, x2], [y1, y2], color='red', s=100, zorder=5)
plt.annotate(f'x={x1}, 梯度={grad_f(x1)}', (x1, y1), textcoords="offset points", xytext=(10,10), ha='left', color='red')
plt.annotate(f'x={x2}, 梯度={grad_f(x2)}', (x2, y2), textcoords="offset points", xytext=(10,10), ha='left', color='red')

# 绘制反梯度方向（梯度下降）的箭头 ✅ 修正核心
# 箭头的dx = -lr * 梯度（x方向反梯度）
# 箭头的dy = f(x + dx) - f(x)（y方向对应函数值变化）
dx1 = -lr * grad_f(x1)
dy1 = f(x1 + dx1) - y1
plt.arrow(x1, y1, dx1, dy1, head_width=0.2, head_length=0.5, fc='green', ec='green', label='梯度下降方向 (反梯度)')

dx2 = -lr * grad_f(x2)
dy2 = f(x2 + dx2) - y2
plt.arrow(x2, y2, dx2, dy2, head_width=0.2, head_length=0.5, fc='green', ec='green')

plt.xlabel('x')
plt.ylabel('y')
plt.title('$y=x^2$ 与梯度下降方向（反梯度）')
plt.legend()
plt.grid(True)
plt.show()
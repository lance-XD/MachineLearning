import numpy as np
import matplotlib.pyplot as plt

# ===================== 解决matplotlib中文显示问题 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# Mac系统替换为：plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# Linux系统替换为：plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ==========================================================================

import numpy as np
import matplotlib.pyplot as plt

# ===================== 基础配置（必加，避免中文/负号显示问题） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# Mac系统替换：plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# Linux系统替换：plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 定义核心函数（必须先定义，否则代码会报错） =====================
def f(x):
    """目标函数：y = x²（凸函数，最小值在x=0处）"""
    return x ** 2


def grad_f(x):
    """目标函数的梯度：f'(x) = 2x"""
    return 2 * x


# ===================== 完善后的“学习率太小导致收敛慢”函数 =====================
def gradient_descent_small_eta():
    x = 4.0  # 初始值（从x=4出发，目标收敛到x=0）
    eta_small = 0.01  # 过小的学习率
    x_history = [x]
    y_history = [f(x)]
    max_iter = 100  # 迭代次数从50→100，更易看出“慢”（50次几乎没动）

    # 迭代过程
    for i in range(max_iter):
        grad = grad_f(x)
        x = x - eta_small * grad  # 梯度下降核心公式
        x_history.append(x)
        y_history.append(f(x))

    # ===================== 可视化优化（核心：强化对比+清晰标注） =====================
    plt.figure(figsize=(12, 7))

    # 1. 绘制目标函数曲线
    x_range = np.linspace(-5, 5, 100)
    plt.plot(x_range, f(x_range), label='$y = x^2$（目标函数）', color='lightblue', linewidth=2)

    # 2. 绘制迭代路径（点+连线，更直观看到“缓慢移动”）
    plt.plot(x_history, y_history, 'o-', color='purple', linewidth=1.2, markersize=5,
             label=f'迭代路径（η={eta_small}，迭代{max_iter}次）')
    plt.scatter(x_history, y_history, color='purple', s=60, zorder=5)

    # 3. 标注关键信息：初始点+最小值点
    plt.scatter(4.0, f(4.0), color='red', s=120, marker='^', label='初始点 (4, 16)', zorder=10)
    plt.scatter(0, 0, color='green', s=150, marker='*', label='最小值点 (0, 0)', zorder=10)

    # 4. 标注最终迭代位置（突出“收敛慢”）
    final_x = x_history[-1]
    final_y = y_history[-1]
    plt.scatter(final_x, final_y, color='orange', s=120, marker='s',
                label=f'迭代{max_iter}次后位置 ({final_x:.2f}, {final_y:.2f})', zorder=10)

    # 5. 图表美化+信息标注
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('学习率太小：梯度下降收敛极慢', fontsize=14, fontweight='bold')
    plt.xlim(-1, 5)  # 聚焦迭代区域，避免画面分散
    plt.ylim(0, 18)  # 限制y轴，突出“下降慢”
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)

    # 6. 打印关键数值（辅助理解）
    print(f"初始值：x = 4.0，y = {f(4.0)}")
    print(f"学习率 η = {eta_small}，迭代{max_iter}次后：")
    print(f"x = {final_x:.4f}，y = {final_y:.4f}")
    print(f"距离最小值点 (0,0) 的误差：|x| = {abs(final_x):.4f}")

    plt.show()


# 执行函数
gradient_descent_small_eta()
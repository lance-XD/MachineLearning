import math

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（以SimHei为例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 训练数据
train_data = np.array([
    [85, 95],  # 容易
    [90, 98],  # 容易
    [80, 90],  # 容易
    [70, 75],  # 中等
    [65, 85],  # 中等
    [60, 75],  # 中等
    [55, 68],  # 困难
    [50, 60],  # 困难
    [52, 65]  # 困难
])
train_labels = ["容易", "容易", "容易", "中等", "中等", "中等", "困难", "困难", "困难"]

# 测试数据
test_data = np.array([68, 62])

# 颜色映射
colors = {"容易": "green", "中等": "orange", "困难": "red"}
train_colors = [colors[label] for label in train_labels]

# 计算距离
distances = np.sqrt(np.sum((train_data - test_data) ** 2, axis=1))

# 创建图像
plt.figure(figsize=(8, 6))

# 绘制训练数据点
for i, (point, color, label) in enumerate(zip(train_data, train_colors, train_labels)):
    plt.scatter(point[0], point[1], color=color, label=f"课程 {i + 1} ({label})" if i < 9 else "")
    # 绘制连线
    plt.plot([test_data[0], point[0]], [test_data[1], point[1]], linestyle="--", color=color, alpha=0.6)
    # 标记距离
    mid_x, mid_y = (test_data[0] + point[0]) / 2, (test_data[1] + point[1]) / 2
    plt.text(mid_x, mid_y, f"{distances[i]:.2f}", fontsize=9, color=color)

# 绘制测试数据点
plt.scatter(test_data[0], test_data[1], color="blue", label="测试数据", edgecolor="black", s=100)

# 设置图例和轴标签
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.xlabel("平均分(分)")
plt.ylabel("完成率(%)")
plt.title("测试数据与训练数据点的距离")

# 调整图像布局以适应图例
plt.tight_layout()

# 显示图像
plt.grid(alpha=0.5)
plt.show()

a = [[85, 95],  # 容易
     [90, 98],  # 容易
     [80, 90],  # 容易
     [70, 75],  # 中等
     [65, 85],  # 中等
     [60, 75],  # 中等
     [55, 68],  # 困难
     [50, 60],  # 困难
     [52, 65]  # 困难
     ]

b = [68, 62]
for i in range(9):
    x_1, x_2 = a[i]
    print(f"{math.sqrt((x_1 - b[0]) ** 2 + (x_2 - b[1]) ** 2):.2f}")

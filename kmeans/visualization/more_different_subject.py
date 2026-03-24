import matplotlib.pyplot as plt
import numpy as np


# 设置中文字体,解决matplotlib无法正常显示中文的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 数据
students = list(range(1, 11))
physics_scores = [86, 94, 87, 80, 70, 75, 85, 85, 81, 71]
history_scores = [59, 50, 55, 90, 86, 72, 70, 65, 68, 70]

# 初始中心点 (序号2和序号5)
centers = {
    1: (physics_scores[1], history_scores[1]),  # 第1组中心
    2: (physics_scores[4], history_scores[4])   # 第2组中心
}

# 计算距离并分组
group1 = []
group2 = []

for i, (x, y) in enumerate(zip(physics_scores, history_scores)):
    dist_to_center1 = np.sqrt((x - centers[1][0])**2 + (y - centers[1][1])**2)
    dist_to_center2 = np.sqrt((x - centers[2][0])**2 + (y - centers[2][1])**2)
    if dist_to_center1 <= dist_to_center2:
        group1.append((x, y, students[i]))
    else:
        group2.append((x, y, students[i]))

# 分组后的点
group1_x = [x[0] for x in group1]
group1_y = [x[1] for x in group1]
group2_x = [x[0] for x in group2]
group2_y = [x[1] for x in group2]

# 创建散点图
plt.figure(figsize=(8, 6))
plt.scatter(group1_x, group1_y, color='blue', s=100, label='第1组', edgecolors='black')
plt.scatter(group2_x, group2_y, color='red', s=100, label='第2组', edgecolors='black')
plt.scatter(*centers[1], color='blue', s=200, marker='X', label='第1组中心', edgecolors='black')
plt.scatter(*centers[2], color='red', s=200, marker='X', label='第2组中心', edgecolors='black')

# 添加数据点标签
for x, y, student in group1 + group2:
    plt.text(x + 0.5, y + 0.5, f"学生{student}", fontsize=10)

# 图表标题和轴标签
plt.title("分组散点图", fontsize=16)
plt.xlabel("物理成绩(分)", fontsize=14)
plt.ylabel("历史成绩(分)", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图表
plt.show()

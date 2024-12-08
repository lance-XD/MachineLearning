import matplotlib.pyplot as plt

# 设置中文字体,解决matplotlib无法正常显示中文的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 数据
students = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
physics_scores = [86, 94, 87, 80, 70, 75, 82, 85, 81, 71]
history_scores = [59, 50, 55, 90, 86, 72, 70, 65, 68, 70]

# 创建散点图
plt.figure(figsize=(8, 6))
plt.scatter(physics_scores, history_scores, color='blue', s=100, edgecolors='black')

# 添加数据点标签
for i, (x, y) in enumerate(zip(physics_scores, history_scores)):
    plt.text(x + 0.5, y + 0.5, f"学生{i + 1}", fontsize=10)

# 图表标题和轴标签
plt.title("物理成绩与历史成绩散点图", fontsize=16)
plt.xlabel("物理成绩(分)", fontsize=14)
plt.ylabel("历史成绩(分)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图表
plt.show()

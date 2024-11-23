import matplotlib.pyplot as plt

# 设置中文字体,解决matplotlib无法正常显示中文的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据
data = {
    "编号": [1, 2, 3, 4, 5, 6, 7, 8],
    "水果类型": ["A", "A", "A", "A", "B", "B", "B", "B"],
    "重量（g）": [150, 160, 140, 155, 2000, 1800, 2100, 1900],
    "直径（cm）": [7, 7, 6, 7, 25, 22, 26, 24]
}
test_point = {"重量（g）": 170, "直径（cm）": 8}  # 测试点

# 创建散点图
plt.figure(figsize=(10, 6))
colors = {'A': 'blue', 'B': 'green'}
labels = {'A': '水果A', 'B': '水果B'}

# 绘制已知水果数据点
for i in range(len(data["编号"])):
    plt.scatter(data["重量（g）"][i], data["直径（cm）"][i],
                color=colors[data["水果类型"][i]], label=labels[data["水果类型"][i]] if i < 1 or i == 4 else "")
    p_x = data["重量（g）"][i] + 90 if i != 3 else data["重量（g）"][i] - 80
    p_y = data["直径（cm）"][i] - 0.3 if i != 1 else data["直径（cm）"][i] - 1
    # 添加数值标注
    plt.text(p_x, p_y,
             f"({data['重量（g）'][i]},{data['直径（cm）'][i]})", fontsize=9, ha='center')

# 绘制测试点
plt.scatter(test_point["重量（g）"], test_point["直径（cm）"], color='red', label='测试点', edgecolor='black', s=40)
plt.text(test_point["重量（g）"] + 90, test_point["直径（cm）"] - 0.3,
         f"({test_point['重量（g）']},{test_point['直径（cm）']})", fontsize=9, ha='center', color='red')

# 图例与标题
plt.title("水果分类的散点图", fontsize=14)
plt.xlabel("重量（g）", fontsize=12)
plt.ylabel("直径（cm）", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# 显示图表
plt.show()

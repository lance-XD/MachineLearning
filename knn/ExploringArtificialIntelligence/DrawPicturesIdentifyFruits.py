import matplotlib.pyplot as plt

# 设置中文字体,解决matplotlib无法正常显示中文的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# # 数据
# data = {
#     "编号": [1, 2, 3, 4, 5, 6, 7, 8],
#     "水果类型": ["A", "A", "A", "A", "B", "B", "B", "B"],
#     "重量（g）": [150, 160, 140, 155, 2000, 1800, 2100, 1900],
#     "直径（cm）": [7, 7, 6, 7, 25, 22, 26, 24]
# }
# test_point = {"重量（g）": 170, "直径（cm）": 8}  # 测试点

data = {
    "编号": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "水果类型": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"],
    "重量（g）": [155, 153, 153, 150, 150, 149, 148, 144, 143, 140, 155, 160, 162, 163, 165, 165, 166, 168, 169, 170],
    "直径（cm）": [7.1, 7.2, 7.1, 7, 7, 6.9, 6.8, 6.7, 6.8, 6.8, 7.2, 7.4, 7.4, 7.3, 7.3, 7.4, 7.3, 7.6, 7.6, 7.5]
}
test_point = {"重量（g）": 157, "直径（cm）": 7.3}  # 测试点

# 创建散点图
plt.figure(figsize=(10, 8))
colors = {'A': 'blue', 'C': 'green'}
labels = {'A': '水果A', 'C': '水果C'}

# 绘制已知水果数据点
for i in range(len(data["编号"])):
    plt.scatter(data["重量（g）"][i], data["直径（cm）"][i],
                color=colors[data["水果类型"][i]], label=labels[data["水果类型"][i]] if i < 1 or i == 10 else "")
    p_x = data["重量（g）"][i] + 90 if i != 3 else data["重量（g）"][i] - 80
    p_y = data["直径（cm）"][i] - 0.3 if i != 1 else data["直径（cm）"][i] - 1
    # 添加数值标注
    # plt.text(p_x, p_y,
    #          f"({data['重量（g）'][i]},{data['直径（cm）'][i]})", fontsize=9, ha='center')

# 绘制测试点
plt.scatter(test_point["重量（g）"], test_point["直径（cm）"], color='green', label='测试点', edgecolor='black', s=40)
# plt.text(test_point["重量（g）"] + 90, test_point["直径（cm）"] - 0.3,
#          f"({test_point['重量（g）']},{test_point['直径（cm）']})", fontsize=9, ha='center', color='red')

# 图例与标题
plt.title("KNN水果分类图", fontsize=14)
plt.xlabel("重量（g）", fontsize=12)
plt.ylabel("直径（cm）", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# 显示图表
plt.show()

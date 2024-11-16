import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# 设置中文字体（以SimHei为例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 示例颜色数据（RGB三通道）
X = np.array([
    [255, 0, 0], [255, 0, 10], [235, 0, 0], [250, 30, 30], [250, 40, 40],
    [0, 255, 0], [0, 230, 10], [10, 235, 10], [50, 230, 50], [40, 210, 40],
    [0, 0, 255], [20, 15, 200], [0, 90, 180], [50, 50, 250], [0, 70, 160],
    [255, 255, 0], [240, 230, 10], [240, 250, 0], [235, 235, 10], [220, 250, 30],
    [255, 165, 0], [240, 155, 0], [250, 165, 20], [240, 165, 5], [235, 175, 0],
    [255, 255, 255], [240, 255, 240], [240, 240, 240], [250, 250, 250], [235, 235, 235]
])

# 颜色标签 0: 红色, 1: 绿色, 2: 蓝色, 3: 黄色, 4: 橙色, 5: 白色
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])

# 测试点
test_point = np.array([20, 150, 110])

# 计算测试点与所有训练数据点的欧几里得距离
distances = np.linalg.norm(X - test_point, axis=1)

# **绘制测试点与所有点的连线**
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'white']

# 绘制数据点
for i, color in enumerate(colors):
    ax.scatter(X[y == i][:, 0], X[y == i][:, 1], X[y == i][:, 2],
               color=color, label=f'{colors[i]}', s=50)

# 测试点
ax.scatter(test_point[0], test_point[1], test_point[2],
           color='black', label="测试点", s=100, marker='x')

neighbors = []
dists = []
# 绘制连线并标注距离
for dist, neighbor in zip(distances, X):
    ax.plot([test_point[0], neighbor[0]], [test_point[1], neighbor[1]],
            [test_point[2], neighbor[2]], 'k--')

    # 标注距离（保留整数部分）
    mid_x = (test_point[0] + neighbor[0]) / 2
    mid_y = (test_point[1] + neighbor[1]) / 2
    mid_z = (test_point[2] + neighbor[2]) / 2
    ax.text(mid_x, mid_y, mid_z, f'{int(dist)}', color='blue')

    # 输出距离信息
    print(f"测试点 {test_point} 到点 {neighbor} 的距离: {int(dist)}")

    neighbors.append(neighbor)
    dists.append(int(dist))

# 设置图形
ax.set_title("测试点与所有点的连线及距离")
ax.set_xlabel("R值")
ax.set_ylabel("G值")
ax.set_zlabel("B值")
ax.legend()

# 显示图形
plt.show()

file_name = f'test.csv'
# 打开文件并使用csv.writer写入数据
with open(file_name, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # print(f"试室：高一{i + 1}班")
    result = []
    result.append(
        ["红点", "到红点的距离", "绿点", "到绿点的距离", "蓝点", "到蓝点的距离", "黄点", "到黄点的距离", "橙点",
         "到橙点的距离", "白点", "到白点的距离", ])
    for i in range(5):
        curr_row = []
        for k in range(6):
            curr_row.append(list(neighbors[i + k * 5]))
            curr_row.append(dists[i + k * 5])
        # print(curr_row)
        result.append(curr_row)
    # 写入多行数据
    writer.writerows(result)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 设置中文字体（以SimHei为例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 示例颜色数据（RGB三通道）
# 用你自己的数据替换这些
X = np.array([
    [255, 0, 0], [255, 0, 10], [235, 0, 0], [250, 30, 30], [250, 40, 40],
    [0, 255, 0], [0, 230, 10], [10, 235, 10], [50, 230, 50], [40, 210, 40],
    [0, 0, 255], [20, 15, 200], [0, 90, 180], [50, 50, 250], [0, 70, 160],
    [255, 255, 0], [240, 230, 10], [240, 250, 0], [235, 235, 10], [220, 250, 30],
    [255, 165, 0], [240, 155, 0], [250, 165, 20], [240, 165, 5], [235, 175, 0],
    [255, 255, 255], [240, 255, 240], [240, 240, 240], [250, 250, 250], [235, 235, 235]
])

# 颜色标签 0: 红色, 1: 绿色, 2: 蓝色, 3: 黄色, 4: 橙色, 5:白色
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])

# 测试点
test_point = np.array([20, 150, 110])

# 创建 3D 图形
fig = plt.figure(figsize=(16, 8))

# **第一张图：K=3 的分类结果**
knn_k3 = KNeighborsClassifier(n_neighbors=3)
knn_k3.fit(X, y)

# 预测测试点
k3_result = knn_k3.predict([test_point])[0]
k3_neighbors = knn_k3.kneighbors([test_point], return_distance=False)[0]

# 第一张图绘制
ax1 = fig.add_subplot(121, projection='3d')
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'gray']
for i, color in enumerate(colors):
    ax1.scatter(X[y == i][:, 0], X[y == i][:, 1], X[y == i][:, 2],
                color=color, label=f'{colors[i]}', s=50)

# 测试点
ax1.scatter(test_point[0], test_point[1], test_point[2],
            color='black', label="测试点", s=100, marker='x')

# 连接测试点与最近的 3 个邻居
for neighbor_index in k3_neighbors:
    neighbor = X[neighbor_index]
    ax1.plot([test_point[0], neighbor[0]], [test_point[1], neighbor[1]],
             [test_point[2], neighbor[2]], 'k--')

ax1.set_title("K=3 分类结果")
ax1.set_xlabel("R值")
ax1.set_ylabel("G值")
ax1.set_zlabel("B值")
ax1.legend()

# **第二张图：K=5 的分类结果**
knn_k5 = KNeighborsClassifier(n_neighbors=5)
knn_k5.fit(X, y)

# 预测测试点
k5_result = knn_k5.predict([test_point])[0]
k5_neighbors = knn_k5.kneighbors([test_point], return_distance=False)[0]

# 第二张图绘制
ax2 = fig.add_subplot(122, projection='3d')
for i, color in enumerate(colors):
    ax2.scatter(X[y == i][:, 0], X[y == i][:, 1], X[y == i][:, 2],
                color=color, label=f'{colors[i]}', s=50)

# 测试点
ax2.scatter(test_point[0], test_point[1], test_point[2],
            color='black', label="测试点", s=100, marker='x')

# 连接测试点与最近的 3 个邻居
for neighbor_index in k5_neighbors:
    neighbor = X[neighbor_index]
    ax2.plot([test_point[0], neighbor[0]], [test_point[1], neighbor[1]],
             [test_point[2], neighbor[2]], 'k--')

ax2.set_title("K=3 分类结果")
ax2.set_xlabel("R值")
ax2.set_ylabel("G值")
ax2.set_zlabel("B值")
ax2.legend()

# 显示图形
plt.suptitle("KNN 分类过程（K=3 vs K=5）", fontsize=16)
plt.tight_layout()
plt.show()

# 打印分类结果
print(f"K=3 时分类结果: {k3_result}（错误分类）")
print(f"K=5 时分类结果: {k5_result}（正确分类）")

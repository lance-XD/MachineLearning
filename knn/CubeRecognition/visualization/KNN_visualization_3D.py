import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 设置中文字体（以SimHei为例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 示例颜色数据（RGB三通道）
# 用你自己的数据替换这些
X = np.array([
    [255, 0, 0], [250, 30, 30], [210, 40, 40],
    [10, 250, 10], [50, 230, 50], [40, 210, 40],
    [10, 5, 200], [50, 50, 250], [40, 40, 210],
    [250, 250, 0], [255, 255, 10], [255, 255, 30],
    [255, 165, 0], [255, 165, 5], [255, 175, 0],
    [250, 255, 255], [250, 250, 255], [255, 255, 250]
])

# 颜色标签 0: 红色, 1: 绿色, 2: 蓝色, 3: 黄色, 4: 橙色, 5:白色
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])

# 创建并训练 KNN 分类器
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# 创建 3D 图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制训练数据点
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'white']
for i, color in enumerate(colors):
    ax.scatter(X[y == i][:, 0], X[y == i][:, 1], X[y == i][:, 2],
               color=color, label=f'{colors[i]}', s=50)

# 设置轴标签
ax.set_xlabel("红色通道")
ax.set_ylabel("绿色通道")
ax.set_zlabel("蓝色通道")
ax.legend()
plt.title("KNN 颜色分类的 3D 可视化")
plt.show()

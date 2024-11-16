import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# 魔方色块的RGB颜色数据和标签
# 测试数据和标签会直接决定了边界在哪里
colors = [
    (255, 0, 0), (250, 30, 30), (210, 40, 40),  # 红色
    (10, 250, 10), (50, 230, 50), (40, 210, 40),  # 绿色
    (10, 5, 200), (50, 50, 250), (40, 40, 210),  # 蓝色
    (250, 250, 0), (255, 255, 10), (255, 255, 30),  # 黄色
    (255, 165, 0), (255, 165, 5), (255, 175, 0),  # 橙色
    (250, 255, 255), (250, 250, 255), (255, 255, 250),  # 白色
]
# 0: 红色, 1: 绿色, 2: 蓝色, 3: 黄色, 4: 橙色, 5:白色
labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]

# 将颜色数据和标签转换为 NumPy 数组
X = np.array(colors)
y = np.array(labels)

# 使用 PCA 将 RGB 数据降维到二维
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# 创建 KNN 分类器并训练
k_neighbors = 3
knn = KNeighborsClassifier(n_neighbors=k_neighbors)
knn.fit(X_2d, y)

# 生成 2D 网格以绘制决策边界
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))  # 使用 linspace

# 预测网格上每个点的分类
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 1. 绘制输入数据点
plt.figure(figsize=(6, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=100)
plt.title("Input datapoints")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# 2. 绘制 KNN 分类器边界
plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=100)
plt.title("KNN classifier boundaries")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# 3. 添加测试数据点
test_datapoint = [10, 10, 250]  # 假设测试数据点
test_datapoint_2d = pca.transform([test_datapoint])
plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=100)
plt.scatter(test_datapoint_2d[0, 0], test_datapoint_2d[0, 1], marker='x', s=200, color='black')
plt.title("Test datapoint")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# 4. 绘制最近邻居
dist, indices = knn.kneighbors(test_datapoint_2d)
plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=100)
plt.scatter(test_datapoint_2d[0, 0], test_datapoint_2d[0, 1], marker='x', s=200, color='black')
for i in indices[0]:
    plt.scatter(X_2d[i, 0], X_2d[i, 1], s=100, facecolors='none', edgecolor='black')
plt.title("k nearest neighbors")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

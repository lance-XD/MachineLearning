import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 设置中文字体,解决matplotlib无法正常显示中文的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 训练颜色数据（RGB三通道）
X = np.array([
    [255, 0, 0], [255, 0, 10], [235, 0, 0], [250, 30, 30], [250, 40, 40],  # 红色
    [0, 255, 0], [0, 230, 10], [10, 235, 10], [50, 230, 50], [40, 210, 40],  # 绿色
    [0, 0, 255], [20, 15, 200], [0, 90, 180], [50, 50, 250], [0, 70, 160],  # 蓝色
    [255, 255, 0], [240, 230, 10], [240, 250, 0], [235, 235, 10], [220, 250, 30],  # 黄色
    [255, 165, 0], [240, 155, 0], [250, 165, 20], [240, 165, 5], [235, 175, 0],  # 橙色
    [255, 255, 255], [240, 255, 240], [240, 240, 240], [250, 250, 250], [235, 235, 235]  # 白色
])

# 颜色标签 0: 红色, 1: 绿色, 2: 蓝色, 3: 黄色, 4: 橙色, 5: 白色
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])

# 测试点
test_point = np.array([20, 150, 110])

# 创建 3D 图形, 图形大小为 16x8 英寸
fig = plt.figure(figsize=(16, 8))

# 将预测的标签转换为对应的颜色名称
color_names = ['红', '绿', '蓝', '黄', '橙', '白']


# **绘制图像方法**
def plot_knn(ax, X, y, test_point, k, title):
    # 创建 KNN 模型
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    # 获取最近邻的索引和距离
    distances, indices = knn.kneighbors([test_point])

    # 颜色设置
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'white']

    # 绘制数据点
    for i, color in enumerate(colors):
        ax.scatter(X[y == i][:, 0], X[y == i][:, 1], X[y == i][:, 2],
                   color=color, label=f'{colors[i]}', s=50)

    # 测试点
    ax.scatter(test_point[0], test_point[1], test_point[2],
               color='black', label="测试点", s=100, marker='x')

    # 绘制连线并标注距离
    for dist, idx in zip(distances[0], indices[0]):
        neighbor = X[idx]
        ax.plot([test_point[0], neighbor[0]], [test_point[1], neighbor[1]],
                [test_point[2], neighbor[2]], 'k--')

        # 标注距离（保留整数部分）
        mid_x = (test_point[0] + neighbor[0]) / 2
        mid_y = (test_point[1] + neighbor[1]) / 2
        mid_z = (test_point[2] + neighbor[2]) / 2
        ax.text(mid_x, mid_y, mid_z, f'{int(dist)}', color='blue')

        # 输出距离信息
        print(f"K={k} 时，测试点 {list(test_point)} 到点 {list(neighbor)} 的距离为: {int(dist)}")

    # 设置图形属性
    ax.set_title(title)
    ax.set_xlabel("R值")
    ax.set_ylabel("G值")
    ax.set_zlabel("B值")
    ax.legend()

    predicted_labels = knn.predict([test_point])[0]
    print(f"k={k}时，测试点 {list(test_point)}预测的颜色为：{color_names[predicted_labels]}. ")


# **第一张图：K=3 的分类结果**
ax1 = fig.add_subplot(121, projection='3d')
plot_knn(ax1, X, y, test_point, k=3, title="K=3 最近邻")

# **第二张图：K=5 的分类结果**
ax2 = fig.add_subplot(122, projection='3d')
plot_knn(ax2, X, y, test_point, k=5, title="K=5 最近邻")

# 显示图形
plt.suptitle("KNN 测试点与邻居连线示意图 (K=3 vs K=5)", fontsize=16)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 设置中文字体,解决matplotlib无法正常显示中文的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 训练颜色数据（RGB三通道）
X = np.array([
    [85, 95],  # 容易
    [90, 98],  # 容易
    [80, 90],  # 容易
    [70, 75],  # 中等
    [65, 85],  # 中等
    [60, 75],  # 中等
    [55, 68],  # 困难
    [50, 60],  # 困难
    [52, 65]   # 困难
])

# 颜色标签 0: 容易, 1: 中等, 2: 困难
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# 测试点
test_point = np.array([68, 62])  # 测试点的RGB

# 创建平面图形，图形大小为 16x8 英寸
fig = plt.figure(figsize=(16, 8))

# 将预测的标签转换为对应的颜色名称
color_names = ['容易', '中等', '困难']


# **绘制图像方法**
def plot_knn(ax, X, y, test_point, k, title):
    # 创建 KNN 模型
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    # 获取最近邻的索引和距离
    distances, indices = knn.kneighbors([test_point])

    # 颜色设置
    colors = ['red', 'green', 'blue']

    # 绘制数据点
    for i, color in enumerate(colors):
        ax.scatter(X[y == i][:, 0], X[y == i][:, 1],
                   color=color, label=f'{color_names[i]}', s=50)

    # 测试点
    ax.scatter(test_point[0], test_point[1],
               color='black', label="测试点", s=100, marker='x')

    # 绘制连线并标注距离
    for dist, idx in zip(distances[0], indices[0]):
        neighbor = X[idx]
        ax.plot([test_point[0], neighbor[0]], [test_point[1], neighbor[1]], 'k--')

        # 标注距离（保留整数部分）
        mid_x = (test_point[0] + neighbor[0]) / 2
        mid_y = (test_point[1] + neighbor[1]) / 2
        ax.text(mid_x, mid_y, f'{dist:.2f}', color='blue')

        # 输出距离信息
        print(f"K={k} 时，测试点 {test_point} 到点 {neighbor} 的距离为: {dist:.2f}")

    # 设置图形属性
    ax.set_title(title)
    ax.set_xlabel("平均分(分)")
    ax.set_ylabel("完成率(%)")
    ax.legend()

    predicted_labels = knn.predict([test_point])[0]
    if k == 2:
        print(f"k={k}时，测试点 {test_point}预测的难度为： ")
    else:
        print(f"k={k}时，测试点 {test_point}预测的难度为：{color_names[predicted_labels]}")


# **第一张图：K=2 的分类结果**
ax1 = fig.add_subplot(121)
plot_knn(ax1, X, y, test_point, k=2, title="K=2 最近邻")

# **第二张图：K=3 的分类结果**
ax2 = fig.add_subplot(122)
plot_knn(ax2, X, y, test_point, k=3, title="K=3 最近邻")

# 显示图形
plt.suptitle("KNN 测试点与邻居连线示意图 (K=2 vs K=3)", fontsize=16)
plt.tight_layout()
plt.show()

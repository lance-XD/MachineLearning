# 导入所需的库
import cv2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 1. 定义颜色标签和对应的 RGB 特征数据
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

# 2. 初始化 KNN 分类器
# 设定 n_neighbors=5，表示每次分类时会考虑最邻近的5个点
# 考虑邻近点的个数跟训练数据每个类别的个数相关，会直接影响到最终的分类效果
knn = KNeighborsClassifier(n_neighbors=5)

# 3. 用训练数据进行模型训练
# 将 训练数据 和 标签 提供给 KNN 分类器，以学习颜色和标签的关系
knn.fit(X, y)

# 4. 测试数据 - 3*3的RGB数据
# 将9个色块的RGB数据放到测试数据中，用列表保存，格式为[R,G,B].
# 将要识别的数据放到此处，按照从左到右、从上到下排列
test_colors = [
    [[250, 10, 10], [5, 240, 10], [10, 10, 245]],
    [[240, 240, 5], [250, 130, 10], [250, 250, 250]],
    [[250, 10, 10], [10, 240, 10], [20, 150, 110]]
]

# 预测的标签对应的颜色名称
color_names = ['red', 'green', 'blue', 'yellow', 'orange', 'white']

# 5. 使用模型对测试数据进行预测,并按照原本的顺序添加到predicted_result中，用以后续显示图片
predicted_result = []
for i in range(len(test_colors)):
    curr_row = []
    for j in range(len(test_colors[i])):
        # 预测单个测试数据的颜色标签
        predicted_labels = knn.predict([test_colors[i][j]])[0]
        print(f"测试数据{list(test_colors[i][j])}的预测颜色为：{color_names[predicted_labels]}")
        curr_row.append(predicted_labels)
    predicted_result.append(curr_row)


def show_predict_image(labels):
    """
    用OpenCV显示预测的魔方颜色图，模拟3阶魔方的排布，形象直观
    :param labels: 预测的结果
    """
    # 颜色名称到 RGB 值的映射
    color_map = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'orange': (255, 165, 0),
        'white': (255, 255, 255)
    }


    # 色块尺寸，绘制预测结果的100 * 100色块
    block_size = 100
    max_width = max(len(row) for row in labels)  # 最宽的列数
    height = len(labels) * block_size
    width = max_width * block_size
    # 创建空白图像
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    # 显示测试颜色对应的色块图片
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            # 计算当前色块在图像中的位置
            start_x = i * block_size
            start_y = j * block_size
            end_x = start_x + block_size
            end_y = start_y + block_size
            # 设置色块颜色，注意 OpenCV 使用 BGR 格式
            color_image[start_x:end_x, start_y:end_y] = color_map.get(color_names[labels[i][j]], (0, 0, 0))[::-1]

    # 显示颜色网格
    cv2.imshow('Color Grid', color_image)
    # 按任意键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 将预测结果按原本的RGB排布顺序展示
show_predict_image(predicted_result)

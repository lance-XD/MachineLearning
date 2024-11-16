# 导入所需的库
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 1. 定义颜色标签和对应的 RGB 特征数据
# 假设我们有一组训练数据，代表魔方常见的颜色
# colors_data 是训练数据，包含了已知颜色的 RGB 值
# colors_labels 是对应的标签，例如 0 = 红色，1 = 绿色，依次表示
colors_data = [
    [255, 0, 0],  # 红色
    [0, 255, 0],  # 绿色
    [0, 0, 255],  # 蓝色
    [255, 255, 0],  # 黄色
    [255, 165, 0],  # 橙色
    [255, 255, 255],  # 白色
]

# 颜色标签：分别对应0-红、1-绿、2-蓝、3-黄、4-橙、5-白
colors_labels = [0, 1, 2, 3, 4, 5]

# 2. 初始化 KNN 分类器
# 设定 n_neighbors=1，表示每次分类时会考虑最邻近的1个点
# 考虑邻近点的个数跟训练数据每个类别的个数相关，会直接影响到最终的分类效果
knn = KNeighborsClassifier(n_neighbors=1)

# 3. 用训练数据进行模型训练
# 将 colors_data 和 colors_labels 提供给 KNN 分类器，以学习颜色和标签的关系
knn.fit(colors_data, colors_labels)

# 4. 测试数据 - 9 个色块的 RGB 值
# 将9个色块的RGB数据放到测试数据中，用列表保存，格式为[R,G,B].
# 将要识别的数据放到此处，按照从左到右、从上到下排列
test_colors = [
    [250, 10, 10],  # 假设为红色
    [5, 240, 10],  # 假设为绿色
    [10, 10, 245],  # 假设为蓝色
    [240, 240, 5],  # 假设为黄色
    [250, 130, 10],  # 假设为橙色
    [250, 250, 250],  # 假设为白色
    [250, 10, 10],  # 假设为红色
    [10, 240, 10],  # 假设为绿色
    [10, 10, 250],  # 假设为蓝色
]

# 5. 使用模型对测试数据进行预测
# 调用 knn.predict()，将测试颜色数据输入模型进行分类预测
predicted_labels = knn.predict(test_colors)

# 6. 输出预测结果
# 将预测的标签转换为对应的颜色名称
color_names = ['红', '绿', '蓝', '黄', '橙', '白']
# 输出色块的值
# for i, label in enumerate(predicted_labels):
#     print(f"色块 {i + 1}: {color_names[label]}")

# 将预测的颜色标签转化为颜色名称
color_matrix = [color_names[predicted_labels[i]] for i in range(9)]

# 输出成魔方的排列
for i in range(0, len(color_matrix) - 1, 3):
    print("-" * 15)
    print(
        f"| {color_matrix[i]} | {color_matrix[i + 1]} | {color_matrix[i + 2]} |")
    if i == 6:
        print("-" * 15)

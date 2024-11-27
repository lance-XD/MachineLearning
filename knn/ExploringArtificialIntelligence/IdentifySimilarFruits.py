from sklearn.neighbors import KNeighborsClassifier

# 训练数据
train_data = [
    [155, 7.1], [153, 7.2], [153, 7.1], [150, 7], [150, 7], [149, 6.9],
    [148, 6.8], [144, 6.7], [143, 6.8], [140, 6.8],  # 种类A
    [155, 7.2], [160, 7.4], [162, 7.4], [163, 7.3], [165, 7.3], [165, 7.4],
    [166, 7.3], [168, 7.6], [169, 7.6], [170, 7.5]  # 种类C
]
train_labels = ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
                "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"]  # 标签
# 测试数据
test_data = [[157, 7.3]]

# 创建KNN分类器
k = 3  # 设定K值
knn = KNeighborsClassifier(n_neighbors=k)

# 训练模型
knn.fit(train_data, train_labels)

# 进行预测
predicted_label = knn.predict(test_data)

# 输出结果
print(f"k={k}时，测试数据 {test_data[0]} 的预测分类为: 种类{predicted_label[0]}")
# print(predicted_label[0])

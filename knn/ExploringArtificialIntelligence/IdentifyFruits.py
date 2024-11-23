from sklearn.neighbors import KNeighborsClassifier

# 训练数据
train_data = [
    [150, 7], [160, 7], [140, 6], [155, 7],  # 种类A
    [2000, 25], [1800, 22], [2100, 26], [1900, 24]  # 种类B
]
train_labels = ["A", "A", "A", "A", "B", "B", "B", "B"]  # 标签
# 测试数据
test_data = [[170, 8]]

# 创建KNN分类器
k = 3  # 设定K值
knn = KNeighborsClassifier(n_neighbors=k)

# 训练模型
knn.fit(train_data, train_labels)

# 进行预测
predicted_label = knn.predict(test_data)

# 输出结果
print(f"测试数据 {test_data[0]} 的预测分类为: {predicted_label[0]}")

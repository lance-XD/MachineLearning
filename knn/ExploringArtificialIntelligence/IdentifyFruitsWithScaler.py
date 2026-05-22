from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# 已知种类水果数据(训练数据)
train_data = [
    [155, 7.1], [153, 7.2], [153, 7.1], [150, 7], [150, 7], [149, 6.9],
    [148, 6.8], [144, 6.7], [143, 6.8], [140, 6.8],  # 种类A
    [160, 7.2], [160, 7.4], [162, 7.4], [163, 7.3], [165, 7.3], [165, 7.4],
    [166, 7.3], [168, 7.6], [169, 7.6], [170, 7.5]   # 种类C
]
train_labels = ["A"] * 10 + ["C"] * 10  # 标签

# 未知种类水果数据(测试数据)
test_data = [[157, 7.3]]

# 归一化处理
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

k = 3  # 设定K值
knn = KNeighborsClassifier(n_neighbors=k)   # 创建KNN分类器
knn.fit(train_scaled, train_labels)   # 训练模型

predicted_label = knn.predict(test_scaled)    # 进行预测

# 输出结果
print(f"k={k}时，测试数据 {test_data[0]} 的预测分类为: 种类{predicted_label[0]}")
# print(predicted_label[0])

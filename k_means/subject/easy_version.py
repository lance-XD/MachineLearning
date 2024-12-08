from sklearn.cluster import KMeans

# 数据
data = [[86, 59], [94, 50], [87, 55], [80, 90], [70, 86], [75, 72], [85, 70], [85, 65], [81, 68], [71, 70]]

# 手动选择的初始中心点 (序号2和序号5)
initial_centers = [[94, 50], [70, 86]]

# 创建KMeans模型，n_clusters即为k,=2表示分成两类
# init用以指定初始中心点，可以不指定，默认情况下，K-Means会随机选择聚类中心点
# n_init 会设置执行算法的次数，KMeans 会选择最佳的结果作为最终聚类结果
# K-Means中初始化中心点是随机的，通过random_state可以控制每次运行时生成的随机数是相同的
kmeans = KMeans(n_clusters=2, init=initial_centers, n_init=1, random_state=0)

# 训练模型
kmeans.fit(data)

# 输出每个样本的类别标签
print("每个学生的聚类结果（0或1）:")
print(kmeans.labels_)

# 输出聚类中心（每一类的中心点）
print("\n聚类中心:")
print(kmeans.cluster_centers_)


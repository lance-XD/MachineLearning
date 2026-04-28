from neuralnetworks.src import mnist_loader
from neuralnetworks.src import network4

# 本文件用来检测过拟合，只取前1000张图片训练，然后用测试集来进行验证

# 加载数据: 训练数据，验证数据，测试数据
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# 建立一个3层的神经网络，第一层输入神经元为784个，第二层隐藏层为30个，第三层输出层为10个
# net = network.Network([784, 30, 10])
# 使用交叉熵的损失函数
net = network4.Network([784, 30, 10], cost=network4.CrossEntropyCost)

net.large_weight_initializer()

# 使用随机梯度下降算法进行训练，训练400轮，每批为10组数据，学习率相应减小为0.5，测试数据为test_data的10000张手写数字图片
# net.SGD(training_data=training_data[:1000], epochs=400, mini_batch_size=10, eta=0.5, evaluation_data=test_data,
#         monitor_evaluation_accuracy=True, monitor_training_accuracy=True)

# 通过增大数据量，验证过拟合的幅度，可观察到数据量增大后，过拟合的幅度有所降低
net.SGD(training_data=training_data[:1000], epochs=400, mini_batch_size=10, eta=0.5, evaluation_data=test_data,
        lmbda=0.1, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
        monitor_training_cost=True, monitor_training_accuracy=True)

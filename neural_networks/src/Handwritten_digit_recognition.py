import mnist_loader
import network

# 加载数据: 训练数据，验证数据，测试数据
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# 建立一个3层的神经网络，第一层输入神经元为784个，第二层隐藏层为30个，第三层输出层为10个
net = network.Network([784, 30, 10])
# 使用随机梯度下降算法进行训练，训练30轮，每批为10组数据，学习率为3.0，测试数据为test_data的10000张手写数字图片
net.SGD(training_data=training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

import random
import numpy as np

"""
实现随机梯度下降算法的前馈神经网络，通过反向传播计算梯度
"""


class Network:

    def __init__(self, sizes):
        """
        初始化神经网络，通过sizes参数的形状初始化大小、层数，并随机初始化权重和偏置值
        :param sizes: 每层神经网络包含神经元的数目列表，如[2,3,1]表示输入层有2个神经元，第2层有3个神经元，第3层有1个神经元
        """
        # 层数
        self.num_layers = len(sizes)
        # 各层信息
        self.sizes = sizes
        # 随机生成对应数量的偏置值，第一层为输入神经元，无需偏置值计算。randn(i, 1)生成i行1列个标准正态分布的随机值
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]
        # 生成符合标准正态分布的随机初始权重值。如第1层有i个神经元、第2层有j个神经元，则两层之间需要生成j行i列个权重值
        self.weights = [np.random.randn(j, i) for i, j in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z):
        """
        激活函数sigmoid: 1 / 1 + e^(-z)
        :param z: 计算的数值
        :return: 计算的结果
        """
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        """
        计算sigmoid函数的导数，可证明对于f(x)=1 / 1 + e^(-z), f‘（x）=f(x)(1-f(x))
        :param z: 原始数值
        :return: 原始数值的f'(z)
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self, a):
        """
        进行前馈计算，即用∑w.a+b计算逐层计算每一层的输出
        :param a: 上一层神经网络的输出，本层的输入
        :return: 输出结果，仍保存在a中
        """
        for b, w in zip(self.biases, self.weights):
            # np.dot为向量的点乘，即w.a+b
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        """
        返回测试输出样本中预测正确的数目
        :param test_data: 测试数据
        :return: 预测正确的数量
        """
        # 预测的结果保存在最后一层神经元中，结果为哪一个，哪一个神经元的输出值就最大。通过np.argmax函数找出最大值的索引
        test_result = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        # 如果找到的索引和y一样，说明预测正确。将x==y的布尔值转换为int值，int(True) = 1
        return sum(int(x == y) for x, y in test_result)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        将数据分成批量{mini_batch_size}逐次更新权重和偏置值
        :param training_data: 训练数据，包含输入数据x, 输出y。以（x,y）的形式
        :param epochs: 训练的轮次（随机取完所有的数据一次为一轮）
        :param mini_batch_size: 每次进行随机取样的大小
        :param eta: 学习率η
        :param test_data: 测试数据，如有则会进行验证
        :return: 无
        """
        # 将训练数据转换为列表结构
        training_data = list(training_data)
        # 将测试数据转换为列表结构
        test_data = list(test_data) if test_data else None
        # 测试数据的数量，先转换为列表，再得到列表元素的个数
        n_test = len(test_data) if test_data else 0
        # 输入、输出数据对数，一个输入x,一个输出y为一对
        n = len(training_data)
        # 进行epochs轮训练
        for j in range(epochs):
            # 随机打乱训练数据，从此种方式实现每次随机取样mini_batch_size对数据
            random.shuffle(training_data)
            # 将数据切分成大小为k的块，每块的大小为k + mini_batch_size - k = mini_batch_size
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # 使用本轮的所有块逐次更新神经网络的权重和偏置值
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # 如果设定了测试数据，则使用本轮训练好的神经网络预测所有测试样本，输出计算预测正确的个数/总测试样本数
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} completed.")

    def update_mini_batch(self, mini_batch, eta):
        """
        对一批数据应用梯度下降和反向传播算法更新整个神经网络的权重和偏置
        :param mini_batch: 由若干个(x,y)组成的列表
        :param eta: 学习率η
        :return: 无
        """
        # 用np.zeros生成偏置值、权重对应形状（几行几列）的用0填充的列表,记录梯度信息（小球滚动的方向）。结构和weights、biases一样
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # 通过反向传播，找到w, b的梯度
            delta_nabla_b, delta_nabla_W = self.backprop(x, y)
            # 将梯度信息更新到nabla_w， nabla_b
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_W)]
            # 更新所有的偏置值
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        # 用w - η / m * 梯度（对权重的梯度）计算出更新之后的权重值
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        # 用w - η / m * 梯度（对偏置值的梯度）计算出更新之后的偏置值值
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        反向传播算法
        :param x: 输入数据
        :param y: 输出数据
        :return: 返回一个和weights, biases结构一样的梯度数据
        """
        # 用np.zeros生成偏置值、权重对应形状（几行几列）的用0填充的列表。结构和weights、biases一样
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 本次激活函数计算的结果，初始化为输入
        activation = x
        # 保存为所有激活的结果
        activations = [x]
        # 保存所有的z,运算式为∑w.a+b
        zs = []
        for b, w in zip(self.biases, self.weights):
            # 本层的激活前的结果，z = ∑w.a+b
            z = np.dot(w, activation) + b
            zs.append(z)
            # 计算经过激活函数之后的结果
            activation = self.sigmoid(z)
            # 保存所有激活后的值
            activations.append(activation)
        # 反向传播，从后往前，逐层使用。初始的误差项由损失函数提供
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 从后往前逐层计算权重和偏置值的偏导数，-2为倒数第2层
        for l in range(2, self.num_layers):
            z = zs[-l]
            # 计算导数
            sp = self.sigmoid_prime(z)
            # -l层的误差
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            # -l层的∇𝑤和∇𝑏
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def cost_derivative(self, output_activations, y):
        """
        取得最终输出激活值的偏导数的向量，对于损失函数(∑(a-y)^2) / 2的导数，为a-y
        :param output_activations: 实际的输出值
        :param y: 真实的输出值
        :return: 损失函数的导数
        """
        return output_activations - y

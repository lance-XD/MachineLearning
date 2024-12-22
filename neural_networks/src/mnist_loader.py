"""
加载MNIST图像数据的库
"""

import _pickle as cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """
    加载mnist的图像数据，并以(训练数据，验证数据，测试数据)的格式返回, 图像数据的格式为np.ndarray，即为多维数组。
    mnist中的图片为28*28=784像素
    :return: (训练数据，验证数据，测试数据)，训练数据为50000张手写数字图像及对应的图像数字（0-9），验证数据、测试数据为另外的10000张
    """
    # 打开文件, gzip以二进制编码的方式读取
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    # 从文件中读取内容，cPickle以ASCII编码的方式读取文件
    training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    """
    基于load_data的数据进一步处理，使数据更易于神经网络处理（将y由单个的数据，转换维10维的数据）
    :return:
    """
    tr_d, va_d, te_d = load_data()
    # 将图像转换为784维的数据
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # 将数字的结果转换为10维的结果
    training_results = [vectorized_result(y) for y in tr_d[1]]
    # 只有训练的数据的y需要供神经网络读取，所以封装成（x,y）的形式
    training_data = zip(training_inputs, training_results)
    # 转换验证数据的格式
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    # 验证数据，此时y还是单个数字
    validation_data = zip(validation_inputs, va_d[1])
    # 转换测试数据的格式
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    # 测试数据，此时y还是单个数字
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data


def vectorized_result(j):
    """
    返回一个10维的数据，如果j=5, 则返回[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    :param j: 数值
    :return: 10维的单位向量
    """
    # 初始化为10行1列的全为0.0的数
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def show_sample_image():
    """
    显示一张mnist中的图片作为示例
    :return:
    """
    # 加载训练数据
    training_data, validation_data, test_data = load_data()

    # 提取一个样本图像和对应的标签
    image = training_data[0][0]  # 获取第一个训练样本
    label = training_data[1][0]  # 获取第一个训练样本的标签

    # 将图像数据调整为28x28格式
    image = np.reshape(image, (28, 28))

    # 显示图像
    plt.imshow(image, cmap='gray')  # 使用灰度显示
    plt.title(f"Label: {label}")  # 显示标签作为标题
    plt.axis('off')  # 关闭坐标轴
    plt.show()

# 显示一张示例图片
# show_sample_image()

"""network4.py
~~~~~~~~~~~~~~

基于 network2.py 注释翻译与扩展而来。

这是一个用于前馈神经网络（Feedforward Neural Network）的实现，
使用小批量随机梯度下降（Mini-batch Stochastic Gradient Descent）算法进行训练。

与最基础的 network.py 相比，本版本的主要改进包括：
1. 交叉熵代价函数（Cross-Entropy Cost）：相比传统的二次代价函数，
   交叉熵能让网络在输出与目标相差较大时学习得更快，缓解梯度消失问题。
2. L2 正则化（Regularization）：通过在代价函数中增加权重衰减项，
   防止网络过拟合训练数据，提高泛化能力。
3. 改进的权重初始化：根据每层输入神经元的数量调整初始权重的标准差，
   使得信号在网络前向传播时保持稳定的方差。

代码设计注重简洁性和可读性，未做性能优化，适合初学者理解神经网络的核心原理。

"""

#### 库
# 标准库：Python 内置模块
import json      # 用于将神经网络结构保存为 JSON 格式的文件，或从文件读取
import random    # 用于随机打乱训练数据，使每个训练轮次（epoch）的小批量组成不同
import sys       # 用于动态获取当前模块中的代价类，以便加载网络时恢复代价函数类型

# 第三方库
import numpy as np  # NumPy 是 Python 科学计算的核心库，提供了高效的多维数组和矩阵运算


#### 定义二次代价函数和交叉熵代价函数
# 代价函数（Cost Function）用于衡量神经网络的预测输出与真实目标之间的差距。
# 训练的目标就是通过调整权重和偏置，使代价函数的值尽可能小。
# 这里提供两种代价函数，用户可以在创建网络时选择使用哪一种。

class QuadraticCost(object):
    """二次代价函数（Quadratic Cost），也称为均方误差（Mean Squared Error, MSE）。

    计算公式：C = 0.5 * ||a - y||^2
    其中 a 是网络的输出，y 是期望输出（真实标签），||.|| 表示向量的 L2 范数。

    优点：形式简单，易于理解。
    缺点：当网络输出与目标差距很大且神经元处于饱和区（sigmoid 导数接近 0）时，
          梯度会变得非常小，导致学习速度缓慢（梯度消失）。
    """

    @staticmethod
    def fn(a, y):
        """计算并返回当前代价函数的值。

        参数:
            a (numpy.ndarray): 神经网络的输出向量，形状为 (输出层神经元数, 1)。
            y (numpy.ndarray): 期望的输出向量（真实标签），形状与 a 相同。

        返回:
            float: 代价函数的值，是一个标量，表示预测与真实的差距。
        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """计算输出层的误差 delta，用于反向传播算法。

        在反向传播中，delta 表示代价函数对加权输入 z 的偏导数，
        它告诉我们应该朝哪个方向、以多大程度调整前一层的权重。

        推导过程：
            对于二次代价和 sigmoid 激活函数，
            delta = (dC/da) * (da/dz) = (a - y) * sigmoid'(z)

        参数:
            z (numpy.ndarray): 输出层的加权输入（加权求和后再加偏置）。
            a (numpy.ndarray): 输出层的激活值，即 sigmoid(z)。
            y (numpy.ndarray): 期望的输出向量（真实标签）。

        返回:
            numpy.ndarray: 输出层的误差向量，形状与 z 相同。
        """
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    """交叉熵代价函数（Cross-Entropy Cost）。

    计算公式：C = -sum[ y * ln(a) + (1-y) * ln(1-a) ]

    交叉熵来源于信息论，衡量两个概率分布之间的差异。
    对于分类问题，当网络输出可以被解释为概率时，交叉熵是非常自然的选择。

    核心优势：
        即使在输出 a 严重错误（例如目标为 1 但输出接近 0）的情况下，
        由于误差 delta = a - y（不乘以 sigmoid'），梯度仍然较大，
        网络能够快速学习，避免了二次代价在饱和区的梯度消失问题。
    """

    @staticmethod
    def fn(a, y):
        """计算并返回当前代价函数的值。

        实现细节：
            当 a 非常接近 0 或 1 时，log(a) 或 log(1-a) 可能会趋向负无穷，
            如果此时 y 也恰好对应为 0 或 1，理论上代价应为 0，但计算会得到 nan。
            因此使用 np.nan_to_num 将这些无效值安全地转换为 0.0，保证数值稳定性。

        参数:
            a (numpy.ndarray): 神经网络的输出向量（经过 sigmoid，值域 (0,1)）。
            y (numpy.ndarray): 期望的输出向量（真实标签，通常也是 0 或 1）。

        返回:
            float: 交叉熵代价的值，是一个标量。
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """计算输出层的误差 delta，用于反向传播算法。

        推导结果（对于交叉熵 + sigmoid 的组合）：
            delta = a - y
        这是一个非常简洁的形式，也是交叉熵代价收敛更快的原因之一。

        注意：
            参数 z 在本方法中实际上并未参与计算（因为 sigmoid' 项被约去了），
            但为了保持与 QuadraticCost.delta 接口一致（都是 (z, a, y) 三个参数），
            仍然保留在参数列表中。这样 Network 类可以在不区分代价类型的情况下统一调用。

        参数:
            z (numpy.ndarray): 输出层的加权输入（未使用，仅用于接口统一）。
            a (numpy.ndarray): 输出层的激活值。
            y (numpy.ndarray): 期望的输出向量（真实标签）。

        返回:
            numpy.ndarray: 输出层的误差向量，形状与 a 相同。
        """
        return (a - y)


#### 主网络类
class Network(object):
    """前馈神经网络类。

    一个前馈网络由多层神经元组成，信息从输入层逐层向前传递，
    经过隐藏层，最终到达输出层。每一层与下一层之间通过权重和偏置连接。

    使用示例:
        net = Network([784, 30, 10])  # 输入层784个神经元，隐藏层30个，输出层10个
        net.SGD(training_data, 30, 10, 3.0, lmbda=5.0,
                evaluation_data=test_data,
                monitor_evaluation_accuracy=True)
    """

    def __init__(self, sizes, cost=CrossEntropyCost):
        """初始化神经网络。

        参数:
            sizes (list[int]): 一个列表，定义了网络每一层的神经元数量。
                例如 [2, 3, 1] 表示一个 3 层网络：
                - 第 1 层（输入层）：2 个神经元
                - 第 2 层（隐藏层）：3 个神经元
                - 第 3 层（输出层）：1 个神经元
                列表的长度决定了网络的层数（num_layers），
                第一个元素是输入层，最后一个元素是输出层。
            cost (class, 可选): 要使用的代价函数类，默认为 CrossEntropyCost。
                也可以传入 QuadraticCost 来使用二次代价函数。
                注意这里传入的是类本身，不是实例。

        初始化内容:
            - 记录层数 num_layers 和每层大小 sizes
            - 调用默认权重初始化方法，随机生成权重和偏置
            - 保存代价函数的类引用
        """
        self.num_layers = len(sizes)    # 网络总层数（包括输入层和输出层）
        self.sizes = sizes              # 每层神经元数量的列表
        self.default_weight_initializer()  # 随机初始化权重和偏置
        self.cost = cost                # 代价函数类（CrossEntropyCost 或 QuadraticCost）

    def default_weight_initializer(self):
        """使用改进的方法初始化网络的权重和偏置。

        思路：
            如果权重初始化得太大，神经元的加权输入 z 会很大，
            sigmoid 函数会进入饱和区（导数接近 0），导致学习缓慢。
            如果初始化得太小，信号会在深层网络中逐层衰减。

            这里采用一种常用的启发式方法：
            - 权重 w 服从均值为 0、标准差为 1/sqrt(n_in) 的高斯分布，
              其中 n_in 是连接到该神经元的输入权重数量（即前一层神经元数）。
              这样可以使加权输入 z 的方差保持适中，避免过早饱和。
            - 偏置 b 服从均值为 0、标准差为 1 的高斯分布。

        注意：
            输入层（第 1 层）只是接收数据，不进行计算，因此不需要偏置。
            偏置只存在于第 2 层及以后的层。
        """
        # biases: 为除输入层外的每一层生成一个 (该层神经元数, 1) 的偏置向量
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # weights: 为每对相邻层生成权重矩阵，形状为 (后一层神经元数, 前一层神经元数)
        # 除以 sqrt(x) 进行缩放，x 是前一层神经元数量（输入连接数）
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """使用传统方法初始化权重和偏置（标准差为 1，不做缩放）。

        这种方法与《神经网络与深度学习》一书第 1 章的示例保持一致，
        包含它主要是为了教学对比：
        - 标准差为 1 的初始化往往会导致神经元在训练初期大量饱和，
          使得前几轮的训练效果很差。
        - 改进的初始化（default_weight_initializer）通常效果更好。

        参数和返回值：无（直接修改 self.biases 和 self.weights）。
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # 注意这里没有除以 sqrt(x)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """前向传播：计算网络对给定输入的输出。

        过程：
            输入 a 从第 1 层（输入层）开始，逐层计算：
            z = w · a + b    （加权输入）
            a = sigmoid(z)   （通过激活函数得到该层的输出）
            计算结果作为下一层的输入，直到输出层。

        参数:
            a (numpy.ndarray): 输入向量，形状为 (输入层神经元数, 1)。

        返回:
            numpy.ndarray: 输出层的激活值，形状为 (输出层神经元数, 1)。
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """使用小批量随机梯度下降（Mini-batch Stochastic Gradient Descent）训练神经网络。

        这是神经网络训练的核心方法。算法思路：
        1. 将训练数据随机打乱，划分为若干个小批量（mini-batch）。
        2. 对每个小批量，计算平均梯度，并据此更新权重和偏置。
        3. 重复上述过程指定的轮数（epochs）。
        4. 可选地在每轮结束时评估模型在训练集或验证/测试集上的表现。

        参数:
            training_data (list[tuple]): 训练数据集，是一个列表，其中每个元素是一个元组 (x, y)。
                x 是输入向量（如手写数字图像展平后的像素值），
                y 是对应的期望输出向量（如 one-hot 编码的标签）。
            epochs (int): 训练轮数。每一轮都会遍历全部训练数据一次。
            mini_batch_size (int): 每个小批量包含的样本数量。
                例如设为 10，则每次用 10 个样本来估计梯度并更新参数。
                值太小会导致梯度估计噪声大；值太大则失去随机性优势。
            eta (float): 学习率（Learning Rate），控制每次参数更新的步长。
                太大可能导致震荡不收敛，太小则学习速度过慢。
            lmbda (float, 可选): L2 正则化参数（lambda，避免与 Python 关键字冲突而写作 lmbda）。
                默认 0.0 表示不使用正则化。
                正则化通过在代价函数中增加 (lambda/2n) * sum(||w||^2) 项，
                惩罚过大的权重，从而防止过拟合。
            evaluation_data (list[tuple], 可选): 用于评估的数据集，
                通常是验证集（validation set）或测试集（test set）。
                如果提供，可以在训练过程中监控模型在未见过数据上的表现。
            monitor_evaluation_cost (bool, 可选): 是否监控评估数据上的代价。
            monitor_evaluation_accuracy (bool, 可选): 是否监控评估数据上的准确率。
            monitor_training_cost (bool, 可选): 是否监控训练数据上的代价。
            monitor_training_accuracy (bool, 可选): 是否监控训练数据上的准确率。

        返回:
            tuple: 一个包含 4 个列表的元组：
                (evaluation_cost, evaluation_accuracy, training_cost, training_accuracy)
                每个列表按轮次顺序记录了对应指标的值。
                如果某个监控标志设为 False，对应的列表为空。
        """
        # 如果提供了评估数据，记录其样本数量（用于后续计算准确率比例）
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)  # 训练数据总数量（用于正则化和计算比例）

        # 初始化四个空列表，用于记录每轮结束后的各项指标
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        # 外层循环：遍历每个训练轮次（epoch）
        for j in range(epochs):
            # 步骤 1：随机打乱训练数据，确保每个 epoch 的小批量组成不同，增加随机性
            random.shuffle(training_data)

            # 步骤 2：将训练数据切分为多个小批量
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            # 步骤 3：对每个小批量执行一次梯度下降更新
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            # 步骤 4：本轮训练完成，根据用户设置的标志输出监控信息
            print(f"Epoch {j} training complete")

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy / n}")

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {self.accuracy(evaluation_data) / n_data}")

        # 返回四个监控列表，方便调用者进行后续分析（如绘制学习曲线）
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """使用反向传播算法计算梯度，并更新一次网络的权重和偏置。

        思路：
            反向传播（Backpropagation）是计算梯度的高效算法。
            对于小批量中的每个样本，我们先通过反向传播计算出该样本
            对代价函数的权重梯度和偏置梯度，然后将它们累加起来，
            最后除以批量大小（隐式地，通过 eta/len(mini_batch) 实现），
            得到平均梯度，并用梯度下降法更新参数。

        参数:
            mini_batch (list[tuple]): 一个小批量的数据，即若干个 (x, y) 元组的列表。
            eta (float): 学习率，控制更新步长。
            lmbda (float): L2 正则化参数。
            n (int): 整个训练数据集的总样本数（用于正则化项的缩放，lmbda/n）。
        """
        # nabla_b 和 nabla_w 分别存储代价函数对偏置和权重的梯度
        # 初始化为与 biases 和 weights 形状相同的零数组
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 对小批量中的每个样本分别计算梯度，并累加
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 逐层累加偏置梯度
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # 逐层累加权重梯度
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 使用梯度下降更新权重：
        # 新权重 = (1 - eta*lmbda/n) * 旧权重 - (eta/批量大小) * 梯度
        # 其中 (1 - eta*lmbda/n) 这一项来自 L2 正则化，它使权重每次更新都略微衰减
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]

        # 偏置的更新不涉及正则化（通常偏置不做过拟合惩罚）
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """反向传播算法：计算单个样本 (x, y) 对代价函数的梯度。

        反向传播的核心思想是利用链式法则，从输出层向输入层逐层计算误差，
        从而高效地得到每一层权重和偏置的梯度。

        算法分为两个阶段：
        1. 前向传播（Feedforward）：从输入层计算到输出层，
           并保存每一层的加权输入 z 和激活值 a。
        2. 反向传播（Backward pass）：从输出层开始，计算误差 delta，
           然后逐层向前传播，得到每层的 delta，进而求出权重和偏置的梯度。

        参数:
            x (numpy.ndarray): 单个输入样本。
            y (numpy.ndarray): 输入样本对应的期望输出（真实标签）。

        返回:
            tuple: (nabla_b, nabla_w)
                nabla_b 是每层偏置的梯度列表，
                nabla_w 是每层权重的梯度列表。
                它们的形状分别与 self.biases 和 self.weights 一致。
        """
        # 初始化梯度数组（全零）
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # ========== 第一阶段：前向传播 ==========
        # 从输入层开始，逐层计算每一层的 z（加权输入）和 a（激活值）
        activation = x  # 当前层的激活值，初始为输入 x
        activations = [x]  # 列表：按顺序保存每一层的激活值（从输入层到输出层）
        zs = []            # 列表：按顺序保存每一层的加权输入 z（不包含输入层）

        for b, w in zip(self.biases, self.weights):
            # 计算当前层的加权输入：z = w · activation + b
            z = np.dot(w, activation) + b
            zs.append(z)
            # 通过 sigmoid 激活函数得到该层的输出
            activation = sigmoid(z)
            activations.append(activation)

        # ========== 第二阶段：反向传播 ==========
        # 1. 计算输出层的误差 delta
        #    delta^L = cost.delta(z^L, a^L, y)
        #    对于交叉熵代价，这简化为 (a - y)
        delta = (self.cost).delta(zs[-1], activations[-1], y)

        # 输出层的偏置梯度就是 delta，权重梯度是 delta 与前一激活值的乘积
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 2. 从倒数第二层开始，逐层向前计算误差和梯度
        #    注意这里的变量 l 表示"从后往前数的第几层"：
        #    l=1 对应输出层，l=2 对应倒数第二层（即最后一个隐藏层），以此类推。
        #    这与教材第 2 章的编号方式略有不同，这里利用了 Python 列表负索引的便利性。
        for l in range(2, self.num_layers):
            z = zs[-l]                       # 当前层的加权输入
            sp = sigmoid_prime(z)            # 当前层 sigmoid 的导数
            # 将后一层的误差反向传播到当前层：
            # delta^l = ((w^{l+1})^T · delta^{l+1}) ⊙ sigmoid'(z^l)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            # 当前层的偏置梯度
            nabla_b[-l] = delta
            # 当前层的权重梯度：delta 与前一层的激活值的乘积
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """计算神经网络在给定数据集上的分类准确率。

        判断方法：
            对于每个输入，神经网络会输出一个向量（输出层各神经元的激活值）。
            我们认为激活值最大的那个神经元对应的类别就是网络的预测结果。
            如果预测类别与真实标签一致，则计为正确。

        参数:
            data (list[tuple]): 待评估的数据集，每个元素为 (x, y)。
            convert (bool): 是否需要对标签 y 进行转换。
                - 对于验证/测试数据（通常情况），y 是整数标签（如 0~9），设为 False。
                - 对于训练数据，y 是向量化的 one-hot 编码（如 (10,1) 的向量），
                  需要先通过 np.argmax 转换为整数标签，因此设为 True。
                使用不同表示的原因是效率：训练时计算代价需要向量形式，
                而评估准确率时只需要整数比较，分开处理可以加速计算。

        返回:
            int: 预测正确的样本数量（不是比例，如果需要比例请除以 len(data)）。
        """
        if convert:
            # 训练数据：y 是 one-hot 向量，需要先取出最大值的索引
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            # 验证/测试数据：y 已经是整数标签
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        # 统计预测值与真实值相等的个数
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """计算数据集上的总代价（包含正则化项）。

        总代价 = 数据代价 + 正则化代价
        其中：
            数据代价 = (1/n) * sum( cost.fn(a_i, y_i) )
            正则化代价 = 0.5 * (lmbda / n) * sum( ||w||^2 )

        参数:
            data (list[tuple]): 数据集，每个元素为 (x, y)。
            lmbda (float): L2 正则化参数。
            convert (bool): 是否需要转换标签格式。
                - 对于训练数据（通常情况），设为 False。
                - 对于验证/测试数据，y 是整数标签，需要转换为 one-hot 向量，设为 True。
                与 accuracy 方法的 convert 含义相反，请注意区分。

        返回:
            float: 总代价的值。
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            # 如果是验证/测试数据，将整数标签转换为 one-hot 向量，以便计算代价
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)

        # 添加 L2 正则化项：惩罚所有权重的平方和
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        """将神经网络的当前状态保存到 JSON 文件中。

        保存的内容包括：
            - 网络结构（sizes）：各层神经元数量
            - 权重（weights）：转换为 Python 列表格式
            - 偏置（biases）：转换为 Python 列表格式
            - 代价函数名称（cost）：字符串形式，如 "CrossEntropyCost"

        参数:
            filename (str): 保存文件的路径和名称（如 "network.json"）。
        """
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### 加载网络
def load(filename):
    """从 JSON 文件中加载神经网络。

    读取 save 方法保存的文件，恢复网络结构、权重、偏置和代价函数类型，
    返回一个可以直接使用的 Network 实例。

    参数:
        filename (str): 要加载的 JSON 文件路径。

    返回:
        Network: 恢复后的神经网络实例。
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    # 根据保存的字符串名称，从当前模块中动态获取对应的代价类
    cost = getattr(sys.modules[__name__], data["cost"])
    # 使用保存的网络结构和代价函数创建新网络实例
    net = Network(data["sizes"], cost=cost)
    # 恢复权重和偏置（从列表转回 numpy 数组）
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### 其他函数
def vectorized_result(j):
    """将一个整数标签转换为 10 维的 one-hot 向量。

    在 MNIST 手写数字识别任务中，数字类别为 0~9。
    神经网络的输出层有 10 个神经元，期望输出是一个 10 维向量，
    其中正确数字对应的位置为 1.0，其余为 0。

    例如：
        vectorized_result(3) -> [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]^T

    参数:
        j (int): 数字类别（0 到 9 之间的整数）。

    返回:
        numpy.ndarray: 形状为 (10, 1) 的单位向量。
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    """sigmoid 激活函数。

    数学公式：sigma(z) = 1 / (1 + e^{-z})

    作用：
        将任意实数值映射到 (0, 1) 区间，可以解释为概率。
        它是神经网络中引入非线性能力的关键，没有激活函数，
        多层网络将等价于单层线性变换。

    参数:
        z (numpy.ndarray): 加权输入（可以是标量、向量或矩阵）。

    返回:
        numpy.ndarray: sigmoid 函数的输出，形状与 z 相同。
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """sigmoid 函数的导数。

    数学推导：
        sigma'(z) = sigma(z) * (1 - sigma(z))

    这个简洁的形式使得反向传播计算非常方便。
    当 z 很大（正值）时，sigma(z)≈1，导数接近 0（饱和）。
    当 z 很小（负值）时，sigma(z)≈0，导数也接近 0（饱和）。
    当 z=0 时，sigma(z)=0.5，导数最大为 0.25。

    参数:
        z (numpy.ndarray): 加权输入。

    返回:
        numpy.ndarray: sigmoid 导数的值，形状与 z 相同。
    """
    return sigmoid(z) * (1 - sigmoid(z))

import numpy as np

#
# # 创建一个三维数组
# array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
#
# # 获取三维数组的转置
# transposed_3d = array_3d.T
# print("原始三维数组:\n", array_3d)
# print("转置后的三维数组:\n", transposed_3d)

#
# # 创建两个不同形状的二维数组
# array1_2dx3 = np.array([[1, 2, 3], [4, 5, 6]])
# array2_1x3 = np.array([[7, 8, 9]])
#
# # 垂直组合不同形状的数组
# v_complex_combined = np.vstack((array1_2dx3, array2_1x3))
# print("垂直组合不同形状的数组:\n", v_complex_combined)
# # 输出:
# # 垂直组合不同形状的数组:
# # [[1 2 3]
# #  [4 5 6]
# #  [7 8 9]]


# # 创建一个二维数组
# array_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
#
# # 水平分割数组
# hsplit_arrays = np.hsplit(array_2d, 2)
# print("水平分割后的数组:", hsplit_arrays)
# # 输出: 水平分割后的数组: [array([[1, 2],
# #                                  [5, 6]]), array([[3, 4],
# #                                                  [7, 8]])]


# # 创建两个矩阵
# matrix1 = np.array([[1, 2], [3, 4]])
# matrix2 = np.array([[5, 6], [7, 8]])
#
# # 矩阵乘法
# matrix_product = np.dot(matrix1, matrix2)
# print("矩阵乘法结果:\n", matrix_product)
# # 输出:
# # 矩阵乘法结果:
# # [[19 22]
# #  [43 50]]
#
# # 或者使用@符号进行矩阵乘法
# matrix_product_at = matrix1 @ matrix2
# print("使用@符号的矩阵乘法结果:\n", matrix_product_at)
# # 输出:
# # 使用@符号的矩阵乘法结果:
# # [[19 22]
# #  [43 50]]
#
# # 矩阵转置
# matrix_transpose = matrix1.T
# print("矩阵转置结果:\n", matrix_transpose)
# # 输出:
# # 矩阵转置结果:
# # [[1 3]
# #  [2 4]]


#
# # 创建一个二维NumPy数组
# numpy_array_2d = np.array([[1, 2, 3], [4, 5, 6]])
#
# # 将二维数组转换为嵌套列表
# python_list_2d = numpy_array_2d.tolist()
# print("转换后的嵌套Python列表:\n", python_list_2d)
# # 输出:
# # 转换后的嵌套Python列表:
# # [[1, 2, 3], [4, 5, 6]]

# # 创建一个包含中文的字符串数组
# str_array = np.array(['你好', '世界', 'Python', '数据分析'], dtype=np.unicode_)
# print("字符串数组:", str_array)
# # 输出: 字符串数组: ['你好' '世界' 'Python' '数据分析']


# import numpy as np
#
# # 假设的股票历史价格（假设10天的价格）
# stock_prices = np.array([100, 101, 102, 103, 102, 101, 100, 98, 97, 96])
#
# # 计算日均值
# mean_price = np.mean(stock_prices)
# print("股票的平均价格:", mean_price)
#
# # 计算标准差作为波动率的指标
# std_deviation = np.std(stock_prices)
# print("股票价格的标准差（波动率）:", std_deviation)
#
# # 简单的趋势分析（基于最近三天）
# latest_prices = stock_prices[-3:]
# trend = "上升" if latest_prices[-1] > latest_prices[0] else "下降"
# print("最近三天股价趋势:", trend)
#
# # 基于简单的历史均值预估下一天价格
# # 这里仅作为示例，实际预测应该使用更复杂的模型
# next_day_estimate = mean_price
# print("预估的下一天股价:", next_day_estimate)


# def median(lst):
#     n = len(lst)
#     s = sorted(lst)
#     return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]
#
#
# nums = [3, 1, 4, 2, 5]
# print(median(nums))  # 输出：3

#
# import math
#
# def standard_deviation(lst):
#     n = len(lst)
#     mean = sum(lst) / n
#     variance = sum((x - mean) ** 2 for x in lst) / n
#     return math.sqrt(variance)
#
# nums = [1, 2, 3, 4, 5]
# print(standard_deviation(nums)) # 输出：1.4142135623730951

# import numpy as np
#
# arr1 = np.array(range(6))
# result = arr1 > 3
# print(result)
#
#
# # 复合条件运算
# arr_complex = np.where((arr1 > 2) & (arr1 < 5), "2到4之间", "其他")
# print(f"复合条件选择结果: {arr_complex}")
#
#
# # 布尔数组索引
# arr_bool_idx = arr1[arr1 > 3]
# print(f"大于3的元素: {arr_bool_idx}")

# import numpy as np
#
# def simple_moving_average(data, window_size):
#     """计算简单移动平均线"""
#     if len(data) < window_size:
#         return None  # 数据长度小于窗口大小时返回None
#     # 通过与1向量进行卷积操作来求和
#     sma = np.convolve(data, np.ones(window_size), 'valid') / window_size
#
#     return sma
#
# # 示例数据
# data = [1, 2, 3, 7, 8, 9, 10]
# window_size = 3
#
# # 计算SMA
# sma_result = simple_moving_average(data, window_size)
# print(f"简单移动平均线结果: {sma_result}")

# import pandas as pd
#
# data = [1, 2, 3, 7, 8, 9, 10]
# window_size = 3
# # 将数据转换为Pandas序列
# data_series = pd.Series(data)
#
# # 使用rolling()和mean()计算移动平均线
# sma_pandas = data_series.rolling(window=window_size).mean()
# print(f"Pandas中的简单移动平均线结果: \n{sma_pandas}")

# import pandas as pd
#
# def exponential_moving_average(data, span):
#     """计算指数移动平均线"""
#     data_series = pd.Series(data)
#     ema = data_series.ewm(span=span, adjust=False).mean()
#     return ema
#
# # 示例数据
# data = [1, 2, 3, 4, 8, 9, 10]
# span = 3
#
# # 计算EMA
# ema_result = exponential_moving_average(data, span)
# print(f"指数移动平均线结果: \n{ema_result}")


# import pandas as pd
#
#
# def calculate_atr(data, period=3):
#     """计算ATR"""
#     high_low = data['High'] - data['Low']
#     high_close = abs(data['High'] - data['Close'].shift())
#     low_close = abs(data['Low'] - data['Close'].shift())
#
#     # 真实范围（TR）
#     tr = pd.DataFrame([high_low, high_close, low_close]).max()
#
#     # ATR
#     atr = tr.rolling(window=period).mean()
#     return atr
#
#
# # 示例数据（股票的高、低、收盘价）
# data = pd.DataFrame({
#     'High': [120, 125, 130, 128, 134],
#     'Low': [115, 120, 125, 126, 129],
#     'Close': [118, 124, 128, 127, 133]
# })
#
# # 计算ATR
# atr_result = calculate_atr(data)
# print(f"ATR结果: \n{atr_result}")


import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 3, 4, 5, 5, 6, 7, 8, 9, 10])

# # 绘制数据点
# plt.scatter(x, y)
#
# # 绘制趋势线
# z = np.polyfit(x, y, 1)  # 一阶多项式拟合，即线性拟合
# p = np.poly1d(z)
# plt.plot(x, p(x), "r--")  # 绘制红色虚线
#
# plt.show()

# from sklearn.linear_model import LinearRegression
#
# # 将数据转换为适合线性模型的格式
# X = x[:, np.newaxis]
#
# # 创建并拟合模型
# model = LinearRegression().fit(X, y)
#
# # 预测
# y_pred = model.predict(X)
#
# # 绘制数据点和回归线
# plt.scatter(X, y)
# plt.plot(X, y_pred, "g-")  # 绘制绿色实线
#
# plt.show()


# # 使用set函数
# arr = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
# unique_set = set(arr)
# print(f"使用set函数的唯一值: {unique_set}")
#
# # 使用NumPy的unique方法
# import numpy as np
# unique_array = np.unique(arr)
# print(f"使用NumPy unique方法的唯一值: {unique_array}")

# a = {1, 2, 3, 4, 5}
# b = {4, 5, 6, 7, 8}
#
# # 交集
# intersection = a & b
# print(f"交集: {intersection}")
#
# # 并集
# union = a | b
# print(f"并集: {union}")
#
# # 差集
# difference = a - b
# print(f"A与B的差集: {difference}")

# a = np.array([1, 2, 3, 4, 5])
# b = np.array([4, 5, 6, 7, 8])
#
# # 交集
# np_intersection = np.intersect1d(a, b)
# print(f"NumPy交集: {np_intersection}")
#
# # 并集
# np_union = np.union1d(a, b)
# print(f"NumPy并集: {np_union}")
#
# # 差集
# np_difference = np.setdiff1d(a, b)
# print(f"NumPy中A与B的差集: {np_difference}")

# import numpy as np
#
# # 生成单个随机数
# rand_num = np.random.rand()
# print(f"单个随机数: {rand_num}")
#
# # 生成一个随机数数组
# rand_nums = np.random.randn(5)
# print(f"随机数数组: {rand_nums}")


# import numpy as np
# import matplotlib.pyplot as plt
#
# np.random.seed(0)
# n_steps = 1000  # 步数
# walk = np.random.choice([-1, 1], size=n_steps)  # 每步走-1或1
# path = walk.cumsum()  # 累计和
#
# plt.plot(path)
# plt.title("Single Random Walk")
# plt.show()

# n_walks = 5  # 漫步的数量
# n_steps = 1000
#
# # 创建一个二维数组：行表示不同的漫步，列表示每步
# walks = np.random.choice([-1, 1], size=(n_walks, n_steps))
# paths = walks.cumsum(axis=1)  # 沿着每一行计算累计和
#
# # 绘制所有的随机漫步
# for i in range(n_walks):
#     plt.plot(paths[i])
#
# plt.title(f"{n_walks} Random Walk")
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# np.random.seed(0)
#
# n_days = 100  # 模拟的天数
# initial_price = 100  # 股票的初始价格
# daily_fluctuation = 0.02  # 每日价格波动幅度
#
# # 模拟股价变化：每天的变化为-0.02到0.02之间的随机值
# price_changes = np.random.uniform(-daily_fluctuation, daily_fluctuation, n_days)
# price_path = initial_price * np.cumprod(1 + price_changes)
#
# # 绘制股价随机漫步的路径
# plt.figure(figsize=(12, 6))
# plt.plot(price_path, label='Simulated Stock Price')
# plt.title("Stock Price Random Walk Simulation")
# plt.xlabel("Days")
# plt.ylabel("Stock Price")
# plt.legend()
# plt.grid(True)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 生成示例气象数据
np.random.seed(0)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
temperatures = np.random.normal(loc=20, scale=5, size=100)  # 假设平均气温20度，标准差5度
rainfall = np.random.normal(loc=50, scale=15, size=100)  # 假设平均降雨量50mm，标准差15mm
rainfall[rainfall < 0] = 0  # 确保降雨量不为负

# 创建DataFrame
weather_data = pd.DataFrame({
    'Date': dates,
    'Temperature': temperatures,
    'Rainfall': rainfall
})

# 设置日期为索引
weather_data.set_index('Date', inplace=True)

# 数据可视化
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(weather_data['Temperature'], label='Temperature')
plt.title('Daily Temperature')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(weather_data['Rainfall'], label='Rainfall', color='orange')
plt.title('Daily Rainfall')
plt.legend()

plt.tight_layout()
plt.show()

# 时间序列分析 - 月平均温度和降雨量
monthly_data = weather_data.resample('ME').mean()

plt.figure(figsize=(10, 5))
plt.plot(monthly_data['Temperature'], label='Monthly Avg Temperature', color='blue')
plt.plot(monthly_data['Rainfall'], label='Monthly Avg Rainfall', color='orange')
plt.title('Monthly Average Temperature and Rainfall')
plt.legend()
plt.show()

# 相关性分析
plt.figure(figsize=(6, 6))
sns.heatmap(weather_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
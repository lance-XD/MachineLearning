# import numpy as np
#
# # 创建一个NumPy数组
# array_example = np.array([1, 2, 3, 4, 5])
#
# # 数组运算
# array_plus_10 = array_example + 10  # [11, 12, 13, 14, 15]
#
# # 数组形状变换
# reshaped_array = array_example.reshape(1, 5)  # [[1, 2, 3, 4, 5]]
#
# print(reshaped_array)

import time
import numpy as np
#
# # 创建一个大型的Python列表和NumPy数组
# large_list = list(range(1000000))
# large_array = np.array(large_list)
#
# # 测试Python列表性能
# start_time = time.time()
# list_sum = sum(large_list)
# end_time = time.time()
# print("Python List Time:", end_time - start_time)
#
# # 测试NumPy数组性能
# start_time = time.time()
# array_sum = np.sum(large_array)
# end_time = time.time()
# print("NumPy Array Time:", end_time - start_time)

# Python List Time: 0.010301828384399414
# NumPy Array Time: 0.0012760162353515625

# # 创建一个三维数组
# array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
#
# # 显示三维数组的属性
# print("三维数组的形状:", array_3d.shape)  # 输出 (2, 2, 2)
# print("三维数组的维数:", array_3d.ndim)  # 输出 3
# print("三维数组的数据类型:", array_3d.dtype)  # 输出 int32 或 int64


# # 创建一个二维数组
# array_2d = np.array([[1, 2], [3, 4]])
#
# # 二维数组与标量相加
# add_result_2d = array_2d + 5
# print("二维数组加5的结果:\n", add_result_2d)

# # 创建整型数组
# array_int32 = np.array([1, 2, 3], dtype=np.int32)
# array_int64 = np.array([1, 2, 3], dtype=np.int64)
# print("32位整型数组:", array_int32)
# print("64位整型数组:", array_int64)


# # 创建一个结构化数组的dtype
# dt = np.dtype([('age', np.int8), ('height', np.float32)])
# print("结构化数据类型:", dt)
#
# # 使用结构化数据类型创建数组
# array_structured = np.array([(10, 1.75), (12, 1.65)], dtype=dt)
# print("结构化数组:\n", array_structured)
#
# # 访问结构化数组的字段
# ages = array_structured['age']
# heights = array_structured['height']
# print("年龄:", ages)
# print("身高:", heights)

import numpy as np
#
# # 创建一个一维数组
# array_1d = np.array([10, 20, 30, 40, 50])
#
# # 生成布尔数组
# bool_index = array_1d > 30
# print("布尔数组:", bool_index)
# # 输出: 布尔数组: [False False False  True  True]
#
# # 使用布尔数组进行索引
# selected_elements = array_1d[bool_index]
# print("选中的元素:", selected_elements)
# # 输出: 选中的元素: [40 50]
#
# # 使用布尔型索引修改数组中的元素
# array_1d[bool_index] = 100
# print("修改后的数组:", array_1d)
# # 输出: 修改后的数组: [ 10  20  30 100 100]

# 创建一个二维数组
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 在二维数组中使用花式索引
row_indices = [0, 2]
col_indices = [1, 2]
selected_elements_2d = array_2d[row_indices, col_indices]
print("二维数组中使用花式索引选择的元素:\n", selected_elements_2d)
# 输出:
# 二维数组中使用花式索引选择的元素:
# [2 9]
import pandas as pd

# # 使用列表创建Series
# data = [1, 2, 3, 4]
# # series = pd.Series(data)
# # print(series)
#
# # 使用自定义索引
# series = pd.Series(data, index=['a', 'b', 'c', 'd'])
# print(series)

# # 使用字典创建Series
# data_dict = {'a': 1, 'b': 2, 'c': 3}
# series = pd.Series(data_dict)
# print(series)
#
# print(series[series > 2])

# import pandas as pd
#
# # 使用字典创建DataFrame
# data = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
#         'Age': [28, 34, 29, 32],
#         'City': ['New York', 'Paris', 'Berlin', 'London']}
# df = pd.DataFrame(data)
# # print(df)
#
# # 访问行
# # print(df.loc[0])
#
# df['Salary'] = [70000, 80000, 75000, 65000]
# print(df)
#
# # # 修改数据
# # df.at[0, 'Age'] = 29
# # print(df)
#
# # 基于条件的筛选
# # print(df[df['Age'] > 30])
#
# # # 根据某列排序
# # print(df.sort_values(by='Age'))
# #
# # # 处理缺失数据
# # df['Manager'] = pd.Series(['Tom', 'Bob'])
# # print(df.fillna('N/A'))
#
# # 数据聚合
# print(df.groupby('City')[['Age', 'Salary']].mean())
#
# import pandas as pd
#
# # 创建 Series，使用字典
# data = {'a': 10, 'b': 20, 'c': 30, 'd': 40}
# series_from_dict = pd.Series(data)

# # 修改单个值
# series_from_dict['a'] = 50
# print(series_from_dict)

# 批量修改
# series_from_dict.replace({10: 30, 40: 50}, inplace=True)
# print(series_from_dict)
#
# # 添加缺失数据
# series_with_nan = series_from_dict.reindex(['a', 'b', 'e'])
# print(series_with_nan)
#
# # 处理缺失数据
# print(series_with_nan.fillna(0))

# 条件筛选
# print(series_from_dict[series_from_dict > 20])

# import pandas as pd
#
# # 创建二维列表
# data_list = [
#     ['Alice', 25, 'New York'],
#     ['Bob', 30, 'San Francisco'],
#     ['Charlie', 35, 'Los Angeles']
# ]
#
# # 指定列名
# columns = ['Name', 'Age', 'City']
#
# # 从列表创建DataFrame
# df_from_list = pd.DataFrame(data_list, columns=columns)
# # print(df_from_list)
#
# # 使用insert()在指定位置插入新列
# df_from_list.insert(2, 'Department', ['HR', 'IT', 'Finance'])
# print(df_from_list)
#
#
# # # 使用drop()删除列
# # df_from_list.drop('City', axis=1, inplace=True)
# # print(df_from_list)
#
# # 使用del关键字删除列
# del df_from_list['City']
# print(df_from_list)


# import pandas as pd
#
# # 创建DataFrame
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Age': [25, 30, 35],
#     'City': ['New York', 'San Francisco', 'Los Angeles']
# }
#
# df = pd.DataFrame(data, index=['a', 'b', 'c'])
# # print(df)
#
# # # 使用loc访问单个行
# # first_row_loc = df.loc['a']
# # print(first_row_loc)
# #
# # # Name       Alice
# # # Age           25
# # # City    New York
# # # Name: 0, dtype: object
# #
# # # 使用iloc访问单个行
# # first_row_iloc = df.iloc[0]
# # print(first_row_iloc)
#
# # 使用loc方法修改行值
# df.loc["b", ['Age', 'City']] = [37, 'San Diego']
# # print(df)
#
# # # 使用describe()
# # description = df.describe()
# # print(description)
#
# # 计算非空值数量
# # count_values = df.count()
# # print(count_values)
# # 按照Age列升序排序
# # sorted_df = df.sort_values(by='Age')
# # print(sorted_df)
#
# # 按照行索引降序排序
# sorted_index_df = df.sort_index(axis=0, ascending=False)
# print(sorted_index_df)
#
# # # 迭代列
# # for label, series in df.items():
# #     print(f"Column: {label}")
# #     print(series)
#
# # 迭代行
# for index, row in df.iterrows():
#     print(f"Index: {index}, Name: {row['Name']}, Age: {row['Age']}, City: {row['City']}")

# import pandas as pd
#
# # 创建DataFrame
# data = {
#     'A': [1, 2, 3],
#     'B': [4, 5, 6]
# }
#
# df = pd.DataFrame(data, index=['a', 'b', 'c'])
#
# # 重建索引，添加新的索引标签
# df_reindexed = df.reindex(['a', 'b', 'c', 'd'], fill_value=0)
# print(df_reindexed)


# import pandas as pd
# import numpy as np
#
# # 创建带有缺失值的DataFrame
# data = {
#     'A': [1, 2, np.nan, 4],
#     'B': [5, np.nan, np.nan, 8]
# }
#
# df_missing = pd.DataFrame(data)
#
# print(df_missing)
#
# #      A    B
# # 0  1.0  5.0
# # 1  2.0  NaN
# # 2  NaN  NaN
# # 3  4.0  8.0
#
# # 使用常量值填充所有缺失值
# df_filled_constant = df_missing.fillna(0)
# print(df_filled_constant)
#
# #      A    B
# # 0  1.0  5.0
# # 1  2.0  0.0
# # 2  0.0  0.0
# # 3  4.0  8.0
#
# # 使用字典指定不同列的填充值
# fill_values = {'A': df_missing['A'].mean(), 'B': df_missing['B'].median()}
# df_filled_dict = df_missing.fillna(value=fill_values)
# print(df_filled_dict)
#
# #           A    B
# # 0  1.000000  5.0
# # 1  2.000000  6.5
# # 2  2.333333  6.5
# # 3  4.000000  8.0
#
# # 向前填充缺失值
# df_ffill = df_missing.ffill()
# # df_bfill = df_missing.bfill()
# print(df_ffill)
#
# #      A    B
# # 0  1.0  5.0
# # 1  2.0  5.0
# # 2  2.0  5.0
# # 3  4.0  8.0


# import pandas as pd
#
# # 创建DataFrame
# data = {
#     'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
#     'City': ['A', 'B', 'A', 'B'],
#     'Value': [10, 20, 15, 25]
# }
#
# df = pd.DataFrame(data)
#
# # 使用pivot进行数据透视
# pivot_df = df.pivot(index='Date', columns='City', values='Value')
# print(pivot_df)
#
# # City           A     B
# # Date
# # 2025-01-01  10.0   NaN
# # 2025-01-02   NaN  20.0
# # 2025-01-03  15.0   NaN
# # 2025-01-04   NaN  25.0
#
# # 使用pivot_table进行数据透视，计算每个城市的平均值
# pivot_table_df = df.pivot_table(index='Date', columns='City', values='Value', aggfunc='mean')
# print(pivot_table_df)
#
# # City           A     B
# # Date
# # 2025-01-01  10.0   NaN
# # 2025-01-02   NaN  20.0
# # 2025-01-03  15.0   NaN
# # 2025-01-04   NaN  25.0


import pandas as pd

# 创建时间序列
date_rng = pd.date_range(start='2025-01-01', end='2025-01-10', freq='D')
time_series = pd.Series(range(len(date_rng)), index=date_rng)
print(time_series)


# 选择特定时间范围的数据
selected_data = time_series['2025-01-03':'2025-01-07']
print(selected_data)

# 将数据从天转换为周
weekly_resampled = time_series.resample('W').sum()
print(weekly_resampled)

# 计算滚动平均
rolling_mean = time_series.rolling(window=3).mean()
print(rolling_mean)


# 合并两个时间序列
merged_series = pd.concat([time_series, rolling_mean], axis=1)
print(merged_series)
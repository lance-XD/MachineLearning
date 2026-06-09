from sklearn import linear_model
import numpy as np

linear = linear_model.LinearRegression()

y = np.array([165000, 55720, 68550, 65290, 71290, 72950, 78950, 103450])
x = np.array([110, 68, 76, 76, 76, 76, 86, 100])

# reshape(-1, 1)把任意长度的一维数据"竖起来"变成一列
linear.fit(x.reshape(-1, 1), y)

k0 = linear.coef_[0]
b0 = linear.intercept_

print(f"k0={k0:.2f}, b0={b0:.2f}")

# reshape(-1, 1)把任意长度的一维数据"竖起来"变成一列
# res = linear.predict(np.array([135958]).reshape(-1, 1))
# print(res)

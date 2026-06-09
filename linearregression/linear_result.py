import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# ===================== 中文显示设置 =====================
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
# ======================================================

# 数据
x = np.array([110, 68, 76, 76, 76, 76, 86, 100])
y = np.array([165000, 55720, 68550, 65290, 71290, 72950, 78950, 103450])

# 线性回归
linear = linear_model.LinearRegression()
linear.fit(x.reshape(-1, 1), y)
k0 = linear.coef_[0]
b0 = linear.intercept_
print(f"k0={k0:.2f}, b0={b0:.2f}")

# 绘图
plt.figure(figsize=(10, 6))

# 散点
plt.scatter(x, y, color='#2E86AB', s=70, alpha=0.8, label='样本数据')

# 回归直线：在 x 的取值范围内生成连续点
x_line = np.linspace(x.min() - 5, x.max() + 5, 200)
y_line = k0 * x_line + b0
plt.plot(x_line, y_line, color='red', linewidth=2,
         label=f'回归直线: y = {k0:.2f}x + {b0:.2f}')

plt.title('汽车功率与价格线性回归', fontsize=14, fontweight='bold')
plt.xlabel('功率（千瓦）', fontsize=12)
plt.ylabel('价格（元）', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
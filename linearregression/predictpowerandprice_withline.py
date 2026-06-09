import matplotlib.pyplot as plt
import numpy as np

# ===================== 【无报错】解决中文显示 =====================
plt.rcParams["font.family"] = "SimHei"    # 只使用系统自带黑体
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
# ==================================================================

# 功率与价格数据
power = [111, 68, 76, 76, 76, 76, 86, 100, 68, 101, 101, 68, 69, 97]
price = [165000, 55720, 68550, 65290, 71290, 72950, 78950, 103450,
         51950, 109450, 118450, 53890, 54990, 95490]

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(power, price, color='#2E86AB', s=70, alpha=0.8, label='样本数据')

# 添加直线 y = -730x + 105360
x_line = np.linspace(min(power)-5, max(power)+5, 200)  # 稍微扩展范围
y_line = -730 * x_line + 105360
plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2,
         label='$y = -730x + 105360$')

plt.title('汽车功率与价格散点图', fontsize=14, fontweight='bold')
plt.xlabel('功率（千瓦）', fontsize=12)
plt.ylabel('价格（元）', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.show()
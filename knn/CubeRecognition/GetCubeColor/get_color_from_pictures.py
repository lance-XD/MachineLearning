import cv2
import numpy as np

# 加载图像文件,采用相对路径
image = cv2.imread('../RubiksCubePicture/cube_image_1.png')
# 缩放图像成300x300，便于处理
image_resized = cv2.resize(image, (300, 300))

# cvtColor会将图像从 BGR 色彩空间（默认的 OpenCV 图像格式）转换为 RGB 色彩空间。
# 转换后的 image_rgb 变量是一个三维的 NumPy 数组，其中每个像素点的 RGB 值已经按顺序存储。
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
# print(image_rgb)

# 定义小方块的坐标中心（替换为实际坐标）
color_regions = [(50, 50), (150, 50), (250, 50), (50, 150), (150, 150), (250, 150), (50, 250), (150, 250), (250, 250)]

# 提取每个小方块的平均RGB值
colors = []
for (x, y) in color_regions:
    # 设定一个（x-5,y-5）到（x+5,y+5）的10x10区域
    region = image_rgb[y - 5:y + 5, x - 5:x + 5]
    # np.mean()：这是 NumPy 库中的一个函数，用于计算给定数组的平均值。它会计算输入数组中元素的平均值。
    # region：这是一个二维区域（例如，10x10 的像素区域），它代表了图像中某个方块的像素。
    # axis=(0, 1)代表指按整个区域的每个通道（R、G、B）去求平均值
    # astype(int)将本次计算的值转换成int类型（计算完的平均值是float类型）
    avg_color = np.mean(region, axis=(0, 1)).astype(int)

    # 将计算出来的平均值放到三元组（R,G,B）中，map将np.int类型转换成int类型
    colors.append(tuple(map(int, avg_color)))

    # 在原始图像上绘制矩形标记区域。分别计算左上角、右下角
    top_left = (x - 5, y - 5)
    bottom_right = (x + 5, y + 5)
    # 画矩形，绿色边框，线条宽度为2
    cv2.rectangle(image_resized, top_left, bottom_right, (0, 255, 0), 2)  # 绿色边框，厚度2

# 显示带有标记的图像
cv2.imshow('Marked Regions', image_resized)
# 按任意键关闭图像窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 输出每个小方块的颜色
for i, color in enumerate(colors):
    print(f"色块 {i + 1}: RGB = {color}")

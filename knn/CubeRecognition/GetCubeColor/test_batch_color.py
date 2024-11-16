import cv2
import numpy as np


def create_color_grid(rgb_list, block_size=100):
    """
    根据输入的 RGB 值生成一个颜色网格。
    :param rgb_list: 一个 m*n 的 RGB 列表，表示每个色块的颜色
    :param block_size: 每个色块的大小，默认为 100*100
    :return: 一个包含颜色网格的图像
    """

    max_width = max(len(row) for row in rgb_list)  # 最宽的列数
    height = len(rgb_list) * block_size
    width = max_width * block_size

    # 创建空白图像
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(len(rgb_list)):
        for j in range(len(rgb_list[i])):
            # 获取当前色块对应的 RGB 值
            rgb_values = rgb_list[i][j]
            # 计算当前色块在图像中的位置
            start_x = i * block_size
            start_y = j * block_size
            end_x = start_x + block_size
            end_y = start_y + block_size
            # print(f"({start_x}, {start_y}), ({end_x},{end_y})")
            # 设置色块颜色，注意 OpenCV 使用 BGR 格式
            color_image[start_x:end_x, start_y:end_y] = rgb_values[::-1]

    return color_image


# 示例输入 RGB 列表
source = [
    [[255, 0, 0], [255, 0, 10], [235, 0, 0], [250, 30, 30], [250, 40, 40]],
    [[0, 255, 0], [0, 230, 10], [10, 235, 10], [50, 230, 50], [40, 210, 40]],
    [[0, 0, 255], [20, 15, 200], [0, 90, 180], [50, 50, 250], [0, 70, 160]],
    [[255, 255, 0], [240, 230, 10], [240, 250, 0], [235, 235, 10], [220, 250, 30]],
    [[255, 165, 0], [240, 155, 0], [250, 165, 20], [240, 165, 5], [235, 175, 0]],
    [[255, 255, 255], [240, 255, 240], [240, 240, 240], [250, 250, 250], [235, 235, 235]]
]

# 调用函数生成颜色网格
color_grid = create_color_grid(source, block_size=100)

# 显示颜色网格
cv2.imshow('Color Grid', color_grid)
# 按任意键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

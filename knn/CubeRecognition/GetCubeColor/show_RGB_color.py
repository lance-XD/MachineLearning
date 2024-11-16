import cv2
import numpy as np


def show_color(rgb):
    """
    根据输入的 RGB 值显示对应的颜色。
    :param rgb: 一个包含 R、G、B 的三元组 (R, G, B)
    """
    # 创建一张纯色图片
    color_image = np.zeros((300, 300, 3), dtype=np.uint8)
    # OpenCV 使用 BGR 格式, 将传入的RGB值转换为BGR数据
    color_image[:] = rgb[::-1]

    # 显示颜色
    cv2.imshow('Color', color_image)
    # 按任意按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 输入 RGB 值
rgb_input = input("请输入 RGB 值（格式：R,G,B）：")
rgb_values = tuple(map(int, rgb_input.split(',')))

# 检查 RGB 值范围
if all(0 <= value <= 255 for value in rgb_values):
    show_color(rgb_values)
else:
    print("输入的 RGB 值无效，请确保每个值在 0 到 255 之间。")

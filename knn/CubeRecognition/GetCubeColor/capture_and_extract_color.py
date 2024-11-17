import cv2
import numpy as np

# 打开摄像头,通过参数切换使用的摄像设备，0、1、2...
cap = cv2.VideoCapture(0)

print("按下 'c' 拍照并识别颜色，按 'q' 退出。（输入法要切换到英文）")

while True:
    # 读取摄像头的一帧画面
    ret, frame = cap.read()

    # 检查是否成功捕获
    if not ret:
        print("无法捕获视频帧")
        break

    # 显示实时视频流
    cv2.imshow('Camera', frame)

    # 检测按键
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # 按下 'c' 键拍照
        image = frame.copy()
        break
    elif key == ord('q'):
        # 按下 'q' 键退出
        cap.release()
        cv2.destroyAllWindows()
        exit()

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()

# 缩放图片以便后续处理,本行会将图片缩放为300x300像素的图像
image_resized = cv2.resize(image, (300, 300))

# 转换为 RGB 色彩空间
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# 定义感兴趣区域的坐标（假设魔方是3x3的九宫格）
color_regions = [(50, 50), (150, 50), (250, 50),
                 (50, 150), (150, 150), (250, 150),
                 (50, 250), (150, 250), (250, 250)]

# 提取每个小方块的平均RGB值并在图像上标记
colors = []
for (x, y) in color_regions:
    # 定义一个10x10的区域
    region = image_rgb[y - 5:y + 5, x - 5:x + 5]

    # 计算区域的平均颜色
    avg_color = np.mean(region, axis=(0, 1)).astype(int)
    colors.append(tuple(map(int, avg_color)))

    # 在图像上绘制矩形框标记区域
    top_left = (x - 5, y - 5)
    bottom_right = (x + 5, y + 5)
    cv2.rectangle(image_resized, top_left, bottom_right, (0, 255, 0), 2)  # 绿色边框，厚度为2

# 显示带标记的图像，便于验证感兴趣区域
cv2.imshow('Captured Image with Regions', image_resized)

print("按下 's' 保存此照片到程序文件夹，按其他按键关闭。（输入法要切换到英文）")

# 检测按键,参数代表等待的时间,单位为ms,此处为0ms则会无限等待，直到用户按下任意按键
key = cv2.waitKey(0) & 0xFF
# 按's'键保存单张图片
if key == ord('s'):
    filename = 'captured_image.png'
    cv2.imwrite(filename, image_resized)
    print(f"图片已保存为 {filename}，图片位于本文件同一文件夹下")
    cv2.destroyAllWindows()
else:
    # 按下 'q' 键退出
    cv2.destroyAllWindows()

# 输出每个区域的平均颜色值
for i, color in enumerate(colors):
    print(f"色块 {i + 1}: RGB = {color}")

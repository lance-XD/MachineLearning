import cv2
import dlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 设置中文字体,解决matplotlib无法正常显示中文的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def detect_faces_dlib(image_path, output_path=None):
    """
    使用dlib的人脸检测器检测并框出人脸
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # 转换为RGB（dlib需要RGB格式）
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 初始化dlib的人脸检测器
    detector = dlib.get_frontal_face_detector()

    # 检测人脸
    faces = detector(rgb_image, 1)  # 1表示上采样一次，可以检测更多人脸

    print(f"检测到 {len(faces)} 张人脸")

    # 在图像上绘制人脸框
    result_image = image.copy()
    for i, face in enumerate(faces):
        # 获取人脸坐标
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        # 绘制矩形框
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 添加标签
        cv2.putText(result_image, f'Face {i + 1}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print(f"人脸 {i + 1}: 位置({x}, {y}), 大小({w}x{h})")

    # 转换为RGB用于显示
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # 显示结果
    plt.figure(figsize=(12, 8))
    plt.imshow(result_rgb)
    plt.axis('off')
    plt.title(f'人脸检测结果 (dlib) - 检测到 {len(faces)} 张人脸', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 保存结果
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"结果已保存到: {output_path}")

    return result_image, faces


# 使用示例
if __name__ == "__main__":
    image_path = "testpictures/2.png"  # 替换为您的图片路径
    output_path = "output/faces_detected_dlib.jpg"

    detect_faces_dlib(image_path, output_path)
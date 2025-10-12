import cv2
import dlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import matplotlib
matplotlib.use('TkAgg')

class FaceLandmarkDetector:
    def __init__(self, predictor_path='shape_predictor_68_face_landmarks.dat'):
        """
        初始化人脸特征点检测器

        参数:
            predictor_path: dlib形状预测器模型文件路径
        """
        # 下载预训练模型（68点，dlib官方提供）
        # 可以从这里下载: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

        # 初始化dlib的人脸检测器和特征点检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_landmarks(self, image_path):
        """
        检测人脸特征点

        参数:
            image_path: 输入图像路径

        返回:
            image: 原始图像
            faces: 检测到的人脸列表
            all_landmarks: 所有特征点列表
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转换为RGB格式
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 检测人脸
        faces = self.detector(rgb_image)

        all_landmarks = []
        for face in faces:
            # 检测特征点
            landmarks = self.predictor(rgb_image, face)
            landmarks_points = []

            # 提取68个特征点坐标
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))

            all_landmarks.append(landmarks_points)

        return image, faces, all_landmarks

    def draw_landmarks_matplotlib(self, image_path, save_path=None):
        """
        使用matplotlib绘制特征点

        参数:
            image_path: 输入图像路径
            save_path: 保存结果图像的路径（可选）
        """
        image, faces, all_landmarks = self.detect_landmarks(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(rgb_image)
        ax.axis('off')

        # 为每个特征点区域定义颜色
        colors = {
            'jaw': 'red',  # 0-16: 下巴
            'right_eyebrow': 'blue',  # 17-21: 右眉毛
            'left_eyebrow': 'blue',  # 22-26: 左眉毛
            'nose': 'green',  # 27-35: 鼻子
            'right_eye': 'cyan',  # 36-41: 右眼
            'left_eye': 'cyan',  # 42-47: 左眼
            'mouth': 'purple'  # 48-67: 嘴巴
        }

        # 绘制特征点
        for landmarks in all_landmarks:
            # 绘制连接线
            self._draw_connections(ax, landmarks, colors)

            # 绘制点
            for i, (x, y) in enumerate(landmarks):
                color = self._get_point_color(i, colors)
                circle = Circle((x, y), radius=2, color=color, fill=True, alpha=0.8)
                ax.add_patch(circle)
                # 可选：显示点编号
                # ax.text(x, y, str(i), fontsize=8, color='white', ha='center', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"结果已保存到: {save_path}")

        plt.show()

    def _get_point_color(self, point_index, colors):
        """根据特征点索引返回对应的颜色"""
        if 0 <= point_index <= 16:
            return colors['jaw']
        elif 17 <= point_index <= 21:
            return colors['right_eyebrow']
        elif 22 <= point_index <= 26:
            return colors['left_eyebrow']
        elif 27 <= point_index <= 35:
            return colors['nose']
        elif 36 <= point_index <= 41:
            return colors['right_eye']
        elif 42 <= point_index <= 47:
            return colors['left_eye']
        else:  # 48-67
            return colors['mouth']

    def _draw_connections(self, ax, landmarks, colors):
        """绘制特征点之间的连接线"""
        # 定义连接关系
        connections = [
            # 下巴
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
            # 右眉毛
            (17, 18), (18, 19), (19, 20), (20, 21),
            # 左眉毛
            (22, 23), (23, 24), (24, 25), (25, 26),
            # 鼻子
            (27, 28), (28, 29), (29, 30), (30, 33), (31, 32), (32, 33), (33, 34),
            # 右眼
            (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
            # 左眼
            (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
            # 嘴巴外轮廓
            (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
            (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
            # 嘴巴内轮廓
            (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)
        ]

        for start, end in connections:
            x_start, y_start = landmarks[start]
            x_end, y_end = landmarks[end]
            color = self._get_point_color(start, colors)
            ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=1.5, alpha=0.7)


def main():
    # 初始化检测器
    detector = FaceLandmarkDetector('shape_predictor_68_face_landmarks.dat')

    # 检测并绘制特征点
    image_path = 'testpictures/1.jpg'  # 替换为你的图片路径
    output_path = 'output/output_landmarks.jpg'  # 输出图片路径

    try:
        detector.draw_landmarks_matplotlib(image_path, save_path=output_path)
    except Exception as e:
        print(f"错误: {e}")
        print("请确保:")
        print("1. 图像路径正确")
        print("2. 已下载dlib形状预测器模型")
        print("3. 模型文件路径正确")


if __name__ == "__main__":
    main()
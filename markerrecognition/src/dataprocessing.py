"""
标志识别数据整合脚本

该脚本用于整合data文件夹下的8种标志数据：
1. 限高
2. 禁行
3. 注意危险
4. 坡道
5. 禁止通行
6. 环岛
7. 停车
8. 限速

功能：
- 读取每个标志下的所有批次子文件夹中的图片
- 将所有图片合并并重命名为1.jpg, 2.jpg...等连续编号
- 输出到output文件夹中，保持8个标志文件夹结构
"""

import os
import random
import shutil


def get_marker_folders(data_path):
    """
    获取所有标志文件夹列表
    
    Args:
        data_path: data文件夹路径
        
    Returns:
        标志文件夹名称列表
    """
    marker_folders = []
    for item in sorted(os.listdir(data_path)):
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path):
            marker_folders.append(item)
    return marker_folders


def get_all_images(marker_path):
    """
    获取指定标志文件夹下所有子文件夹中的图片文件
    
    Args:
        marker_path: 标志文件夹路径
        
    Returns:
        图片文件路径列表
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []

    # 遍历所有子目录和文件
    # os.walk 返回一个生成器，每次迭代产生一个三元组 (dirpath, dirnames, filenames)，从高层级到低层级依次返回
    # 三元组，最外层时的files是目录，内层目录为root时，则dirs为空，文件名为files
    for root, dirs, files in os.walk(marker_path):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            # 扩展名
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in image_extensions:
                image_files.append(file_path)

    # 按文件路径排序，确保顺序一致
    return sorted(image_files)


def get_summary_filename(marker_name) -> str:
    """
    获取到原本的标记名对应的名字
    :param marker_name: 标志的文件夹名
    :return: 处理后的名字
    """
    # 对应：火源、打架、垃圾堆、限速、掉头、右转、停车、其他
    NAME_COLLECTION = ["fire", "fight", "rubbish", "ratelimit", "turn_round", "right_turn", "park", "other"]

    new_name = ""
    if marker_name == "1.火源":
        new_name = NAME_COLLECTION[0]
    elif marker_name == "2.打架":
        new_name = NAME_COLLECTION[1]
    elif marker_name == "3.垃圾堆":
        new_name = NAME_COLLECTION[2]
    elif marker_name == "4.限速":
        new_name = NAME_COLLECTION[3]
    elif marker_name == "5.掉头":
        new_name = NAME_COLLECTION[4]
    elif marker_name == "6.右转":
        new_name = NAME_COLLECTION[5]
    elif marker_name == "7.停车":
        new_name = NAME_COLLECTION[6]
    else:
        new_name = NAME_COLLECTION[-1]
    return new_name


def process_marker_data(marker_name, data_path, output_path):
    """
    处理单个标志的数据：合并所有批次并重命名
    
    Args:
        marker_name: 标志文件夹名称
        data_path: data文件夹路径
        output_path: 输出文件夹路径
        
    Returns:
        处理的图片数量
    """
    marker_input_path = os.path.join(data_path, marker_name)
    # 将数据划分为训练集和测试集
    marker_train_path = os.path.join(output_path, "train", marker_name)
    marker_test_path = os.path.join(output_path, "test", marker_name)

    # 创建输出对应的训练、测试文件夹, 已存在则跳过创建
    os.makedirs(marker_train_path, exist_ok=True)
    os.makedirs(marker_test_path, exist_ok=True)

    # 检查是否存在汇总的文件夹，不在则创建
    marker_train_summary_path = os.path.join(output_path, "train", "summary")
    marker_test_summary_path = os.path.join(output_path, "test", "summary")
    if not os.path.exists(marker_train_summary_path):
        os.makedirs(marker_train_summary_path, exist_ok=True)
    if not os.path.exists(marker_test_summary_path):
        os.makedirs(marker_test_summary_path, exist_ok=True)

    # 获取所有图片
    image_files = get_all_images(marker_input_path)

    print(f"  处理标志: {marker_name}")
    print(f"    找到 {len(image_files)} 张图片")

    # 随机打乱数据
    random.shuffle(image_files)

    # 划分比例，80%训练， 20%测试
    DIVISION_RATIO = 0.8
    # 图片总数
    total_images = len(image_files)

    # 复制并重命名图片,start参数指定idx的值从1开始
    for idx, image_path in enumerate(image_files, start=1):
        # 获取原始文件扩展名
        _, ext = os.path.splitext(image_path)

        # 划分入训练集、测试集的新文件名: 1.jpg, 2.jpg, ...
        new_filename = f"{idx}{ext.lower()}"
        # 汇总文件夹的文件名
        summary_filename = f"{get_summary_filename(marker_name)}_{idx}{ext.lower()}"

        # 前80%放入训练集
        if idx <= total_images * DIVISION_RATIO:
            output_path = marker_train_path
            summary_output_path = marker_train_summary_path
        else:
            output_path = marker_test_path
            summary_output_path = marker_test_summary_path

        output_file_path = os.path.join(output_path, new_filename)
        summary_file_path = os.path.join(summary_output_path, summary_filename)
        # 复制文件到汇总的文件夹
        shutil.copy2(image_path, output_file_path)
        shutil.copy2(image_path, summary_file_path)

    print(f"    成功复制 {total_images * DIVISION_RATIO} 张图片到 {marker_train_path}")
    print(f"    成功复制 {total_images * (1 - DIVISION_RATIO)} 张图片到 {marker_test_path}")
    return len(image_files)


def main():
    """
    主函数：整合所有标志数据
    """
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置路径,路径上类似于D:\Pycharm\Projects\MachineLearning\markerrecognition\src\..\data
    data_path = os.path.join(script_dir, "..", "data")
    output_path = os.path.join(script_dir, "..", "output")

    # 转换为绝对路径
    data_path = os.path.abspath(data_path)
    output_path = os.path.abspath(output_path)

    print("=" * 60)
    print("标志识别数据整合工具")
    print("=" * 60)
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_path}")
    print()

    # 检查data文件夹是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据路径不存在: {data_path}")
        return

    # 创建输出文件夹
    os.makedirs(output_path, exist_ok=True)

    # 获取所有标志文件夹
    marker_folders = get_marker_folders(data_path)

    if not marker_folders:
        print("错误: 未找到任何标志文件夹")
        return

    print(f"发现 {len(marker_folders)} 个标志文件夹:")
    for folder in marker_folders:
        print(f"  - {folder}")
    print()

    # 处理每个标志
    total_images = 0
    for marker_name in marker_folders:
        count = process_marker_data(marker_name, data_path, output_path)
        total_images += count

    print()
    print("=" * 60)
    print("数据整合完成!")
    print(f"总计处理图片: {total_images} 张")
    print(f"输出目录: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

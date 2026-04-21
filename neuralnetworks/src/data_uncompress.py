import gzip
import pickle
import os
import numpy as np
from PIL import Image


def uncompress_mnist(data_path='../data/mnist.pkl.gz', save_dir='../pictures'):
    """
    将 mnist.pkl.gz 中的所有图片解压并保存到指定目录，不划分数据集
    """
    os.makedirs(save_dir, exist_ok=True)

    with gzip.open(data_path, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding
        ='latin1')

    all_data = [
        train_set,
        valid_set,
        test_set
    ]

    total = 0
    for dataset in all_data:
        images, labels = dataset
        for img_vec, label in zip(images, labels):
            img_array = img_vec.reshape(28, 28)
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')

            path = os.path.join(save_dir, f"mnist_{total}_label_{label}.png")
            img.save(path)

            total += 1
            if total % 5000 == 0:
                print(f"已保存 {total} 张图片...")

    print(f"全部完成，共保存 {total} 张图片到 {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    uncompress_mnist()

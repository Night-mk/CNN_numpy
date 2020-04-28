'''
MNIST.py 本文件用于加载MNIST数据集，包括图像数据、标签数据

MNSIT数据集（http://yann.lecun.com/exdb/mnist/）：
Training set images: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)
Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)
Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)
Test set labels: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)

'''

import numpy as np

'''
    Loader类，基础类，用于加载数据
'''
class Loader(object):
    # 初始化函数，用于初始化加载类
    def __init__(self, path, count):
        # path: 文件加载路径 count: 文件数量
        self.path = path
        self.count = count

    # 使用open函数，读取文件内容，将文件内容以二进制模式读取出来
    def get_content(self):
        # 'r'读模式、'w'写模式、'a'追加模式、'b'二进制模式、'+'读/写模式。
        # 'rb'读二进制模式
        print("load_path: ",self.path)
        f = open(self.path, 'rb')
        # 读取二进制字节流
        content = f.read()
        f.close()
        return content


'''
    ImageLoader类，用于加载MNIST数据集图片
    继承自Loader
    图片数据28x28
'''
class ImageLoader(Loader):
    # 加载单张图片
    def get_picture(self, content, index):
        # 获取图片数据的索引，数据集从第17Byte开始才存储图片，从content中获取数据
        # 加载图片为二维数组[[x,x,x..][x,x,x...][x,x,x...][x,x,x...]]的列表
        start = index*28*28+16
        picture_data = np.zeros([28,28])
        for i in range(28):
            for j in range(28):
                byte = content[start+ i*28+j]
                picture_data[i][j] = byte
        return picture_data

    # 将数据转换为784byte向量
    def get_one_sample(self, picture):
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    # 加载图片
    # 加载方式1：直接加载28x28; 加载方式2：将每张图片转化为784byte的行向量
    # 从压缩文件里循环读取数据
    def load(self, onerow=False):
        # 获取图片的二进制流
        content = self.get_content()
        # task1：读取图片，转换成28x28
        data_set = []
        for index in range(self.count):
            onepic = self.get_picture(content, index)
            # task2：读取图片，转换成784
            if onerow: onepic = self.get_one_sample(onepic)
            # 获取数据集
            data_set.append(onepic)
        # data_set=[array[28][28]]
        return data_set

'''
    LabelLoader类，用于加载MNIST数据集标签
    标签数据，二进制[0~9]
'''
class LabelLoader(Loader):

    # 加载标签[0-9]
    def load(self):
        content = self.get_content()
        labels = []
        for index in range(self.count):
            onelabel = content[index+8]
            # 将label转换为one-hot编码（二进制）
            onelabel_vector = self.norm(onelabel)
            labels.append(onelabel_vector)
        # label=[array[10]]
        return labels

    # 内部函数，one-hot编码。将一个值转换为10维标签向量
    def norm(self, label):
        label_vec = []
        # label_value = self.to_int(label)
        label_value = label  # python3中直接就是int
        for i in range(10):
            if i == label_value:
                label_vec.append(1)
            else:
                label_vec.append(0)
        return label_vec


def get_training_data_set(num,onerow=False):
    image_loader = ImageLoader('./MNIST_data/train-images.idx3-ubyte', num)  # 参数为文件路径和加载的样本数量
    label_loader = LabelLoader('./MNIST_data/train-labels.idx1-ubyte', num)  # 参数为文件路径和加载的样本数量
    data_img = image_loader.load(onerow)
    data_label = label_loader.load()
    return data_img, data_label

# 获得测试数据集。onerow表示是否将每张图片转化为行向量
def get_test_data_set(num,onerow=False):
    image_loader = ImageLoader('./MNIST_data/t10k-images.idx3-ubyte', num)  # 参数为文件路径和加载的样本数量
    label_loader = LabelLoader('./MNIST_data/t10k-labels.idx1-ubyte', num)  # 参数为文件路径和加载的样本数量
    data_img = image_loader.load(onerow)
    data_label = label_loader.load()
    return data_img, data_label


if __name__=="__main__":
    train_dataset, train_labels = get_training_data_set(1)
    print("data_set: ", train_dataset)
    print("train_labels: ", train_labels)
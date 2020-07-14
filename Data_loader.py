'''
    Data_loader.py 用于实现自定义的数据集加载，继承使用pytorch的dataset库
'''

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
import pickle

import torchvision.utils as vutils # 暂时不知道干嘛的（处理图像用的？）
import matplotlib.pyplot as plt

'''
    定义cifar-1O数据集类
    cifar-10数据集：
    batches.meta.bet: 文件存储了每个类别的英文名称。
    data_batch_1.bin: 这5 个文件是cifar-10 数据集中的训练数据。每个文件以二进制格式存储了10000 张3x 32 × 32的彩色图像和这些图像对应的类别标签。一共50000 张训练图像。
    data_batch_2.bin
    data_batch_3.bin
    data_batch_4.bin
    data_batch_5.bin
    test_batch.bin: 文件存储的是测试图像和测试图像的标签。一共10000张
'''

'''加载二进制文件数据'''
def cifar_load_data(data_dir):
    with open(data_dir, 'rb') as f:
        # pickle模块是python中用来持久化对象的一个模块。所谓对对象进行持久化，即将对象的数据类型、存储结构、存储内容等所有信息作为文件保存下来以便下次使用。
        data = pickle.load(f, encoding='bytes')
        labels = data[b'labels']
        samples = data[b'data']
    return samples, labels

class cifar_Resize(object):
    def __init__(self, out_shape):
        self.out_shape = out_shape

    def __call__(self, sample):
        sample_new = np.reshape(sample, self.out_shape)
        return sample_new
    
'''继承Dataset类，构建自定义的cifar-10数据集'''
class CifarDataset(Dataset):
    # __init__方法：对数据集进行整理，得到图像的路径，给图片打标签，划分数据集
    def __init__(self, root_dir, transform=None, train=True):
        # 初始化文件路径或文件名列表。
        # 初始化该类的一些基本参数。
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        # 获取数据集文件的目录
        self.samples_dir = []
        if self.train: # 训练集
            for i in range(5):
                # 获取文件目录
                train_file_dir = self.root_dir+'/data_batch_'+str(i+1)
                self.samples_dir.append(train_file_dir)
                # 获取训练集数据
                # 对于数据量小的cifar-10，可以直接读取，将整个文件加载到内存中
                train_samples, train_labels = cifar_load_data(self.samples_dir[i])
                if i==0: 
                    self.samples = train_samples
                    self.labels = train_labels
                else:
                    # np.concatenate((a,b),axis=0) 记得要把a,b括起来
                    self.samples = np.concatenate((self.samples, train_samples), axis=0) # 垂直方向组合
                    self.labels = np.concatenate((self.labels, train_labels), axis=0)
        else: # 测试集
            train_file_dir = self.root_dir+'/test_batch'
            self.samples_dir.append(train_file_dir)
            # 获取测试集数据
            test_samples, test_labels = cifar_load_data(self.samples_dir[0])
            self.samples = test_samples # 垂直方向组合
            self.labels = test_labels

    # 重写方法 __len__ 返回数据集的大小
    def __len__(self):
        return len(self.samples)

    # 重写方法 __getitem__ 实现可以通过索引来返回图像数据,返回的数据是torch类型
    def __getitem__(self, index):
        # 1。从文件中读取一个数据（例如，plt.imread）。
        # 2。预处理数据（例如torchvision.Transform）。
        # 3。返回数据对（例如图像和标签）。
        sample, label = self.samples[index], int(self.labels[index])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label


if __name__ == '__main__':
    # transform库仅对PIL.Image类型数据有效
    cifar_transform = transforms.Compose([
        cifar_Resize((3,32,32))
    ])
    root_dir = './data/cifar-10/cifar-10-python/cifar-10-batches-py'
    cifar_train_data = CifarDataset(root_dir, transform=cifar_transform, train=True)
    dataloader = DataLoader(cifar_train_data,batch_size=64,shuffle=False)#使用DataLoader加载数据

    num = 0
    '''迭代数据集'''
    for t, (data, target) in enumerate(dataloader, 0):
        for i, target_i in enumerate(target): # 筛选dog类
            if target_i==5:
                num+=1
        print('num_accumulate: ', num)
    print('num_sum: ', num)
        
    # for i, (sample, label) in enumerate(dataloader):
    #     if i<3:
    #         print('sample: ',sample[i].size())
    #         print('label: ',label[i])
    #         # Tensor转成PIL.Image，提供可视化功能(显示单张图片)
    #         img = transforms.ToPILImage()(sample[i]).convert('RGB')
    #         img.show()
    #     else:
    #         break

    '''可视化处理数据集'''
    real_batch = next(iter(dataloader))
    print(real_batch[0].shape)
    # for sample in real_batch[0]:
    #     sample = transforms.ToPILImage()(sample).convert('RGB')
    # 绘制图像
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("train images")
    # 虽然看不懂但是可以直接用就很虚浮，此处的normalize参数判断原来数据集是否标准化过，如果为True则还原
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=False).cpu(),(1,2,0)))
    # 显示该图像（加载成功的）
    plt.show()
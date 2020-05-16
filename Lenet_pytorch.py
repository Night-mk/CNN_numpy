'''
    使用pytorch实现Lenet-5网络
    主要流程：
    1、MNIST数据集处理
    2、CNN网络构建
    3、训练
'''

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets,transforms
import os

class Lenet(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Lenet, self).__init__()
        # 定义层
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,n_class)
        # 计算softmax的log值(可以减少e^x的操作，将乘法改成加法减少计算量)
        # 改版pytorch需要添加dim=1参数（不知道为啥）
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播
    def forward(self, x):
        in_size = x.size(0)
        out_c1s2 = self.relu(self.maxpool(self.conv1(x)))
        out_c3s4 = self.relu(self.maxpool(self.conv2(out_c1s2)))
        out_c5 = self.relu(self.conv3(out_c3s4))
        # reshape 卷积的多维feature map为一个2维矩阵
        out_c5 = out_c5.view(in_size, -1)
        out_f6 = self.relu(self.fc1(out_c5))
        out_f7 = self.fc2(out_f6)
        out_logsoftmax = self.logsoftmax(out_f7)
        return out_logsoftmax


def test_lenet():
    # path1=os.path.abspath('./data/')
    # print('path: ',path1)
    """处理MNIST数据集"""
    train_dataset = datasets.MNIST('./data/',download=True,train=True,transform=transforms.Compose([
                                   transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                               ]))
    test_dataset = datasets.MNIST('./data/',download=True,train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                              ]))
    print('traindata_len: \n',len(train_dataset))
    # 构建数据集迭代器
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=True)

    # 使用GPU加速
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_gpu = torch.cuda.is_available()

    """初始化使用网络"""
    lenet = Lenet(in_dim=1, n_class=10)
    print('model: \n', lenet)
    print('parameters: \n', lenet.parameters())
    print('modules: \n', lenet._modules)
    # if use_gpu: lenet = Lenet(in_dim=1, n_class=10).to(device)

    """构建优化器"""
    # 使用交叉熵计算损失？
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(lenet.parameters(),lr=1e-2,momentum=0.5)

    print('optim_param: \n', optimizer.param_groups)
    # 打印参数
    # for param in lenet.parameters():
    #     print(param)
        # print('isparam: ',isinstance(param, nn.Parameter))

    """迭代训练"""
    n_epochs = 1
    # 迭代轮数
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-"*10)
        # 使用迭代器训练
        for t, (data, target) in enumerate(train_loader):
            break
            # 从Variable获取数据和标签
            data,target = Variable(data),Variable(target)
            # 前向传播计算
            pred = lenet(data)
            # torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引
            _,output = torch.max(pred, 1)
            # print('target: \n',target) # 数字表示类别
            # print('pred: \n',pred) # one-hot编码表示类别
            # 计算损失
            loss = loss_fn(pred,target)
            
            # 反向传播和参数更新
            optimizer.zero_grad() # 清空上一次梯度
            loss.backward()	# 反向传播
            optimizer.step() # 优化器参数更新

            # total += target.size(0)
            running_loss += loss.item()
            # 记录输出的数据正确的个数
            running_correct += (output == target).sum().item()
            if t%50==0 and t!=0:
                print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%".format(running_loss/(t*64.0),100.0*running_correct/(t*64.0)))
            #     print(100*running_correct.data/total)
    
        testing_correct = 0
        for t, (data, target) in enumerate(test_loader):
            x_test, y_test = Variable(data),Variable(target)
            pred = lenet(x_test)
            _, output = torch.max(pred, 1)
            # testing_correct += torch.sum(output == y_test.data)
            testing_correct += (output == y_test.data).sum().item()

        print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}%".format(running_loss/len(train_dataset),100.0*running_correct/len(train_dataset),100.0*testing_correct/len(test_dataset)))

    # 训练结束后可以存储模型  
    torch.save(lenet.state_dict(), "./model_save/lenet_parameter.pkl")                                                  


if __name__ == "__main__":
    test_lenet()
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())



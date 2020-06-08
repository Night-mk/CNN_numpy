'''
    Lenet_numpy.py 
    使用numpy实现可训练、预测的CNN框架
'''
import random
import numpy as np
from Module import Module
from Optimizer_func import SGD
from Optimizer_func import Adam
from Parameter import Parameter
from Conv import ConvLayer
from Pool import MaxPooling
from Activators import ReLU
from FC import FullyConnect
from Logsoftmax import Logsoftmax
from Loss import NLLLoss

import torch
from torchvision import datasets,transforms
import os
import time

import matplotlib.pyplot as plt
import torchvision.utils as vutils # 暂时不知道干嘛的（处理图像用的？）


n_epochs = 1 # 训练轮数
n_epochs_pre = 0 # 预训练轮数（加载已经训练好的模型时可以更新）
batch_size = 64

class Lenet_numpy(Module):
    def __init__(self, in_dim, n_class):
        super(Lenet_numpy, self).__init__()
        self.conv1 = ConvLayer(in_dim, 6, 5,5, zero_padding=2, stride=1, method='SAME')
        self.conv2 = ConvLayer(6, 16, 5,5, zero_padding=0, stride=1, method='VALID')
        self.conv3 = ConvLayer(16, 120, 5,5, zero_padding=0, stride=1, method='VALID')

        self.maxpool1 = MaxPooling(pool_shape=(2,2), stride=(2,2))
        self.maxpool2 = MaxPooling(pool_shape=(2,2), stride=(2,2))
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.relu3 = ReLU()
        self.relu4 = ReLU()
        self.fc1 = FullyConnect(120, 84)
        self.fc2 = FullyConnect(84, n_class)
        self.logsoftmax = Logsoftmax()

    def forward(self, x): # 存在问题是：同一个对象其实是不能多次使用的，因为每个对象都有自己的input和output，如果重复使用反向会错误
        in_size = x.shape[0]
        out_c1s2 = self.relu1.forward(self.maxpool1.forward(self.conv1.forward(x)))
        out_c3s4 = self.relu2.forward(self.maxpool2.forward(self.conv2.forward(out_c1s2)))
        out_c5 = self.relu3.forward(self.conv3.forward(out_c3s4))
        self.conv_out_shape = out_c5.shape

        out_c5 = out_c5.reshape(in_size, -1)
        out_f6 = self.relu4.forward(self.fc1.forward(out_c5))
        out_f7 = self.fc2.forward(out_f6)
        out_logsoftmax = self.logsoftmax.forward(out_f7)

        return out_logsoftmax
    
    def backward(self, dy):
        dy_logsoftmax = self.logsoftmax.gradient(dy)
        dy_f7 = self.fc2.gradient(dy_logsoftmax)
        dy_f6 = self.fc1.gradient(self.relu4.gradient(dy_f7))

        dy_f6 = dy_f6.reshape(self.conv_out_shape)

        dy_c5 = self.conv3.gradient(self.relu3.gradient(dy_f6))
        dy_c3f4 = self.conv2.gradient(self.maxpool2.gradient(self.relu2.gradient(dy_c5)))
        self.conv1.gradient(self.maxpool1.gradient(self.relu1.gradient(dy_c3f4)))


if __name__ == "__main__":
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
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    '''
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # 测试数据集是否加载成功
    real_batch = next(iter(train_loader))
    # 绘制图像
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("train images")
    # 虽然看不懂但是可以直接用就很虚浮
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # 显示该图像（加载成功的）
    plt.show()
    '''

    """初始化CNN网络"""
    Lenet = Lenet_numpy(in_dim=1, n_class=10)
    print('Lenet_numpy: \n', Lenet)

    """构建优化器"""
    loss_fn = NLLLoss()
    # optimizer = SGD(Lenet.parameters(), learning_rate=1e-2, momentum=0.5)
    # optimizer = SGD(Lenet.parameters(), learning_rate=1e-2, momentum=0.9) # momentum改为0.9训练更快
    lr = 1e-3# Adam优化器的学习率
    beta1 = 0.5 # Adam优化器的参数（需要调整试试？）
    optimizer = Adam(Lenet.parameters(), learning_rate=lr, betas=(beta1, 0.999)) # 测试一下Adam优化器
    # for param in Lenet.parameters():
    #     print('parameters() id: ', id(param))
    # print(Lenet.state_dict())

    """加载预训练的模型"""
    '''
    pre_module_path = "./model_save/lenet_numpy_parameters-1.pkl"
    params = torch.load(pre_module_path)
    Lenet.load_state_dict(params['state_dict']) # 加载模型
    n_epochs_pre = params['epoch']
    '''
    
    """迭代训练"""
    for epoch in range(n_epochs):
        # break
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-"*10)
        
        start_time = time.time()
        for t, (data, target) in enumerate(train_loader):
            # 将tensor类型的data转为numpy
            data = data.detach().numpy()
            target = target.detach().numpy()
            # print('data shape: \n', data.shape)
            # print('target: \n', target)

            pred = Lenet.forward(data) # pred=[x1,x2,...,xn]

            output = np.argmax(pred, axis=1)
            # print('pred_result: \n', output)

            loss = loss_fn.cal_loss(pred, target)
            # print('loss_result: \n', loss)

            optimizer.zero_grad()
            dy_loss = loss_fn.gradient()
            # print('dy_loss: \n', dy_loss)
            Lenet.backward(dy_loss)
            optimizer.step()

            # 计算总损失
            running_loss += loss
            running_correct += sum(output == target) 
            
            if t%20==0 and t!=0:
                end_time = time.time()
                print("Step/Epoch:{}/{}, Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Calculate time: {:.4f}min".format(t, epoch,running_loss/(t*batch_size), 100.0*running_correct/(t*batch_size), (end_time-start_time)/60))

        testing_correct = 0
        for t, (data, target) in enumerate(test_loader):
            x_test = data.detach().numpy()
            y_test = target.detach().numpy()
            pred = Lenet.forward(x_test)
            output = np.argmax(pred, axis=1)
            testing_correct += sum(output == y_test) 

        print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}%".format(running_loss/len(train_dataset),100.0*running_correct/len(train_dataset),100.0*testing_correct/len(test_dataset)))
        
    
    '''存储模型'''
    checkpoint_path = "./model_save/lenet_numpy_parameters-"+str(n_epochs+n_epochs_pre)+".pkl"
    torch.save({'epoch':n_epochs+n_epochs_pre, 'state_dict':Lenet.state_dict()}, checkpoint_path)
    
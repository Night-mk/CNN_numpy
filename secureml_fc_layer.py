'''
    使用secureml提供的全FC和square的layer
'''
import random
import numpy as np
from Module import Module
from Optimizer_func import Adam
from Parameter import Parameter
from Conv import ConvLayer
from Pool import MaxPooling
from Activators import ReLU
import Activators
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

'''
    网络结构：
    1、FC: in_size=1x28x28, out_channel=128x1
    2、Square Activation
    3、FC: in_size=128x1, out_size=128x1
    4、Square Activation
    5、FC: in_size=128x1, out_size=10x1
'''

class Secureml_fivelayer(Module):
    def __init__(self, in_dim, n_class):
        super(Secureml_fivelayer, self).__init__()

        self.fc0 = FullyConnect(784,128)
        self.sq1 = Activators.Square()
        self.fc1 = FullyConnect(128, 128)
        self.sq2 = Activators.Square()
        self.fc2 = FullyConnect(128, n_class)
        self.logsoftmax = Logsoftmax()

    def forward(self, x):
        in_size = x.shape[0]
        x = x.reshape(in_size, -1)
        out_1 = self.sq1.forward(self.fc0.forward(x))
        out_2 = self.sq2.forward(self.fc1.forward(out_1))
        out_3 = self.fc2.forward(out_2)
        
        out_logsoftmax = self.logsoftmax.forward(out_3)

        return out_logsoftmax
    
    def backward(self, dy):
        dy_logsoftmax = self.logsoftmax.gradient(dy)
        dy_f3 = self.fc2.gradient(dy_logsoftmax)
        dy_f2 = self.fc1.gradient(self.sq2.gradient(dy_f3))
        self.fc0.gradient(self.sq1.gradient(dy_f2))

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

    """初始化CNN网络"""
    Secureml_5 = Secureml_fivelayer(in_dim=1, n_class=10)
    print('Secureml_5: \n', Secureml_5)
    # for i in Secureml_5.parameters():
    #     print(i)

    """构建优化器"""
    loss_fn = NLLLoss()
    lr = 1e-3# Adam优化器的学习率
    beta1 = 0.5 # Adam优化器的参数（需要调整试试？）
    optimizer = Adam(Secureml_5.parameters(), learning_rate=lr, betas=(beta1, 0.999)) # 测试一下Adam优化器

    """加载预训练的模型"""
    '''
    pre_module_path = "./model_save/Secureml_5/Secureml_5_parameters-1.pkl"
    params = torch.load(pre_module_path)
    Secureml_5.load_state_dict(params['state_dict']) # 加载模型
    n_epochs_pre = params['epoch']
    '''
    
    """迭代训练"""
    for epoch in range(n_epochs):
        break
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

            pred = Secureml_5.forward(data) # pred=[x1,x2,...,xn]

            output = np.argmax(pred, axis=1)
            # print('pred_result: \n', output)

            loss = loss_fn.cal_loss(pred, target)
            # print('loss_result: \n', loss)

            optimizer.zero_grad()
            dy_loss = loss_fn.gradient()
            # print('dy_loss: \n', dy_loss)
            Secureml_5.backward(dy_loss)
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
            pred = Secureml_5.forward(x_test)
            output = np.argmax(pred, axis=1)
            testing_correct += sum(output == y_test) 

        print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}%".format(running_loss/len(train_dataset),100.0*running_correct/len(train_dataset),100.0*testing_correct/len(test_dataset)))
        
    
    '''存储模型'''
    checkpoint_path = "./model_save/Secureml_5/Secureml_5_parameters-"+str(n_epochs+n_epochs_pre)+".pkl"
    torch.save({'epoch':n_epochs+n_epochs_pre, 'state_dict':Secureml_5.state_dict()}, checkpoint_path)
    
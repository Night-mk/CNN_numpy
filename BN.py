'''
BN.py用于实现Batch Normalization
'''

import numpy as np
import types
from Module import Module
from Parameter import Parameter

class BatchNorm(Module):
    def __init__(self, in_channels):
        super(BatchNorm, self).__init__()
        # input_shape = [batchsize, channel_num, h, w] 
        self.in_channels = in_channels

        # 初始化BN层需要的参数gamma、beta,参数的数量和均值、方差后的维度相同
        param_gamma = np.ones(self.in_channels)
        param_beta = np.zeros(self.in_channels)
        self.gamma = Parameter(param_gamma, requires_grad=True)
        self.beta = Parameter(param_beta, requires_grad=True)

        self.gamma_grad = self.gamma.grad
        self.beta_grad = self.beta.grad

        self.epsilon = 1e-5
        self.mean = np.zeros(self.in_channels)
        self.var = np.zeros(self.in_channels)

        # 初始化指数加权移动平均的mean和var，用于在测试环节计算样本均值和方差的无偏估计
        self.moving_mean = np.zeros(self.in_channels)
        self.moving_var = np.zeros(self.in_channels)
        self.moving_decay = 0.9

    def set_gamma(self, gamma):
        if isinstance(gamma, Parameter):
            self.gamma = gamma

    def set_beta(self, beta):
        if isinstance(beta, Parameter):
            self.beta = beta

    # 设置module打印格式
    def extra_repr(self):
        s = ('in_channels={in_channels}, eps={epsilon}, moving_decay={moving_decay}')
        return s.format(**self.__dict__)

    # BN前向传播，分两种模式：训练模式train，测试模式test
    # batchNormalizaion2d,均值和方差的均在channel_num维度求
    def forward(self, input_array, mode="train"):
        # input_data = [batch,channel_num,h,w]
        self.input_data = input_array
        self.input_shape = self.input_data.shape
        self.batchsize = self.input_shape[0]

        # 计算均值的数据总量的维度m
        self.m = self.batchsize
        if self.input_data.ndim == 4:
            self.m = self.batchsize*self.input_data.shape[2]*self.input_data.shape[3]
        # 计算均值mean (axis=1对列求平均值,axis=0对行求平均)
        # keepdims=True可以保证使用np计算均值或者方差的结果保留原始数据的维度大小，可以方便的用于和原输入进行运算
        self.mean = np.mean(self.input_data, axis=(0,2,3), keepdims=True)
        # print('mean.shape: ',self.mean.shape)
        # 记录一下标准差standard矩阵, 反向传播时使用
        self.standard = self.input_data-self.mean
        # 计算方差var
        self.var = np.var(self.input_data, axis=(0,2,3), keepdims=True)
        # 存在多组数据batch的情况下，需要计算方差的无偏估计（b/(b-1)*E(var(x))） [但是pytorch似乎也没这么计算]
        # if self.batchsize>1:
        #     self.var = self.m/(self.m-1)*self.var
        
        # 利用指数加权平均算法计算moving_mean和moving_var,用于测试时作为整体的mean,var的无偏估计
        if np.sum(self.moving_mean)==0 and np.sum(self.moving_var)==0:
            self.moving_mean = self.mean
            self.moving_var = self.var
        else:
            self.moving_mean = self.moving_decay * self.moving_mean + (1-self.moving_decay)*self.mean
            self.moving_var = self.moving_decay * self.moving_var + (1-self.moving_decay)*self.var

        # 计算标准化值normed_x = [batch, bn_shape]
        if mode=='train':
            self.normed_x = (self.input_data-self.mean)/np.sqrt(self.var+self.epsilon)
            # print(self.normed_x)
        if mode=='test':
            self.normed_x = (self.input_data-self.moving_mean)/np.sqrt(self.moving_var+self.epsilon)
        # 计算BN输出 output_y = [batch, -1]
        # 对每个输入都进行标准化，所以输出y的size和输入相同
        # print('gamma.shape: ',self.gamma.shape)
        # print('normed_x.shape: ',self.normed_x.shape)
        # print('type_gamma: ',self.gamma[0])
        # print('type_normed_x: ',type(self.normed_x))
        
        # 对每个channel做一次线性变换
        output_y = np.zeros(self.input_shape)
        for i in range(self.in_channels):
            output_y_i = self.gamma.data[i]*self.normed_x[:,i,:,:] + self.beta.data[i]
            output_y[:,i,:,:] = output_y_i
        # output_y = np.array(output_y)
        # output_y = self.gamma*self.normed_x + self.beta
        return output_y
    
    # 梯度计算函数
    def gradient(self, eta):
        # eta = [batch, channel_num, height, width]
        # 无论上层误差如何，首先将上层传输的误差转化为[batch, -1]
        self.eta = eta
        self.gamma_grad = np.sum(self.eta*self.normed_x, axis=(0,2,3), keepdims=True)
        self.beta_grad = np.sum(self.eta, axis=(0,2,3), keepdims=True)
        # 设置Parameter的grad值
        self.gamma.set_grad(self.gamma_grad)
        self.beta.set_grad(self.beta_grad)
        # 计算向前一层传播的误差参数
        # normed_x_grad
        # normed_x_grad = self.eta*self.gamma 
        # 由于eta[B,C,H,W] gamma=[C],直接乘维度不对，需要针对C这个维度进行乘法，最后还原输出的size[B,C,H,W]
        normed_x_grad = np.zeros(self.eta.shape)
        for i in range(self.in_channels):
            normed_x_grad_i = self.gamma.data[i]*self.eta[:,i,:,:]
            normed_x_grad[:,i,:,:] = normed_x_grad_i
        # print(self.eta)
        # var_grad
        var_grad = -1.0/2*np.sum(normed_x_grad*self.standard, axis=(0,2,3), keepdims=True)/(self.var+self.epsilon)**(3.0/2)
        # mean_grad
        mean_grad = -1*np.sum(normed_x_grad/np.sqrt(self.var+self.epsilon), axis=(0,2,3), keepdims=True) + var_grad*np.sum(-2*self.standard,axis=(0,2,3), keepdims=True)/self.m
        # input_grad
        input_grad = normed_x_grad/np.sqrt(self.var+self.epsilon) + var_grad*2*self.standard/self.m + mean_grad/self.m

        self.eta_next = input_grad
        return input_grad

    '''
    def backward(self):
        # 更新gamma和beta
        self.gamma -= self.learning_rate*self.gamma_grad
        self.beta -= self.learning_rate*self.beta_grad
    '''

def basic_test():
    a = np.arange(24).reshape((2,3,2,2))
    mean = np.mean(a, axis=(0,2,3))
    print(a)
    print(mean)

def bn_test():
    # shape = np.array([12,3])
    # x = np.arange(36).reshape(shape)

    a = np.arange(48).reshape((4,3,2,2))
    bn1 = BatchNorm(3)
    bn_out = bn1.forward(a, 'train')
    # print(a)
    # print(bn1.input_data)
    # print(bn1.mean)
    # print(bn1.var)
    # print(bn1.normed_x)
    print('bn_out: ',bn_out)
    print('bn_out: ',bn_out.shape)

    dy=np.array([[[[1.3028, 0.5017],
       [-0.8432, -0.2807]],

      [[-0.4656, 0.2773],
       [-0.7269, 0.1338]]],

     [[[-3.1020, -0.7206],
       [0.4891, 0.2446]],

      [[0.2814, 2.2664],
       [0.8446, -1.1267]]],

     [[[-2.4999, 1.0087],
       [0.6242, 0.4253]],

      [[2.5916, 0.0530],
       [0.5305, -2.0655]]]])

    # x_grad = bn1.gradient(dy)
    # print(x_grad[0,0])
    

if __name__ == '__main__':
    # basic_test()
    bn_test()
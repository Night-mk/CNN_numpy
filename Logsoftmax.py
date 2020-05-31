'''
Logsoftmax.py用于实现Log版本的softmax
'''
import numpy as np
from Module import Module

class Logsoftmax(Module):
    def __init__(self):
        super(Logsoftmax, self).__init__()
        # input_shape=[batch, class_num]

    # 设置module打印格式
    def extra_repr(self):
        s = ()
        return s

    def cal_softmax(self, input_array):
        softmax = np.zeros(input_array.shape)
        # 对每个batch的数据求softmax
        exps_i = np.exp(input_array-np.max(input_array))
        softmax = exps_i/np.sum(exps_i)
        # softmax[batch, class_num]
        return softmax

    def forward(self, input_array):
        self.input_shape = input_array.shape
        self.batchsize = self.input_shape[0]
        self.eta = np.zeros(self.input_shape)
        # prediction 可以表示从FC层输出的数据 [batch, class_num] 或者 [batch, [p1,p2,...,pk]]
        self.logsoftmax = np.zeros(input_array.shape)
        self.softmax = np.zeros(input_array.shape)
        # 对每个batch的数据求softmax
        for i in range(self.batchsize):
            self.softmax[i] = self.cal_softmax(input_array[i])
            self.logsoftmax[i] = np.log(self.softmax[i])
        # softmax[batch, class_num]
        return self.logsoftmax

    def gradient(self, eta):
        self.eta = eta
        self.eta_next = self.softmax.copy()
        # print('softmax :\n', self.eta_next)
        # y 的标签是 One-hot 编码
        if self.eta.ndim>1: # one-hot
            for i in range(self.batchsize):
                self.eta_next[i] += self.eta[i]
        elif self.eta.ndim==1: # 非one-hot
            for i in range(self.batchsize):
                self.eta_next[i, -self.eta[i]] -= 1
        # eta[batchsize, class_num]
        # 需要除以batchsize用于平均该批次的影响
        self.eta_next /= self.batchsize
        return self.eta_next

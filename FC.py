'''
FC.py用于实现Fully Connected Layer
全连接层应该相当于DNN吧，深度神经网络，第l层每个神经元和l+1层每个神经元都相连
'''

import numpy as np
from Module import Module
from Parameter import Parameter

class FullyConnect(Module):
    def __init__(self, in_num, out_num, required_bias=True):
        super(FullyConnect, self).__init__()
        # input_shape = [batchsize, channel_num, height, width](卷积层)
        #  or [batchsize, input_num](全连接层)
        self.in_num = in_num
        # output_shape = [batchsize, out_num] 其实单个output就是个一维数组,列向量
        self.out_num = out_num
        self.required_bias = required_bias
        '''使用xavier初始化'''
        # 初始化全连接层为输入的weights
        # param_weights = np.random.standard_normal((self.in_num, self.out_num))/100
        param_weights = self.xavier_init(self.in_num, self.out_num, (self.in_num, self.out_num))
        self.weights = Parameter(param_weights, requires_grad=True)
        # bias初始化为列向量
        # param_bias = np.random.standard_normal(self.out_num)/100
        param_bias = self.xavier_init(self.in_num, self.out_num, (self.out_num))
        self.bias = Parameter(param_bias, requires_grad=True)

    # 设置特定的权重和偏移量
    def set_weight(self, weight):
        if isinstance(weight, Parameter):
            self.weights = weight

    def set_bias(self, bias):
        if isinstance(bias, Parameter):
            self.bias = bias

    def xavier_init(self, fan_in, fan_out, shape, constant=1, ):
        # 这个初始化对不收敛的问题没啥帮助= =
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(low, high, shape)

    # 设置module打印格式
    def extra_repr(self):
        s = ('in_features={in_num}, out_features={out_num}')
        if self.bias is None:
            s += ', bias=False'
        if self.required_bias:
            s += ', bias=True'
        return s.format(**self.__dict__)

    # 前向传播计算
    def forward(self, input_array):
        self.input_shape = input_array.shape
        self.batchsize = self.input_shape[0]
        # 对batchsize中的每个输入数据进行全连接计算
        # input_col=[batchsize, in_num], weights=[in_num, out_num]
        # 这种结构适合使用batch计算
        self.input_col = input_array.reshape([self.batchsize, -1])
        # print('input_shape: \n', self.input_col.shape)
        '''
            [Z1,Z2,...Zm]=[m x n矩阵]*[A1,A2,...An]+[B1,B2,...Bm]
            输入输出均拉为列向量
        '''
        # output_array = [batchsize, out_num]
        output_array = np.dot(self.input_col, self.weights.data) + self.bias.data
        # print('output_shape: \n',output_array.shape)
        return output_array
            
    # 梯度计算函数
    def gradient(self, eta):
        # eta=[batchsize, out_num]
        self.eta = eta
        # print('eta.shape: \n',self.eta.shape)
        # DNN反向传播, 计算delta_W
        for i in range(0, self.eta.shape[0]):
            # input_col_i=[in_num, none]
            input_col_i = self.input_col[i][:, np.newaxis]
            # eta_i=[out_num, none], eta_i.T=[none, out_num]
            eta_i = self.eta[i][:, np.newaxis].T
            # print('input_col_i: ',input_col_i.shape)
            # print('eta_i: ',eta_i.shape)
            # 利用每个batch输出参数误差累加计算梯度
            # weights=[out_num, in_num]
            self.weights.grad += np.dot(input_col_i, eta_i)
            self.bias.grad += eta_i.reshape(self.bias.data.shape)

        # print('eta shape: \n',self.eta.shape)
        # print('weight.data shape: \n',self.weights.data.shape)
        # 计算上一层的误差 eta=[batch,out_num], weights=[in_num,out_num]
        self.eta_next = np.dot(self.eta, self.weights.data.T) # eta_next=[batch, in_num]

        return self.eta_next

def test_FC():
    fc_input = np.arange(54).reshape(2,3,3,3)
    # print('input: ', fc_input)
    fc1 = FullyConnect(27, 10)
    # print('fc.weight: ', fc1.weights)
    fc1_out = fc1.forward(fc_input)
    print('fc_out: \n', fc1_out)
    print(fc1_out.shape)

    
    # copy()就是深度拷贝
    fc1_next = fc1_out.copy()+1
    print('fc1_next: ',fc1_next-fc1_out)
    fc1_next1 = fc1.gradient(fc1_next-fc1_out)
    print('fc_next error: ', fc1_next1)
    print(fc1.weights_grad)
    print(fc1.bias_grad)
    # 反向传播
    fc1.backward()
    

if __name__ == '__main__':
    test_FC()
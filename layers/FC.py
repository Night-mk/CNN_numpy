'''
FC.py用于实现Fully Connected Layer
全连接层应该相当于DNN吧，深度神经网络，第l层每个神经元和l+1层每个神经元都相连
'''

import numpy as np

class FullyConnect(object):
    def __init__(self, input_shape, output_num, learning_rate=0.0001):
        # input_shape = [batchsize, channel_num, height, width](卷积层)
        #  or [batchsize, input_num](全连接层)
        self.input_shape = input_shape
        self.batchsize = input_shape[0]

        # output_shape = [batchsize, output_num] 其实单个output就是个一维数组,列向量
        self.output_num = output_num
        self.output_shape = [self.batchsize, self.output_num]
        
        # 初始化全连接层为输入的weights
        self.weights = np.random.uniform(-1e-2, 1e-2,(self.input_shape[1], self.output_num))
        # 初始化卷积层为输入的weights
        if len(input_shape)>2:
            self.weights = np.random.uniform(-1e-2, 1e-2,(self.input_shape[1] * self.input_shape[2] * self.input_shape[3], self.output_num))
        # bias初始化为列向量
        self.bias = np.random.uniform(-1e-2, 1e-2,(self.output_num))

        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros(self.bias.shape)

        self.learning_rate = learning_rate

    # 设置特定的权重和偏移量
    def set_weight(self, weight):
        self.weights = weight

    def set_bias(self, bias):
        self.bias = bias

    # 前向传播计算
    def forward(self, input_array):
        # 对batchsize中的每个输入数据进行全连接计算
        # input_col=[batchsize, in_num], weights=[in_num, out_num]
        # 这种结构适合使用batch计算
        self.input_col = input_array.reshape([self.batchsize, -1])
        print(self.input_col.shape)
        '''
            [Z1,Z2,...Zm]=[m x n矩阵]*[A1,A2,...An]+[B1,B2,...Bm]
            输入输出均拉为列向量
        '''
        # output_array = [batchsize, out_num]
        output_array = np.dot(self.input_col, self.weights) + self.bias
        print(output_array.shape)
        return output_array
            
    # 梯度计算函数
    def gradient(self, eta):
        # eta=[batchsize, out_num]
        self.eta = eta
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
            self.weights_grad += np.dot(input_col_i, eta_i)
            self.bias_grad += eta_i.reshape(self.bias.shape)

        # 计算上一层的误差 eta=[batch,out_num,1], weights=[out_num,in_num]
        eta_next = []
        for i in range(0, self.eta.shape[0]):
            eta_next_i = np.dot(self.weights, self.eta[i][:, np.newaxis])
            eta_next.append(eta_next_i)
        # 将eta_next转换为输入input的shape，因为上一层不一定也是全连接层
        eta_next = np.array(eta_next)
        eta_next = np.reshape(eta_next, self.input_shape)

        return eta_next
    
    # 反向传播更新函数
    def backward(self):
        # 反向传播时更新权重参数
        self.weights -= self.learning_rate * self.weights_grad
        self.bias -= self.learning_rate * self.bias_grad

        # 将该层梯度重新初始化，用于接收下次迭代的梯度计算
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad= np.zeros(self.bias.shape)

def test_FC():
    fc_input = np.arange(54).reshape(2,3,3,3)
    # print('input: ', fc_input)
    fc1 = FullyConnect(fc_input.shape, 10)
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
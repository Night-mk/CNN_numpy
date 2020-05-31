'''
Pool.py用于实现Pooling Layer 池化层，用于减少计算参数，获取重要信息
简单起见，单独实现MaxPooling
'''
import numpy as np
from Module import Module

class MaxPooling(Module):
    # 初始化参数：输入数据大小、池化操作类型
    def __init__(self, pool_shape, stride):
        super(MaxPooling, self).__init__()
        # input_shape = [batchsize, channel_num, height, width]
        # pool_shape = [p_h, p_w]
        self.pool_shape = pool_shape
        self.pool_height = pool_shape[0]
        self.pool_width = pool_shape[1]
        # stride = [h,w]
        self.stride = stride
    
    # 设置module打印格式
    def extra_repr(self):
        s = ('kernel_size={pool_shape}, stride={stride}')
        return s.format(**self.__dict__)

    def forward(self, input_array):
        self.input_shape = input_array.shape
        self.batchsize = self.input_shape[0]
        self.channel_num = self.input_shape[1] 
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        # 计算,初始化pooling输出大小 = [batchsize, channel_num, out_h, out_w]
        self.output_shape = np.zeros((self.batchsize, self.channel_num, (self.input_height-self.pool_height)//self.stride[0]+1, (self.input_width-self.pool_width)//self.stride[1]+1))
        # 记录maxpooling取值的索引index，梯度计算需要？
        self.pool_index = np.zeros(self.input_shape)

        '''计算maxpooling'''
        output_array = np.zeros(self.output_shape.shape)
        # 对每个batch的每个channel的参数矩阵进行取最大值操作
        for b in range(0, self.batchsize):
            for c in range(self.channel_num):
                for i in range(0, self.input_height, self.stride[0]):
                    for j in range(0, self.input_width, self.stride[1]):
                        # 计算最大值输出矩阵,每次矩阵的移动和宽、高的步长相关
                        # 选取最大值的矩阵和filter的大小相关
                        output_array[b,c,i//self.stride[0],j//self.stride[1]] = np.max(input_array[b,c,i:i+self.pool_height,j:j+self.pool_width])
                        # 记录取max值的原矩阵的数据的索引
                        # 取到的argmax是pool_shape[pool_height, pool_width]大小的相对索引
                        index = np.argmax(input_array[b,c,i:i+self.pool_height,j:j+self.pool_width])
                        # 在原矩阵中标记被选取为max的数据位置为1，其他位置为0，方便反向传播时计算input的误差矩阵
                        # index = i*pool_width+j
                        self.pool_index[b, c, i + index//self.pool_width, j + index % self.pool_width] = 1
        return output_array

    # 反向传播函数，由于没有参数需要学习，故不用计算梯度，只需要将误差传递到上一层
    def gradient(self, eta):
        # 使用np.repeat扩展误差矩阵大小为输入矩阵的大小
        self.eta_next = np.repeat(np.repeat(eta, self.stride[1], axis=3), self.stride[0], axis=2) * self.pool_index
        return self.eta_next

def maxpooling_test():
    a = np.arange(16).reshape((1,1,4,4))
    pool1 = MaxPooling(pool_shape=(2,2), stride=(2,2))
    pool_out = pool1.forward(a)
    print("pool result: ",pool_out)
    eta = pool_out.copy()+1
    eta_next = pool1.backward(eta-pool_out)
    print(eta_next)

def argmax_test():
    a = np.arange(16).reshape((4,4))
    index = np.argmax(a[1:4, 1:4])
    print(a)
    print(index)
    print(a[1:4, 1:4])
    


if __name__ == '__main__':
    # maxpooling_test()
    argmax_test()


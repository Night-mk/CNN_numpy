'''
    Activators.py文件构建激活函数的前向，反向传播计算
'''
import numpy as np
from Module import Module

'''
    Square
'''
class Square(Module):
    def __init__(self):
        super(Square, self).__init__()
    # 设置module打印格式
    def extra_repr(self):
        s = ()
        return s
    # 前向传播
    def forward(self, input_array):
        # 求平方，做激活函数？？
        self.input_array = input_array
        return self.input_array**2

    def gradient(self, eta):
        self.eta_next = eta*2*self.input_array
        return self.eta_next


'''
    ReLU
'''
class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        # 反向传播需要使用
        # self.input_array = np.zeros(input_shape)
        # self.eta = np.zeros(input_shape)

    # 设置module打印格式
    def extra_repr(self):
        s = ()
        return s
    # 前向传播
    def forward(self, input_array):
        self.input_array = input_array
        # 使用0和input_array的元素依次比较
        # np.maximum：(X, Y, out=None) X与Y逐位比较取其大者
        return np.maximum(self.input_array, 0)
    
    # 反向传播
    # relu'(x)=1*dy if x>0
    # relu'(x)=0*dy if x<0
    # relu'(x)不存在 if x=0.0000... （代码实现里将=0的结果设置为1）
    def gradient(self, eta):
        self.eta_next = eta
        self.eta_next[self.input_array<0]=0
        # self.eta_next[self.input_array==0]=1 # 将等于0的数据导数设置为1，看看能不能解决梯度消失的问题
        return self.eta_next

'''
    LeakyReLU
'''
class LeakyReLU(Module):
    def __init__(self, alpha1=0.01):
        # self.input_array = np.zeros(input_shape)
        # self.eta = np.zeros(input_shape)
        super(LeakyReLU, self).__init__()
        self.alpha1 = alpha1
    
    # 设置module打印格式
    def extra_repr(self):
        s = ('alpha={alpha1}')
        return s.format(**self.__dict__)

    # 前向传播
    # leakyrelu(x)=x if x>0
    # leakyrelu(x)=ax if x<=0
    def forward(self, input_array):
        self.input_array = input_array
        self.output_array = self.input_array.copy()
        self.output_array[self.input_array<0] *= self.alpha1
        return self.output_array

    # 反向传播
    def gradient(self, eta):
        self.eta_next = eta
        # print('eta shape: ',eta.shape)
        self.eta_next[self.input_array<=0] *= self.alpha1
        return self.eta_next

'''
    Sigmoid
'''
class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
    
    # 设置module打印格式
    def extra_repr(self):
        s = ()
        return s
    # 1/(1+e^-x)
    def forward(self, input_array):
        self.output_array = 1/(1+np.exp(-input_array))
        return self.output_array
    
    def gradient(self, eta):
        self.eta_next = eta * self.output_array*(self.output_array-1) # d(sigmoid)=y*(1-y)
        return self.eta_next


class Sigmoid_CE(Module):
    def __init__(self):
        super(Sigmoid_CE, self).__init__()
    
    # 设置module打印格式
    def extra_repr(self):
        s = ()
        return s
    # 1/(1+e^-x)
    def forward(self, input_array):
        self.output_array = 1/(1+np.exp(-input_array))
        return self.output_array
    
    def gradient(self, eta):
        n_dim = self.output_array.ndim
        for i in range(n_dim-1):
            eta = eta[:,np.newaxis]
        # eta = y, grad = sigmoid(x)-y
        self.eta_next = self.output_array - eta
        self.eta_next /= self.output_array.shape[0] # 需要求平均
        # print('sigmoid eta shape: \n', eta.shape)
        # print('sigmoid output_array shape: \n', self.output_array.shape)
        return self.eta_next

'''
    tanh
'''
class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
    
    # 设置module打印格式
    def extra_repr(self):
        s = ()
        return s

    # tanh=2*sigmoid(2x)-1
    def forward(self, input_array):
        self.output_array = 2/(1+np.exp(-2*input_array))-1
        return self.output_array

    def gradient(self, eta):
        self.eta_next = eta * (1-self.output_array**2)
        return self.eta_next


def test_leakyrelu():
    x = np.random.randn(1,1,4,4).astype(np.float32)
    dy = np.random.randn(1,1,4,4).astype(np.float32)
    print('x: \n', x)
    print('dy: \n', dy)

    lrelu = LeakyReLU()
    l_out = lrelu.forward(x)
    l_eta = lrelu.gradient(dy)
    print(l_out)
    print('----------')
    print(l_eta)

def test_relu():
    x = np.random.randn(1,1,4,4).astype(np.float32)
    dy = np.random.randn(1,1,4,4).astype(np.float32)
    print('x: \n', x)
    print('dy: \n', dy)

    relu = ReLU()
    l_out = relu.forward(x)
    l_eta = relu.gradient(dy)
    
    print(l_out)
    print('----------')
    print(l_eta)

if __name__ == "__main__":
    test_leakyrelu()
    # test_relu()
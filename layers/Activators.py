'''
    Activators.py文件构建激活函数的前向，反向传播计算
'''
import numpy as np

'''
    ReLU
'''
class ReLU(object):
    def __init__(self, input_shape):
        # 反向传播需要使用
        self.input_array = np.zeros(input_shape)
        self.eta = np.zeros(input_shape)

    # 前向传播
    def forward(self, input_array):
        self.input_array = input_array
        # 使用0和input_array的元素依次比较
        # np.maximum：(X, Y, out=None) X与Y逐位比较取其大者
        return np.maximum(input_array, 0)
    
    # 反向传播
    # relu'(x)=1*dy if x>0
    # relu'(x)=0*dy if x<0
    # relu'(x)不存在 if x=0.0000... （代码实现里将=0的结果设置为1）
    def gradient(self, eta):
        self.eta = eta
        self.eta[self.input_array<0]=0
        return self.eta


'''
    LeakyReLU
'''
class LeakyReLU(object):
    def __init__(self, input_shape, alpha=0.01):
        self.input_array = np.zeros(input_shape)
        self.eta = np.zeros(input_shape)
        self.alpha = alpha

    # 前向传播
    # leakyrelu(x)=x if x>0
    # leakyrelu(x)=ax if x<=0
    def forward(self, input_array):
        self.input_array = input_array
        self.output_array = self.input_array.copy()
        self.output_array[self.input_array<0] *= self.alpha
        return self.output_array

    # 反向传播
    def gradient(self, eta):
        self.eta = eta
        self.eta[self.input_array<0] *= self.alpha
        # print('input_array: \n', self.input_array)
        return self.eta
        
'''
    tanh
'''


'''
    Sigmoid
'''



def test_leakyrelu():
    x = np.random.randn(1,1,4,4).astype(np.float32)
    dy = np.random.randn(1,1,4,4).astype(np.float32)
    print('x: \n', x)
    print('dy: \n', dy)

    lrelu = LeakyReLU(x.shape)
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

    relu = ReLU(x.shape)
    l_out = relu.forward(x)
    l_eta = relu.gradient(dy)
    
    print(l_out)
    print('----------')
    print(l_eta)

if __name__ == "__main__":
    test_leakyrelu()
    # test_relu()
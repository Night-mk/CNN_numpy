'''
    Lenet_numpy.py 
    使用numpy实现可训练、预测的CNN框架
'''
import random
import numpy as np
from Module import Module
from Optimizer import Optimizer
from Parameter import Parameter
from Conv import ConvLayer

class Lenet_numpy(Module):
    def __init__(self):
        super(Lenet_numpy, self).__init__()
        self.conv1 = ConvLayer((1,3,5,5),3,3,5, zero_padding=1, stride=2, learning_rate=0.0001, method='SAME')
        # self.conv2 = ConvLayer((5,3,3,3),3,3,5, zero_padding=1, stride=2, learning_rate=0.0001, method='VALID')
        print(len(self._modules))
        print(self.conv1)

    def forward(self, x):
        out_c1s2 = self.conv1.forward(x)

        return out_c1s2
    
    def backward(self, y):
        out_back1 = self.conv1.gradient(y)
        return out_back1


if __name__ == "__main__":
    Lenet = Lenet_numpy()
    print('Lenet_numpy: \n', Lenet)
    print(Lenet._modules)
    x_numpy = np.random.randn(1,3,5,5).astype(np.float32)
    out_f = Lenet.forward(x_numpy)
    dy_numpy = np.random.random(out_f.shape).astype(np.float32)
    out_b = Lenet.backward(dy_numpy)
    
    for param in Lenet.parameters():
        print(param.grad)

    optim = Optimizer(Lenet.parameters(), None)
    optim.zero_grad()

    for param in Lenet.parameters():
        print('zero: \n', param.grad)
        
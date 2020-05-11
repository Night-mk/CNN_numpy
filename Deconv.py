'''
    Deconv.py 用于实现转置卷积（反卷积）计算，可以使用在GAN网络中
'''
import numpy as np
import Conv

class Deconv(object):
    def __init__(self, input_shape, out_channel, filter_size, learning_rate=0.001):
        return 0

    # 前向传播计算
    # 需要填充输入矩阵，计算填充大小，并执行卷积计算
    def forward(self, input_array):
        return 0

    # 计算w,b梯度，并计算向上一层传输的误差
    def gradient(self, eta):
        return 0

    def backward(self):
        return 0


def test_deconv():
    print('test')


if __name__ == "__main__":
    test_deconv()
'''
    component_verify.py 使用pytorch构建CNN网络及其组件，并验证numpy实现CNN的组件的正确性
'''

import torch 
import torch.nn as nn
from torch.nn import functional as F
import Loss
import Conv
import Pool
import Activators
import FC
import BN
import BN_other
import numpy as np

'''
    验证：损失函数CrossEntropyLoss（成功）
'''
def loss_test():
    # input=[batch, 3],表示每个batch中对每个类的预测值，总共有3类数据
    x_input = torch.randn(5, 3, requires_grad=True)
    x_input_numpy = x_input.detach().numpy()
    print('x_input: ',x_input)
    # print('x_input_numpy: ',x_input_numpy)

    # target=[batch] 表示每个数据分类的标签
    y_target_t = torch.tensor(5, dtype=torch.long).random_(3)
    print(y_target_t)
    y_target = torch.tensor([0,2,1,0,1])
    y_target_numpy = y_target.detach().numpy()
    print('y_target: ',y_target)
    print('y_target_numpy: ',y_target_numpy)

    # 初始化Softmax类
    softmax_numpy = Loss.Softmax(x_input_numpy.shape)

    '''
    # softmax (成功)
    softmax_func = nn.Softmax(dim=1)
    softmax_output = softmax_func(x_input)
    print('softmax_output: \n',softmax_output)

    softmax_output_numpy = softmax_numpy.predict(x_input_numpy)
    print('softmax_output_numpy: \n',softmax_output_numpy)
    '''

    # softmax+cross entropy (成功)
    cross_entropy_loss = nn.CrossEntropyLoss()
    output = cross_entropy_loss(x_input, y_target)
    eta = output.backward()
    print('cross_entropy_output: \n',output)
    print('grad_output: \n',x_input.grad)

    cross_entropy_loss_numpy = softmax_numpy.cal_loss(x_input_numpy, y_target_numpy)
    eta_numpy = softmax_numpy.gradient_with_loss()
    print('cross_entropy_loss_numpy: \n',cross_entropy_loss_numpy)
    print('grad_output_numpy: \n',eta_numpy)


'''
    验证：卷积计算Conv（成功）
'''
def conv_test():
    # 自定义卷积核和偏移量，使用pytorch计算卷积，和反向传播结果
    """手动定义卷积核(weight)和偏置"""
    w = torch.rand(5, 3, 3, 3)  # 5种3通道的3乘3卷积核
    b = torch.rand(5)  # 和卷积核种类数保持一致(不同通道共用一个bias)
    w_numpy = w.detach().numpy()
    b_numpy = b.detach().numpy()

    """定义输入样本"""
    x = torch.tensor(np.random.randn(1, 3, 5, 5).astype(np.float32), requires_grad=True)  # 1张3通道的5乘5的图像
    x_numpy = x.detach().numpy()

    """定义输出误差"""
    dy = torch.tensor(np.random.randn(1,5,3,3).astype(np.float32), requires_grad=True).float()
    dy_numpy = dy.detach().numpy()

    print('-------参数打印-------')
    print('w: \n', w_numpy)
    print('b: \n', b_numpy)
    print('x: \n', x_numpy)
    print('dy: \n', dy_numpy)

    # pytorch计算卷积前向传播
    conv_out = F.conv2d(x, w, b, stride=1, padding=0)
    # numpy计算卷积前向传播
    cl1 = Conv.ConvLayer(x_numpy.shape, 3,3,5, zero_padding=0, stride=1, learning_rate=0.0001, method='VALID') # convlayer
    cl1.set_weight(w_numpy)
    cl1.set_bias(b_numpy)
    conv_out_numpy = cl1.forward(x_numpy) # forward 
    
    # pytorch 计算卷积反向传播
    conv_out.backward(dy)
    # numpy 计算卷积反向传播
    eta_next = cl1.gradient(dy_numpy)
    cl1.backward() 

    # 输出结果对比
    print('-----对比输出-----')
    print('conv_out: \n', conv_out)
    print('conv_out.shape: \n', conv_out.shape)

    print('conv_out_numpy: \n', conv_out_numpy)
    print('conv_out_numpy.shape: \n', conv_out_numpy.shape)
    
    print('-----对比x_grad-----')
    print('x_grad: \n', x.grad)
    print('x_grad.shape: \n', x.grad.shape)

    print('x_grad_numpy: \n', eta_next)
    print('x_grad_numpy.shape: \n', eta_next.shape)

'''
    验证：激活层AC（成功）
'''
def ac_test():
    """定义输入样本"""
    # 输入数据大小为 1x2x4x4
    x = torch.tensor(np.random.randn(1,2,4,4).astype(np.float32), requires_grad=True)
    x_numpy = x.detach().numpy()

    """定义误差"""
    dy = torch.tensor(np.random.randn(1,2,4,4).astype(np.float32), requires_grad=True)
    dy_numpy = dy.detach().numpy()

    # pytorch
    r_out = F.relu(x)
    r_out.backward(dy)
    print('r_out: \n', r_out)

    # numpy
    relu1 = Activators.ReLU(x_numpy.shape)
    r_out_numpy = relu1.forward(x_numpy)
    r_eta = relu1.gradient(dy_numpy)
    print('r_out: \n', r_out_numpy)

    # 反向传播
    print('r_out_grad: \n', x.grad)
    print('r_out_grad_numpy: \n', r_eta)


'''
    验证：批量标准化层BN（成功）
'''
def bn_test():
    """定义输入"""
    x_numpy = np.random.randn(1,5,2,2).astype(np.float32)
    x = torch.tensor(x_numpy, requires_grad=True)
    # 初始化
    # pytorch (需要添加affine=False参数)
    bn_tensor = torch.nn.BatchNorm2d(5, affine=False)
    # numpy
    bn_numpy = BN.BatchNorm(x_numpy.shape)
    # bn_numpy1 = BN_other.BatchNormal()
    """计算前向传播"""
    bn_out_tensor = bn_tensor(x)
    bn_out_numpy = bn_numpy.forward(x_numpy,'train')
    # bn_out_numpy1 = bn_numpy1.forward(x_numpy, axis=1)

    """计算反向传播"""
    # 误差参数初始化
    dy_numpy = np.random.random(bn_out_numpy.shape).astype(np.float32)
    dy = torch.tensor(dy_numpy, requires_grad=True)
    # 反向计算
    # pytorch
    bn_out_tensor.backward(dy)
    x_grad_tensor = x.grad
    # numpy
    x_grad_numpy = bn_numpy.gradient(dy_numpy)
    
    
    """打印输出"""
    print('-----对比输出-----')
    print('bn_out_tensor: \n',bn_out_tensor)
    print('bn_out_tensor: \n',bn_out_tensor.shape)
    print('bn_out_numpy: \n',bn_out_numpy)
    print('bn_out_numpy: \n',bn_out_numpy.shape)

    print('-----对比x_grad-----')
    print('x_grad_tensor: \n',x_grad_tensor)
    print('x_grad_numpy: \n',x_grad_numpy)


'''
    验证：全连接层FC（成功）
'''
def fc_test():
    """定义输入样本"""
    # 输入数据大小为 1x5x8x8 输出大小定为10
    x_numpy = np.random.randn(1,5,8,8).astype(np.float32)
    x = torch.tensor(x_numpy, requires_grad=True)
    output_size = 10
    # 定义全连接层参数
    w_numpy = np.random.randn(5*8*8, output_size).astype(np.float32)
    b_numpy = np.random.randn(output_size,).astype(np.float32)
    w = torch.tensor(w_numpy.T, requires_grad=True)
    b = torch.tensor(b_numpy, requires_grad=True)

    
    # pytorch
    fc_tensor = torch.nn.Linear(5, output_size, bias=True)
    fc_tensor.weight = torch.nn.Parameter(w, requires_grad=True)
    fc_tensor.bias = torch.nn.Parameter(b, requires_grad=True)

    # numpy
    fc1 = FC.FullyConnect(x.shape, output_size)
    fc1.set_weight(w_numpy)
    fc1.set_bias(b_numpy)
    
    """计算反向传播"""
    fc_out_tensor = fc_tensor(x.view(5*8*8))
    fc_out_numpy = fc1.forward(x_numpy)

    """计算反向传播"""
    # 定义误差
    dy_numpy = np.random.random(fc_out_numpy.shape)
    dy = torch.FloatTensor(dy_numpy)

    fc_out_tensor.backward(dy)
    x_grad = x.grad
    w_grad = fc_tensor.weight.grad
    b_grad = fc_tensor.bias.grad

    x_grad_numpy = fc1.gradient(dy_numpy)
    w_grad_numpy = fc1.weights_grad
    b_grad_numpy = fc1.bias_grad

    # 打印结果
    print('-----对比输出-----')
    print('fc_out_tensor: \n',fc_out_tensor)
    print('fc_out_tensor: \n',fc_out_tensor.shape)
    print('fc_out_numpy: \n',fc_out_numpy)
    print('fc_out_numpy: \n',fc_out_numpy.shape)

    print('-----对比x_grad-----')
    print('x_grad: \n',x_grad)
    print('x_grad_numpy: \n',x_grad_numpy-x_grad.numpy())

    print('-----对比w_grad-----')
    print('w_grad: \n',w_grad)
    print('w_grad: \n',w_grad.shape)
    print('w_grad_numpy: \n',w_grad_numpy)
    print('w_grad_numpy: \n',w_grad_numpy.shape)
    
    print('-----对比b_grad-----')
    print('b_grad: \n',b_grad)
    print('b_grad_numpy: \n',b_grad_numpy)
    

'''
    验证：池化层Pooling（成功）
'''
def pooling_test():
    """定义输入样本"""
    # 输入数据大小为 1x5x8x8
    x = torch.tensor(np.random.randn(1,5,8,8).astype(np.float32), requires_grad=True)
    x_numpy = x.detach().numpy()

    """定义误差"""
    dy = torch.tensor(np.random.randn(1,5,4,4).astype(np.float32), requires_grad=True)
    dy_numpy = dy.detach().numpy()

    # pytorch pool=(2,2) stride=2
    pool_out = F.max_pool2d(x, kernel_size=2, stride=2)
    pool_out.backward(dy)
    print('pool_out: \n', pool_out)
    print('pool_out.shape: \n', pool_out.shape)

    # numpy
    pool1 = Pool.MaxPooling(x_numpy.shape,  pool_shape=(2,2), stride=(1,1,2,2))
    pool_out_numpy = pool1.forward(x_numpy)
    pool_eta = pool1.backward(dy_numpy)
    print('pool_out_numpy: \n', pool_out_numpy)
    print('pool_out_numpy.shape: \n', pool_out_numpy.shape)

    # 反向传播误差对比
    print('pool_out_grad: \n', x.grad)
    print('pool_out_grad.shape: \n', x.grad.shape)

    print('pool_out_numpy_grad: \n', pool_eta)
    print('pool_out_numpy_grad.shape: \n', pool_eta.shape)

    
if __name__ == '__main__':
    # loss_test()
    # conv_test()
    # ac_test()
    bn_test()
    # fc_test()
    # pooling_test()
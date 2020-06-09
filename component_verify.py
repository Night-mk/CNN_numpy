'''
    component_verify.py 使用pytorch构建CNN网络及其组件，并验证numpy实现CNN（重构）的组件的正确性
    -- 卷积计算Conv
    -- 激活层AC
    -- 批量标准化层BN
    -- 全连接层FC
    -- 池化层Pooling
    -- 损失函数CrossEntropyLoss
    -- 反卷积层Deconv
'''

import torch 
import torch.nn as nn
from torch.nn import functional as F
import Loss
import Logsoftmax
import Conv
import Pool
import Activators
import FC
import BN
import Deconv
import numpy as np
from Parameter import Parameter

'''
    验证：损失函数CrossEntropyLoss（成功）
'''
def loss_test():
    # input=[batch, 3],表示每个batch中对每个类的预测值，总共有3类数据
    x_input = torch.randn(6, 3, requires_grad=True)
    x_input_numpy = x_input.detach().numpy()
    print('x_input: ',x_input)
    # print('x_input_numpy: ',x_input_numpy)

    # target=[batch] 表示每个数据分类的标签
    # y_target = torch.tensor([[1,0,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[1,0,0,0,0],[0,0,0,0,1]])
    y_target = torch.tensor([0,2,1,0,1,2])
    y_target_numpy = y_target.detach().numpy()
    print('y_target: ',y_target)
    print('y_target_numpy: ',y_target_numpy)

    # 初始化LogSoftmax类
    logsoftmax_tensor = nn.LogSoftmax(dim=1)
    logsoftmax_output_tensor = logsoftmax_tensor(x_input)

    logsoftmax_numpy = Logsoftmax.Logsoftmax()
    logsoftmax_output_numpy = logsoftmax_numpy.forward(x_input_numpy)

    print('-----对比输出-----')
    print('logsoftmax_output_tensor: \n',logsoftmax_output_tensor)
    print('logsoftmax_output_numpy: \n',logsoftmax_output_numpy)
    
    loss_tensor = nn.NLLLoss()
    cross_entropy_loss_output_tensor = loss_tensor(logsoftmax_output_tensor, y_target)
    cross_entropy_loss_output_tensor.backward()
    loss_grad_tensor = x_input.grad
    
    loss_numpy = Loss.NLLLoss()
    cross_entropy_loss_output_numpy = loss_numpy.cal_loss(logsoftmax_output_numpy, y_target_numpy)
    eta = loss_numpy.gradient()
    print('eta: \n', eta)
    loss_grad_numpy = logsoftmax_numpy.gradient(eta)

    print('-----前向传播-----')
    print('cross_entropy_loss_output_tensor: \n',cross_entropy_loss_output_tensor)
    print('cross_entropy_loss_output_numpy: \n',cross_entropy_loss_output_numpy)

    print('-----反向传播-----')
    print('loss_grad_tensor: \n',loss_grad_tensor)
    print('loss_grad_numpy: \n',loss_grad_numpy)


    '''
    # softmax (成功)
    softmax_func = nn.Softmax(dim=1)
    softmax_output = softmax_func(x_input)
    print('softmax_output: \n',softmax_output)

    softmax_output_numpy = softmax_numpy.predict(x_input_numpy)
    print('softmax_output_numpy: \n',softmax_output_numpy)
    '''
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

'''
    验证：卷积计算Conv（成功）
    测试了多种条件（stride,padding,batch）变化下的卷积正确性
    测试时需要注释掉Conv.py中的kaiming初始化
'''
def conv_test():
    # 自定义卷积核和偏移量，使用pytorch计算卷积，和反向传播结果
    """手动定义卷积核(weight)和偏置"""
    w = torch.rand(5, 3, 3, 3)  # 5种3通道的3乘3卷积核
    b = torch.rand(5)  # 和卷积核种类数保持一致(不同通道共用一个bias)
    w_numpy = w.detach().numpy()
    b_numpy = b.detach().numpy()

    w_2 = torch.rand(5, 5, 3, 3)  # 5种5通道的3乘3卷积核
    b_2 = torch.rand(5)  # 和卷积核种类数保持一致(不同通道共用一个bias)
    w_numpy_2 = w_2.detach().numpy()
    b_numpy_2 = b_2.detach().numpy()

    """定义输入样本"""
    # 单张图片输入
    x = torch.tensor(np.random.randn(3, 3, 5, 5).astype(np.float32), requires_grad=True)  # 1张3通道的5乘5的图像
    # 多张图片输入
    # x = torch.tensor(np.random.randn(2, 3, 5, 5).astype(np.float32), requires_grad=True)  # 2张3通道的5乘5的图像
    x_numpy = x.detach().numpy()

    print('-------参数打印-------')
    # print('w: \n', w_numpy)
    # print('b: \n', b_numpy)
    # print('x: \n', x_numpy)

    """前向传播"""
    # pytorch计算卷积前向传播
    ## padding=0 stride=0
    # cl_tensor = torch.nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=0)
    ## padding=1 stride=1
    cl_tensor_1 = torch.nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1)
    ## stride=2 padding=0
    # cl_tensor_1 = torch.nn.Conv2d(3, 5, kernel_size=3, stride=2, padding=0)
    ## stride=2 padding=1
    # cl_tensor = torch.nn.Conv2d(3, 5, kernel_size=3, stride=2, padding=1)
    ## stride=1 padding=2
    # cl_tensor_1 = torch.nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=2)
    cl_tensor_2 = torch.nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=0)

    cl_tensor_1.weight = torch.nn.Parameter(w, requires_grad=True)
    cl_tensor_1.bias = torch.nn.Parameter(b, requires_grad=True)
    cl_tensor_2.weight = torch.nn.Parameter(w_2, requires_grad=True)
    cl_tensor_2.bias = torch.nn.Parameter(b_2, requires_grad=True)

    conv_out_tensor_1 = cl_tensor_1(x)
    conv_out_tensor_2 = cl_tensor_2(conv_out_tensor_1)
    # numpy计算卷积前向传播
    ## padding=0 stride=0
    # cl1 = Conv.ConvLayer(3, 5, 3, 3, zero_padding=0, stride=1, method='VALID')
    ## padding=1 stride=1
    cl1 = Conv.ConvLayer(3, 5, 3, 3, zero_padding=1, stride=1, method='SAME')
    ## stride=2 padding=0
    # cl1 = Conv.ConvLayer(3, 5, 3, 3, zero_padding=0, stride=2, method='VALID')
    ## stride=2 padding=1
    # cl1 = Conv.ConvLayer(3, 5, 3,3, zero_padding=1, stride=2, method='SAME')
    ## stride=1 padding=2
    # cl1 = Conv.ConvLayer(3, 5, 3,3, zero_padding=2, stride=1, method='SAME')
    cl2 = Conv.ConvLayer(5, 5, 3,3, zero_padding=0, stride=1, method='VALID')

    cl1.set_weight(Parameter(w_numpy, requires_grad=True))
    cl1.set_bias(Parameter(b_numpy, requires_grad=True))
    cl2.set_weight(Parameter(w_numpy_2, requires_grad=True))
    cl2.set_bias(Parameter(b_numpy_2, requires_grad=True))

    conv_out_numpy_1 = cl1.forward(x_numpy) # forward
    print('conv_out_numpy_1.shape: ',conv_out_numpy_1.shape)
    conv_out_numpy_2 = cl2.forward(conv_out_numpy_1) 
    print('conv_out_numpy_2.shape: ',conv_out_numpy_2.shape)

    # 输出结果对比
    print('-----对比输出-----')
    print('conv_out_tensor_1: \n', conv_out_tensor_1[0][0])
    # print('conv_out_tensor_1.shape: \n', conv_out_tensor_1.shape)
    # print('conv_out_numpy_1: \n', conv_out_numpy_1)
    # print('conv_out_numpy_1.shape: \n', conv_out_numpy_1.shape)
    print('conv_out_1 error: \n', conv_out_numpy_1-conv_out_tensor_1.detach().numpy())

    print('conv_out_tensor_2: \n', conv_out_tensor_2[0][0])
    # print('conv_out_tensor_2.shape: \n', conv_out_tensor_2.shape)
    print('conv_out_numpy_2: \n', conv_out_numpy_2[0][0])
    # print('conv_out_numpy_2.shape: \n', conv_out_numpy_2.shape)
    print('conv_out_2 error: \n', conv_out_numpy_2-conv_out_tensor_2.detach().numpy())

    # print('conv_out_2 weight error: \n', cl2.weights.data-cl_tensor_2.weight.detach().numpy())
    # print('conv_out_2 bias error: \n', cl2.bias.data-cl_tensor_2.bias.detach().numpy())

    
    """梯度计算"""
    """定义输出误差"""
    dy_numpy = np.random.random(conv_out_numpy_2.shape).astype(np.float32)
    dy = torch.tensor(dy_numpy, requires_grad=True).float()
    # print('dy: \n', dy_numpy)
    print('dy.shape: \n', dy_numpy.shape)

    ## pytorch 计算卷积反向传播
    conv_out_tensor_2.backward(dy)
    # conv_out_tensor_1.backward(dy_grad)
    # dy.backward()
    # x_grad_1 = x.grad
    w_grad_1 = cl_tensor_1.weight.grad
    b_grad_1 = cl_tensor_1.bias.grad
    w_grad_2 = cl_tensor_2.weight.grad
    b_grad_2 = cl_tensor_2.bias.grad

    ## numpy 计算卷积反向传播
    x_grad_numpy_2 = cl2.gradient(dy_numpy)
    x_grad_numpy_1 = cl1.gradient(x_grad_numpy_2)

    w_grad_numpy_2 = cl2.weights.grad
    b_grad_numpy_2 = cl2.bias.grad
    
    w_grad_numpy_1 = cl1.weights.grad
    b_grad_numpy_1 = cl1.bias.grad

    
    
    print('-----对比x_grad-----')
    # print('x_grad: \n', x_grad)
    # print('x_grad.shape: \n', x_grad.shape)

    # print('x_grad_numpy: \n', x_grad_numpy_1)
    # print('x_grad_numpy.shape: \n', x_grad_numpy_1.shape)

    # print('x_grad error mean: \n', np.mean(x_grad_numpy-x_grad.detach().numpy(), axis = 3))
    # print('x_grad error shape: \n', np.mean(x_grad_numpy-x_grad.detach().numpy(), axis = 3).shape)

    print('-----对比w_grad-----')
    # print('w_grad: \n', w_grad_1)
    # print('w_grad_numpy: \n', w_grad_numpy_1)
    print('w_grad_1 error: \n', w_grad_numpy_1-w_grad_1.detach().numpy())
    print('w_grad_2 error: \n', w_grad_numpy_2-w_grad_2.detach().numpy())

    print('-----对比b_grad-----')
    # print('b_grad: \n', b_grad_1)
    # print('b_grad_numpy: \n', b_grad_numpy_1)

    print('b_grad_1 error: \n', b_grad_numpy_1-b_grad_1.detach().numpy())
    print('b_grad_2 error: \n', b_grad_numpy_2-b_grad_2.detach().numpy())


'''
    Conv梯度检测
'''
def conv_checkgrad():
    """梯度检测"""
    eps = 1e-4

    """手动定义卷积核(weight)和偏置"""
    w = torch.rand(5, 3, 3, 3)  # 5种3通道的3乘3卷积核
    b = torch.rand(5)  # 和卷积核种类数保持一致(不同通道共用一个bias)
    w_numpy = w.detach().numpy()
    b_numpy = b.detach().numpy()

    """定义输入样本"""
    x = torch.tensor(np.random.randn(1, 3, 3, 3).astype(np.float32), requires_grad=True)  # 1张3通道的5乘5的图
    x_numpy = x.detach().numpy()

    """卷积初始化"""
    cl_tensor_1 = torch.nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=0)
    cl_tensor_1.weight = torch.nn.Parameter(w, requires_grad=True)
    cl_tensor_1.bias = torch.nn.Parameter(b, requires_grad=True)

    cl1 = Conv.ConvLayer(3, 5, 3,3, zero_padding=0, stride=1, method='VALID')

    # cl2 = Conv.ConvLayer(3, 5, 3,3, zero_padding=1, stride=1, method='SAME')
    # cl3 = Conv.ConvLayer(3, 5, 3,3, zero_padding=1, stride=1, method='SAME')

    cl1.set_weight(Parameter(w_numpy, requires_grad=True))
    cl1.set_bias(Parameter(b_numpy, requires_grad=True))
    # cl2.set_weight(Parameter(w_numpy, requires_grad=True))
    # cl2.set_bias(Parameter(b_numpy, requires_grad=True))
    # cl3.set_weight(Parameter(w_numpy, requires_grad=True))
    # cl3.set_bias(Parameter(b_numpy, requires_grad=True))

    conv_out_tensor_1 = cl_tensor_1(x)
    conv_out_numpy_1 = cl1.forward(x_numpy) # forward
    # conv_out_numpy_2 = cl2.forward(x_numpy-eps) # forward
    # conv_out_numpy_3 = cl3.forward(x_numpy+eps) # forward

    print('-----对比输出-----')
    print('conv_out_tensor_1: \n', conv_out_tensor_1)
    # print('conv_out_tensor_1.shape: \n', conv_out_tensor_1.shape)
    print('conv_out_numpy_1: \n', conv_out_numpy_1)
    # print('conv_out_numpy_1.shape: \n', conv_out_numpy_1.shape)
    print('conv_out_1 error: \n', conv_out_numpy_1-conv_out_tensor_1.detach().numpy())

    """梯度计算"""
    """定义输出误差"""
    dy_numpy = np.ones(conv_out_numpy_1.shape).astype(np.float32)
    dy = torch.tensor(dy_numpy, requires_grad=True).float()

    x_grad = cl1.gradient(dy_numpy)
    
    # x_grad_check = (conv_out_numpy_1-conv_out_numpy_2)/2/eps

    # print("-------梯度检测-------")
    # print('x_grad: \n',x_grad)
    # print('x_grad_check: \n',x_grad_check)
    # print('x_grad_check error: \n',x-x_grad_check)




'''
    验证：激活层relu（成功）
'''
def relu_test():
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
    relu1 = Activators.ReLU()
    r_out_numpy = relu1.forward(x_numpy)
    r_eta = relu1.gradient(dy_numpy)
    print('r_out_numpy: \n', r_out_numpy)

    # 反向传播
    print('r_out_grad: \n', x.grad)
    print('r_out_grad_numpy: \n', r_eta)


'''
    验证：激活层leakyrelu（成功）
'''
def leakyrelu_test():
    """定义输入样本"""
    # 输入数据大小为 1x2x4x4
    x = torch.tensor(np.random.randn(1,2,4,4).astype(np.float32), requires_grad=True)
    x_numpy = x.detach().numpy()

    """定义误差"""
    dy = torch.tensor(np.random.randn(1,2,4,4).astype(np.float32), requires_grad=True)
    dy_numpy = dy.detach().numpy()

    """前向传播"""
    # pytorch
    lr_tensor = nn.LeakyReLU(0.2)
    lr_out_tensor = lr_tensor(x)
    # numpy
    lr_numpy = Activators.LeakyReLU(0.2)
    lr_out_numpy = lr_numpy.forward(x_numpy)

    """反向传播"""
    # pytorch
    lr_out_tensor.backward(dy)
    x_grad = x.grad

    # numpy
    x_grad_numpy = lr_numpy.gradient(dy_numpy)

    print('-----对比输出-----')
    print('lr_out_tensor: \n', lr_out_tensor)
    print('lr_out_numpy: \n', lr_out_numpy)
    print('lr_out_error: \n', lr_out_numpy-lr_out_tensor.detach().numpy())

    print('-----对比x_grad-----')
    print('x_grad: \n', x_grad)
    print('x_grad.shape: \n', x_grad.shape)

    print('x_grad_numpy: \n', x_grad_numpy)
    print('x_grad_numpy.shape: \n', x_grad_numpy.shape)

    print('x_grad_error: \n', x_grad_numpy-x_grad.detach().numpy())


'''
    验证：激活层sigmoid（成功）
'''
def sigmoid_test():
    x_numpy = np.random.randn(6,1).astype(np.float32)
    x = torch.tensor(x_numpy, requires_grad=True)  

    y_target_numpy = np.array([0,0,1,0,1,1]).astype(np.float32) # target要是浮点数
    y_target = torch.tensor(y_target_numpy)  

    loss_tensor = nn.BCELoss()
    loss_numpy = Loss.BECLoss()
    """前向传播"""
    s_tensor = nn.Sigmoid()
    s_out_tensor = s_tensor(x).view(-1)
    
    s_numpy = Activators.Sigmoid_CE()
    s_out_numpy = s_numpy.forward(x_numpy).reshape(-1)

    print('-----对比输出-----')
    print('s_out_tensor: \n', s_out_tensor)
    print('s_out_tensor shape: \n', s_out_tensor.shape)
    print('s_out_numpy: \n', s_out_numpy)
    print('s_out_numpy shape: \n', s_out_numpy.shape)
    print('s_out_error: \n', s_out_numpy-s_out_tensor.detach().numpy())

    """反向传播"""
    err_tensor = loss_tensor(s_out_tensor, y_target) 
    err_numpy  = loss_numpy.forward(s_out_numpy, y_target_numpy)

    err_tensor.backward()
    x_grad = x.grad

    dy_loss = loss_numpy.gradient()
    x_grad_numpy = s_numpy.gradient(dy_loss)


    print('-----对比loss-----')
    print('err_tensor: \n', err_tensor)
    print('err_numpy: \n', err_numpy)
    print('err_error: \n', err_numpy-err_tensor.detach().numpy())

    print('-----对比x_grad-----')
    print('x_grad: \n', x_grad)
    print('x_grad_numpy: \n', x_grad_numpy)
    print('x_grad_error: \n', x_grad_numpy-x_grad.detach().numpy())


'''
    验证：激活层tanh（成功）
'''
def tanh_test():
    x_numpy = np.random.randn(6,1).astype(np.float32)
    x = torch.tensor(x_numpy, requires_grad=True)  

    """前向传播"""
    s_tensor = nn.Tanh()
    s_out_tensor = s_tensor(x)
    
    s_numpy = Activators.Tanh()
    s_out_numpy = s_numpy.forward(x_numpy)

    print('-----对比输出-----')
    print('s_out_tensor: \n', s_out_tensor)
    print('s_out_tensor shape: \n', s_out_tensor.shape)
    print('s_out_numpy: \n', s_out_numpy)
    print('s_out_numpy shape: \n', s_out_numpy.shape)
    print('s_out_error: \n', s_out_numpy-s_out_tensor.detach().numpy())

    """反向传播"""
    dy_numpy = np.random.random(s_out_numpy.shape).astype(np.float32)
    dy = torch.tensor(dy_numpy, requires_grad=True)

    s_out_tensor.backward(dy)
    x_grad = x.grad

    x_grad_numpy = s_numpy.gradient(dy_numpy)

    print('-----对比x_grad-----')
    print('x_grad: \n', x_grad)
    print('x_grad_numpy: \n', x_grad_numpy)
    print('x_grad_error: \n', x_grad_numpy-x_grad.detach().numpy())

'''
    验证：批量标准化层BN（成功）
'''
def bn_test():
    """定义输入"""
    x_numpy = np.random.randn(1,5,2,2).astype(np.float32)
    x = torch.tensor(x_numpy, requires_grad=True)
    """定义参数"""
    w_numpy = np.random.normal(1.0, 0.02, size=(5)).astype(np.float32)
    w = torch.tensor(w_numpy, requires_grad=True)
    b_numpy = np.zeros(5).astype(np.float32)
    b = torch.tensor(b_numpy, requires_grad=True)
    # 初始化
    # pytorch (需要添加affine=False参数)
    # affine定义了BN层的参数γ和β是否是可学习的(不可学习默认是常数1和0). 通常需要设置为True，但测试时numpy的γ=1和β=0，故此时需要将参数设置为False
    # bn_tensor = torch.nn.BatchNorm2d(5, affine=False)
    bn_tensor = torch.nn.BatchNorm2d(5, affine=True)
    bn_tensor.weight = torch.nn.Parameter(w, requires_grad=True)
    bn_tensor.bias = torch.nn.Parameter(b, requires_grad=True)
    # numpy
    bn_numpy = BN.BatchNorm(5)
    bn_numpy.set_gamma(Parameter(w_numpy, requires_grad=True))
    bn_numpy.set_beta(Parameter(b_numpy, requires_grad=True))
    """计算前向传播"""
    bn_out_tensor = bn_tensor(x)
    bn_out_numpy = bn_numpy.forward(x_numpy,'train')

    """计算反向传播"""
    # 误差参数初始化
    dy_numpy = np.random.random(bn_out_numpy.shape).astype(np.float32)
    dy = torch.tensor(dy_numpy, requires_grad=True)
    # 反向计算
    # pytorch
    bn_out_tensor.backward(dy)
    x_grad_tensor = x.grad
    w_grad_tensor = bn_tensor.weight.grad
    b_grad_tensor = bn_tensor.bias.grad

    # numpy
    x_grad_numpy = bn_numpy.gradient(dy_numpy)
    w_grad_numpy = bn_numpy.gamma.grad
    b_grad_numpy = bn_numpy.beta.grad
    
    
    """打印输出"""
    print('-----对比输出-----')
    print('bn_out_tensor: \n',bn_out_tensor)
    print('bn_out_tensor.shape: \n',bn_out_tensor.shape)
    print('bn_out_numpy: \n',bn_out_numpy)
    print('bn_out_numpy.shape: \n',bn_out_numpy.shape)
    print('bn_out_error: \n', bn_out_numpy-bn_out_tensor.detach().numpy())

    print('-----对比x_grad-----')
    print('x_grad_tensor: \n',x_grad_tensor)
    print('x_grad_numpy: \n',x_grad_numpy)
    print('x_grad_error: \n', x_grad_numpy-x_grad_tensor.detach().numpy())

    print('-----对比w_grad-----')
    print('w_grad_tensor: \n',w_grad_tensor)
    print('w_grad_numpy: \n',w_grad_numpy)
    print('w_grad_error: \n', w_grad_numpy-w_grad_tensor.detach().numpy())

    print('-----对比b_grad-----')
    print('b_grad_tensor: \n',b_grad_tensor)
    print('b_grad_numpy: \n',b_grad_numpy)
    print('b_grad_error: \n', b_grad_numpy-b_grad_tensor.detach().numpy())
    

'''
    验证：全连接层FC（成功）
'''
def fc_test():
    """定义输入样本"""
    # 输入数据大小为 1x5x8x8 输出大小定为10
    x_numpy = np.random.randn(2,5,8,8).astype(np.float32)
    x = torch.tensor(x_numpy, requires_grad=True)
    output_size = 10
    # 定义全连接层参数
    w_numpy = np.random.randn(5*8*8, output_size).astype(np.float32)
    b_numpy = np.random.randn(output_size,).astype(np.float32)
    w = torch.tensor(w_numpy.T, requires_grad=True)
    b = torch.tensor(b_numpy, requires_grad=True)

    
    # pytorch
    fc_tensor = torch.nn.Linear(320, output_size, bias=True)
    fc_tensor.weight = torch.nn.Parameter(w, requires_grad=True)
    fc_tensor.bias = torch.nn.Parameter(b, requires_grad=True)

    # numpy
    fc1 = FC.FullyConnect(320, output_size)
    fc1.set_weight(Parameter(w_numpy, requires_grad=True))
    fc1.set_bias(Parameter(b_numpy, requires_grad=True))
    
    """计算正向传播"""
    fc_out_tensor = fc_tensor(x.view(2,-1))
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
    w_grad_numpy = fc1.weights.grad
    b_grad_numpy = fc1.bias.grad

    # 打印结果
    print('-----对比输出-----')
    print('fc_out_tensor: \n',fc_out_tensor)
    print('fc_out_tensor: \n',fc_out_tensor.shape)
    print('fc_out_numpy: \n',fc_out_numpy)
    print('fc_out_numpy: \n',fc_out_numpy.shape)

    print('-----对比x_grad-----')
    print('x_grad: \n',x_grad)
    print('x_grad_shape: \n',x_grad.shape)
    print('x_grad_numpy: \n',x_grad_numpy)
    print('x_grad_numpy_shape: \n',x_grad_numpy.shape)

    print('x_grad error: \n',x_grad_numpy-x_grad.detach().numpy())

    print('-----对比w_grad-----')
    print('w_grad: \n',w_grad)
    print('w_grad shape: \n',w_grad.shape)
    print('w_grad_numpy: \n',w_grad_numpy)
    print('w_grad_numpy shape: \n',w_grad_numpy.shape)
    print('w_numpy shape: \n',fc1.weights.data.shape)
    
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
    pool1 = Pool.MaxPooling(pool_shape=(2,2), stride=(2,2))
    pool_out_numpy = pool1.forward(x_numpy)
    pool_eta = pool1.gradient(dy_numpy)
    print('pool_out_numpy: \n', pool_out_numpy)
    print('pool_out_numpy.shape: \n', pool_out_numpy.shape)

    # 反向传播误差对比
    print('pool_out_grad: \n', x.grad)
    print('pool_out_grad.shape: \n', x.grad.shape)

    print('pool_out_numpy_grad: \n', pool_eta)
    print('pool_out_numpy_grad.shape: \n', pool_eta.shape)

    print('pool_out_numpy_grad error: \n', pool_eta-x.grad.detach().numpy())


'''
    验证：反卷积层Deconv（成功）
'''
def deconv_test():
    """手动定义卷积核(weight)和偏置"""
    ## pytorch对于deconv的weight维度是[in_channel,out_channel,H,W]
    ## 通常Conv的维度是[out_channel,in_channel,H,W]
    # w_numpy = np.random.randn(3,5,4,4).astype(np.float32)
    w_numpy = np.random.randn(3,5,3,3).astype(np.float32)
    # w_numpy = np.ones((3,5,3,3)).astype(np.float32)
    b_numpy = np.random.randn(5).astype(np.float32)
    w = torch.tensor(w_numpy, requires_grad=True)
    b = torch.tensor(b_numpy, requires_grad=True)
    w_numpy = w_numpy.transpose((1,0,2,3))
    print('w_numpy.shape: ',w_numpy.shape)

    """定义输入样本"""
    # x_numpy = np.random.randn(1,3,4,4).astype(np.float32)
    # x_numpy = np.random.randn(1,3,2,2).astype(np.float32)
    x_numpy = np.ones((2,3,2,2)).astype(np.float32)
    x = torch.tensor(x_numpy, requires_grad=True)

    '''前向传播'''
    ## torch
    # decl_tensor = torch.nn.ConvTranspose2d(3, 5, kernel_size=4, stride=2, padding=1)
    decl_tensor = torch.nn.ConvTranspose2d(3, 5, kernel_size=3, stride=1, padding=0)
    decl_tensor.weight = torch.nn.Parameter(w, requires_grad=True)
    decl_tensor.bias = torch.nn.Parameter(b, requires_grad=True)
    deconv_out_tensor = decl_tensor(x)
    
    ## numpy
    # decl_numpy = Deconv.Deconv(3, 5, filter_size=4,  zero_padding=1, stride=2)
    decl_numpy = Deconv.Deconv(3, out_channels=5, filter_size=3,  zero_padding=0, stride=1)
    decl_numpy.set_weight(Parameter(w_numpy, requires_grad=True))
    decl_numpy.set_bias(Parameter(b_numpy, requires_grad=True))
    deconv_out_numpy = decl_numpy.forward(x_numpy)

    print('-----对比输出-----')
    print('deconv_out_tensor: \n', deconv_out_tensor[0])
    print('deconv_out_tensor.shape: \n', deconv_out_tensor.shape)

    print('deconv_out_numpy: \n', deconv_out_numpy[0])
    print('deconv_out_numpy.shape: \n', deconv_out_numpy.shape)

    print('deconv_out_error: \n', deconv_out_numpy-deconv_out_tensor.detach().numpy())


    '''反向传播'''
    dy_numpy = np.random.random(deconv_out_numpy.shape).astype(np.float32)
    dy = torch.tensor(dy_numpy, requires_grad=True)
    
    ## pytorch
    deconv_out_tensor.backward(dy)
    x_grad = x.grad
    w_grad = decl_tensor.weight.grad
    b_grad = decl_tensor.bias.grad

    ## numpy
    x_grad_numpy = decl_numpy.gradient(dy_numpy)
    w_grad_numpy = decl_numpy.weights.grad
    b_grad_numpy = decl_numpy.bias.grad

    print('-----对比x_grad-----')
    print('x_grad: \n', x_grad[0])
    print('x_grad.shape: \n', x_grad.shape)

    print('x_grad_numpy: \n', x_grad_numpy[0])
    print('x_grad_numpy.shape: \n', x_grad_numpy.shape)

    print('x_grad_error: \n', x_grad_numpy-x_grad.detach().numpy())

    print('-----对比w_grad-----')
    print('w_grad: \n', w_grad)
    print('w_grad_numpy: \n', w_grad_numpy)

    print('w_grad_error: \n', w_grad_numpy.transpose((1,0,2,3))-w_grad.detach().numpy())

    print('-----对比b_grad-----')
    print('b_grad: \n', b_grad)
    print('b_grad_numpy: \n', b_grad_numpy)



    
if __name__ == '__main__':
    # loss_test()
    # conv_test()
    # conv_checkgrad()
    # relu_test()
    # leakyrelu_test()
    # sigmoid_test()
    tanh_test()
    # bn_test()
    # fc_test()
    # pooling_test()
    # deconv_test()
    
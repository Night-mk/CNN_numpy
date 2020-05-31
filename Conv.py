'''
CNN.py 用于实现CNN卷积层计算，包括：
ConvLayer类，处理CNN前向传播，反向传播
Filter类，存储模型
'''
import numpy as np
import Activators
from Module import Module
from Parameter import Parameter

'''
通用函数：img2col
将图像在卷积窗口中的数拉成一行,每行k^2列,总共(out_h*out_w)行
[B,Cin,H,W]->[B,Cin*k*k,(H-k+1)*(W-k+1)]
'''
def img2col(input_array, filter_size, stride=1, zp=0):
    # input_array 4d tensor [batch, channel, height, width]
    output_matrix=[]
    width = input_array.shape[3]
    height = input_array.shape[2]
    # range的上限应该是(input_size - filter_size + 1)
    for i in range(0, height-filter_size+1, stride):
        for j in range(0, width-filter_size+1, stride):
            input_col = input_array[:, :, i:i+filter_size, j:j+filter_size].reshape([-1])
            # print('inputcol: \n', input_col.shape)
            output_matrix.append(input_col)
    output_matrix = np.array(output_matrix).T
    # print('output_matrix:', output_matrix.shape)
    # output_shape = [B,Cin*k*k,(H-k+1)*(W-k+1)] stride默认为1
    # output_matrix 2d tensor [height, width]
    # 输出之前需要转置
    return output_matrix

'''
通用函数: padding
填充方法["VALID"截取, "SAME"填充]
'''
def padding(input_array, method, zp):
    # "VALID"不填充
    if method=='VALID':
        return input_array
    # "SAME"填充
    elif method=='SAME':
        # (before_1, after_1)表示第1轴两边缘分别填充before_1个和after_1个数值
        input_array = np.pad(input_array, ((0, 0), (0, 0), (zp, zp), (zp, zp)), 'constant', constant_values=0)
        return input_array

'''
通用函数：element_wise_op
对numpy数组进行逐个元素的操作。op为函数。element_wise_op函数实现了对numpy数组进行按元素操作，并将返回值写回到数组中
'''
def element_wise_op(array, op):
    for i in np.nditer(array,op_flags=['readwrite']):
        i[...] = op(i)   # 将元素i传入op函数，返回值，再修改i

'''
    ConvLayer类，实现卷积层以及前向传播函数，反向传播函数
'''
class ConvLayer(Module):
    # 初始化卷积层函数
    # 参数包括：输入数据大小[batch大小、通道数、输入高度、输入宽度]，滤波器宽度、滤波器高度、滤波器数目、补零数目、步长、学习速率、补零方法
    def __init__(self, in_channels, out_channels, filter_width, filter_height, zero_padding, stride, method='VALID'):
        super(ConvLayer, self).__init__()
        # input_array 4d tensor [batch, channel, height, width]
        self.in_channels = in_channels
        # filter参数
        self.filter_width = filter_width  # 过滤器的宽度
        self.filter_height = filter_height  # 过滤器的高度
        self.out_channels = out_channels  # 过滤器组的数量（每组filter算一个）,输出通道数量
        self.zero_padding = zero_padding  # 补0圈数
        self.stride = stride # 步幅
        self.method = method

        # 卷积层过滤器初始化
        '''filter_num = output_channel,就是卷积输出feature map的通道数'''
        param_weights = np.random.uniform(-1e-2, 1e-2,(self.out_channels, self.in_channels, self.filter_height, self.filter_width))
        param_bias = np.zeros(self.out_channels)
        self.weights = Parameter(param_weights, requires_grad=True)
        self.bias = Parameter(param_bias, requires_grad=True)

    # 设置特定的权重和偏移量
    def set_weight(self, weight):
        if isinstance(weight, Parameter):
            self.weights = weight

    def set_bias(self, bias):
        if isinstance(bias, Parameter):
            self.bias = bias

    def kaiming_uniform(self, fan_in, shape, a=0):
        bound = np.sqrt(6/((1+a**2)*fan_in))
        return np.random.uniform(-bound, bound, shape)

    # 设置module打印格式
    def extra_repr(self):
        s = ('in_channels={in_channels}, out_channels={out_channels}, kernel_size={filter_width}'
             ', stride={stride}, padding={zero_padding}')
        if self.bias is None:
            s += ', bias=False'
        if self.method != None:
            s += ', method={method}'
        return s.format(**self.__dict__)

    # 静态方法计算卷积层输出尺寸大小=(W-F+2P)/S+1
    @staticmethod
    def compute_output_size(input_size, filter_size, zero_padding, stride):
        # 使用/会得到浮点数，使用//向下取整
        return (input_size-filter_size+2*zero_padding)//stride+1

    # 前向传播函数 img2col
    def forward(self, input_array):
        '''初始化输入、输出数据size'''
        self.input_shape = input_array.shape
        self.batchsize = self.input_shape[0]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        # 卷积层输出宽度计算 
        self.output_width = ConvLayer.compute_output_size(self.input_width, self.filter_width, self.zero_padding, self.stride)
        # 卷积层输出高度计算
        self.output_height = ConvLayer.compute_output_size(self.input_height, self.filter_height, self.zero_padding, self.stride)
        # 卷积层输出矩阵初始化 [batch, output_channel, height, width]
        self.output_array = np.zeros((self.batchsize ,self.out_channels, self.output_height, self.output_width))

        '''使用kaiming初始化'''
        fan_in = self.in_channels*self.input_height*self.input_width
        # fan_out = self.out_channels*self.output_height*self.output_width
        param_weights = self.kaiming_uniform(fan_in, shape=(self.out_channels, self.in_channels, self.filter_height, self.filter_width))
        self.weights = Parameter(param_weights, requires_grad=True)
        param_bias = self.kaiming_uniform(fan_in, shape=(self.out_channels))
        self.bias = Parameter(param_bias, requires_grad=True)

        '''计算卷积'''
        # 转换filter为矩阵, 将每个filter拉为一列, filter [Cout,depth,height,width]
        weights_col = self.weights.data.reshape([self.out_channels, -1])
        bias_col = self.bias.data.reshape([self.out_channels, -1])

        # padding方法计算填充关系
        input_pad = padding(input_array, self.method, self.zero_padding)
        # print('input_pad.shape: \n',input_pad.shape)
        # 将输入数据拉成矩阵（费时），反向传播时需要使用
        self.input_col = []
        conv_out = np.zeros(self.output_array.shape)
        # print('conv_out.shape: \n',conv_out.shape)
        
        # 对输入数据batch的每个图片、特征图进行卷积计算
        for i in range(0, self.batchsize):
            input_i = input_pad[i][np.newaxis,:] #获取每个batch的输入内容
            input_col_i = img2col(input_i, self.filter_width, self.stride, self.zero_padding) #将每个batch的输入拉为矩阵
            '''
                Kernel[Cout,Cin*k*k] dot X[Batch,Cin*k*k,(H-k+1)*(W-k+1)] = Y[Batch, Cout, (H-k+1)*(W-k+1)]
            '''
            # print('input_col_i.shape: \n',input_col_i.shape)
            # print('conv_result: \n',np.dot(weights_col, input_col_i))
            conv_out_i = np.dot(weights_col, input_col_i)+bias_col #计算矩阵卷积，输出大小为[Cout,(H-k+1)*(W-k+1)]的矩阵输出
            conv_out[i] = np.reshape(conv_out_i, self.output_array[0].shape) #转换为[Cout,Hout,Wout]的输出
            self.input_col.append(input_col_i) #记录输入数据的col形式，用于反向传播？？(暂时未知)
        self.input_col = np.array(self.input_col)
        
        return conv_out
        
    # 计算卷积梯度
    def gradient(self, eta):
        # eta表示上层（l+1层）向下层（l层）传输的误差
        # 即Z_ij, eta=[batch,Cout,out_h,out_w]
        self.eta = eta
        # print('eta.shape: \n', self.eta.shape)
        # eta_col=[batch,Cout,out_h*out_w]
        eta_col = np.reshape(eta, [self.batchsize, self.out_channels, -1])
        
        '''计算W的梯度矩阵 delta_W=a^(l-1) conv delta_Z^l'''
        for i in range(0, self.batchsize):
            # a^(l-1)=[batch,Cin*k*k,()*()]
            # input_col[i]=[Cin*k*k, out_h*out_w]
            # eta_col[i] = [Cout,out_h*out_w]
            # dot后的值=[Cout,Cin*k*k]
            self.weights.grad += np.dot(eta_col[i], self.input_col[i].T).reshape(self.weights.data.shape)
        '''计算b的梯度矩阵'''
        # print('eta_col: \n',eta_col)
        # print('eta.shape: \n',self.eta.shape)
        self.bias.grad += np.sum(eta_col, axis=(0, 2))
        
        """计算传输到上一层的误差"""
        ## 针对stride>=2时对误差矩阵的填充，需要在每个误差数据中间填充(stride-1) ##
        eta_pad = self.eta
        if self.stride>=2:
            # 计算中间填充后矩阵的size
            pad_size = (self.eta.shape[3]-1)*(self.stride-1)+self.eta.shape[3]
            eta_pad = np.zeros((self.eta.shape[0], self.eta.shape[1], pad_size, pad_size))
            for i in range(0, self.eta.shape[3]):
                for j in range(0, self.eta.shape[3]):
                    eta_pad[:,:,self.stride*i,self.stride*j] = self.eta[:,:,i,j]
        # print('eta: \n', self.eta[0,1])
        # print('eta_pad stride shape: \n', eta_pad.shape)

        # 使用输出误差填充零 conv rot180[weights]
        # 计算填充后的误差delta_Z_pad,即使用0在eta_pad四周填充,'VALID'填充数量为ksize-1，'SAME'填充数量为ksize/2
        if self.method=='VALID':
            eta_pad = np.pad(eta_pad, ((0,0),(0,0),(self.filter_height-1, self.filter_height-1),(self.filter_width-1, self.filter_width-1)),'constant',constant_values = (0,0))

        same_pad_height = (self.input_height-1+self.filter_height-eta_pad.shape[2])//2
        same_pad_width = (self.input_width-1+self.filter_width-eta_pad.shape[3])//2
        if self.method=='SAME':
            eta_pad = np.pad(eta_pad, ((0,0),(0,0),(same_pad_height, same_pad_height),(same_pad_width, same_pad_width)),'constant',constant_values = (0,0))
        
        # print('eta.shape: \n', self.eta.shape[0])
        # print('eta_pad SAME VALID shape: \n', eta_pad.shape)

        ## 计算旋转180度的权重矩阵，rot180(W)
        # self.weight[Cout,depth,h,w]
        # A[::-1]对于行向量可以左右翻转；对于二维矩阵可以实现上下翻转
        # 对于4维数据的h,w翻转
        flip_weights = self.weights.data[...,::-1,::-1]
        # print('flip_weights.shape: \n',flip_weights.shape)
        # 参数矩阵需要维度转换 W[Cout,Cin,h,w]->W[Cin,Cout,h,w]这样变化为col矩阵的时候，可以直接使用reshape(channel_num, -1)获得对应的矩阵大小，并完成和填充误差进行卷积计算获得L-1层的误差
        flip_weights = flip_weights.swapaxes(0, 1)
        flip_weights_col = flip_weights.reshape([self.in_channels, -1])
        eta_pad_col = []
        for i in range(0, self.batchsize):
            eta_pad_col_i = img2col(eta_pad[i][np.newaxis,:], self.filter_width, 1, self.zero_padding)
            # print('eta_pad_col_i.shape: \n', eta_pad_col_i.shape)
            eta_pad_col.append(eta_pad_col_i)
        eta_pad_col = np.array(eta_pad_col)
        # print('flip_weights_col.shape: \n',flip_weights_col.shape)
        # print('eta_pad_col.shape: \n',eta_pad_col.shape)

        ## 计算向上一层传播的误差eta_next,采用卷积乘计算
        # 原本，delta_Z^(l)=delta_Z^(l+1) conv rot180(W^(l))
        # 这里没有像前向计算里的那种batchsize的概念了，batch刚好把行向量补齐
        eta_next = np.dot(flip_weights_col, eta_pad_col)
        # print('eta_next.shape:\n', eta_next.shape)
        # input_shape就是上一层的output_shape
        eta_next = np.reshape(eta_next, self.input_shape)

        self.eta_next = eta_next
        return eta_next

def unit_test():
    print("------test-------")
    '''
    # img2col
    a = np.arange(36).reshape(2, 2, 3, 3)
    img_col = img2col(a, 2, 1)
    print(a)
    print(img_col)
    print(img_col.shape)
    '''
    a = np.arange(36).reshape(2, 2, 3, 3)
    print(a)
    a1 = a[...,::-1,::-1]
    print(a1)
    print(a1.shape)


def cnn_forward_test():
    print('-------forward_test-------')
    # arange生成的是浮点数序列
    input_img = np.arange(27).reshape(1,3,3,3)
    cl1 = ConvLayer(3, 5, 2,2, zero_padding=0, stride=1, method='VALID')
    print('input_img', input_img)
    # forward
    conv_out = cl1.forward(input_img)
    print(conv_out)
    print('-----shape-----', conv_out.shape)


def cnn_backward_test():
    input_img = np.arange(27).reshape(1,3,3,3) # input
    cl1 = ConvLayer(3, 5,2,2, zero_padding=0, stride=1, method='VALID') # convlayer
    conv_out = cl1.forward(input_img) # forward calculation

    # 假设误差为1
    conv_out1 = conv_out.copy()+1
    eta_next = cl1.gradient(conv_out1-conv_out) # gradient calculation

    print('eta_next: \n', eta_next)
    print('cl1.weight_grad: \n',cl1.weights.grad)
    print('cl1.bias_grad: \n',cl1.bias.grad)
    
if __name__ == '__main__':
    # unit_test()
    # cnn_forward_test()
    cnn_backward_test()
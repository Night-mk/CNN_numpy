'''
    Deconv.py 用于实现转置卷积（反卷积）计算，可以使用在GAN网络中
'''
import numpy as np
import Conv
from Module import Module
from Parameter import Parameter

"""根据stride填充矩阵"""
def padding_stride(input_array, stride):
    pad_size = (input_array.shape[3]-1)*(stride-1)+input_array.shape[3]
    input_array_pad = np.zeros((input_array.shape[0], input_array.shape[1], pad_size, pad_size))
    for i in range(0, input_array.shape[3]):
        for j in range(0, input_array.shape[3]):
            input_array_pad[:,:, stride*i, stride*j] = input_array[:,:, i, j]
    return input_array_pad

"""额外填充矩阵（左边，上边各填充一行0）"""
def padding_additional(input_array):
    input_array_pad = np.pad(input_array, ((0,0), (0,0), (1,0), (1,0)), 'constant', constant_values=0)
    return input_array_pad


"""转置卷积类Deconv"""
class Deconv(Module):
    def __init__(self, in_channels, out_channels, filter_size, zero_padding, stride, method='SAME'):
        super(Deconv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.zero_padding = zero_padding
        self.stride = stride
        self.method = method
        
        # 定义参数
        param_weights = np.random.uniform(-1e-2, 1e-2, (self.out_channels, self.in_channels, self.filter_size, self.filter_size))
        param_bias = np.zeros(self.out_channels)
        self.weights = Parameter(param_weights, requires_grad=True)
        self.bias = Parameter(param_bias, requires_grad=True)

    # 静态方法计算反卷积层输出尺寸大小 
    # O=(W-1)*S-2P+F [(O-F+2P)%S==0]
    # O=(W-1)*S-2P+F+(O-F+2P)%S [(O-F+2P)%S!=0]
    @staticmethod
    def compute_output_size(input_size, filter_size, zero_padding, stride):
        output_size = (input_size-1)*stride-2*zero_padding+filter_size
        residue = (output_size-filter_size+2*zero_padding)%stride

        if residue==0: return output_size
        else: return output_size+residue

    # 设置特定的权重和偏移量
    def set_weight(self, weight):
        if isinstance(weight, Parameter):
            self.weights = weight

    def set_bias(self, bias):
        if isinstance(bias, Parameter):
            self.bias = bias

    # 设置module打印格式
    def extra_repr(self):
        s = ('in_channels={in_channels}, out_channels={out_channels}, kernel_size={filter_size}'
             ', stride={stride}, padding={zero_padding}')
        if self.bias is None:
            s += ', bias=False'
        if self.method != None:
            s += ', method={method}'
        return s.format(**self.__dict__)

    # 前向传播计算
    # 需要填充输入矩阵，计算填充大小，并执行卷积计算
    def forward(self, input_array):
        self.input_array = input_array
        self.input_shape = self.input_array.shape # [B,C,H,W]
        self.batchsize = self.input_shape[0]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        # 计算反卷积输出大小
        self.output_size = Deconv.compute_output_size(self.input_width, self.filter_size, self.zero_padding, self.stride)

        self.output_array = np.zeros((self.batchsize, self.out_channels, self.output_size, self.output_size))  

        '''反卷积参数填充：需要对input进行两次填充'''
        # 第一次，根据stride在input内部填充0，每个元素间填充0的个数为n=stride-1
        input_pad = input_array
        if self.stride>=2:
            input_pad = padding_stride(input_pad, self.stride)
        # 第二次填充，根据输出大小，计算以stride=1的卷积计算的输入需要的padding，如果padding%2不为0，则优先在input的左侧和上侧填充0【2P=(O-1)*1+F-W】
        input_pad = Conv.padding(input_pad, 'SAME', self.zero_padding) # 必要填充

        '''卷积计算填充：需要对input再次填充'''
        padding_num_2 = self.output_size-1+self.filter_size-input_pad.shape[3]
        input_pad = Conv.padding(input_pad, 'SAME', padding_num_2//2) # 必要填充
        if padding_num_2%2!=0: # 在input的左侧和上侧填充0
            input_pad = padding_additional(input_pad)

        '''转换filter为矩阵'''
        flip_weights = self.weights.data[...,::-1,::-1]
        weights_col = flip_weights.reshape([self.out_channels, -1])
        bias_col = self.bias.data.reshape([self.out_channels, -1])

        # print('input_pad.shape: \n',input_pad.shape)
        # print('input_pad: \n', input_pad)

        '''计算反卷积前向传播'''
        self.input_col = []
        deconv_out = np.zeros(self.output_array.shape)
        for i in range(0, self.batchsize):
            input_i = input_pad[i][np.newaxis,:] #获取每个batch的输入内容
            input_col_i = Conv.img2col(input_i, self.filter_size, 1, self.zero_padding) #将每个batch的输入拉为矩阵(注意此处的stride=1)
            # print('input_col_i.shape: \n',input_col_i.shape)
            deconv_out_i = np.dot(weights_col, input_col_i)+bias_col #计算矩阵卷积，输出大小为[Cout,(H-k+1)*(W-k+1)]的矩阵输出
            # print('deconv_out_i.shape: \n',deconv_out_i.shape)
            deconv_out[i] = np.reshape(deconv_out_i, self.output_array[0].shape) #转换为[Cout,Hout,Wout]的输出
            self.input_col.append(input_col_i) 
        self.input_col = np.array(self.input_col)

        return deconv_out

    # 计算w,b梯度，并计算向上一层传输的误差
    def gradient(self, eta):
        self.eta = eta # eta=[batch,out_c,out_h,out_w]
        # print('eta.shape: \n', eta.shape)
        eta_col = np.reshape(eta, [self.batchsize, self.out_channels, -1])
        
        '''计算weight，bias的梯度'''
        for i in range(0, self.batchsize):
            self.weights.grad += np.dot(eta_col[i], self.input_col[i].T).reshape(self.weights.data.shape)
        self.bias.grad += np.sum(eta_col, axis=(0, 2))
        # print('weight_grad.shape: \n', self.weights.grad.shape)

        '''计算向上一层传播的误差eta_next'''
        eta_pad = self.eta
        eta_pad = Conv.padding(eta_pad, self.method, self.zero_padding)
        # print('eta_pad.shape: \n', eta_pad.shape)

        # 为啥还要再rot180一次= =
        # flip_weights = self.weights[...,::-1,::-1]
        flip_weights = self.weights.data.swapaxes(0, 1)
        flip_weights_col = flip_weights.reshape([self.in_channels, -1])
        eta_pad_col = []
        for i in range(0, self.batchsize):
            # print('eta_pad[i].shape: \n', eta_pad[i].shape)
            eta_pad_col_i = Conv.img2col(eta_pad[i][np.newaxis,:], self.filter_size, self.stride, self.zero_padding)
            # print('eta_pad_col_i.shape: \n', eta_pad_col_i.shape)
            eta_pad_col.append(eta_pad_col_i)
        eta_pad_col = np.array(eta_pad_col)

        eta_next = np.dot(flip_weights_col, eta_pad_col)
        # print('eta_next.shape:\n', eta_next.shape)
        # input_shape就是上一层的output_shape
        eta_next = np.reshape(eta_next, self.input_shape)
        self.eta_next = eta_next
        
        return eta_next

def deconv_forward_test():
    print('-------forward_test-------')
    # arange生成的是浮点数序列
    input_img = np.arange(48).reshape(1,3,4,4)
    # input_img = np.arange(192).reshape(1,3,8,8)
    de_cl1 = Deconv(in_channels=3, out_channels=3, filter_size=4,  zero_padding=1, stride=2, method='SAME')
    print('input_img', input_img)
    # forward
    deconv_out = de_cl1.forward(input_img)
    print(deconv_out)
    print('-----shape-----', deconv_out.shape)

    dy_numpy = np.random.random(deconv_out.shape).astype(np.float32)
    x_grad_numpy = de_cl1.gradient(dy_numpy)

    print('x_grad_numpy: \n', x_grad_numpy)
    print('-----shape-----', x_grad_numpy.shape)

def deconv_backward_test():
    return 0

if __name__ == "__main__":
    deconv_forward_test()
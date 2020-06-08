'''
    DCGAN_numpy.py 使用numpy实现DCGAN网络
'''

import random
import numpy as np
from Module import Module
from Optimizer_func import Adam
from Parameter import Parameter
from Conv import ConvLayer
from Deconv import Deconv
from Loss import BECLoss
import Activators
from BN import BatchNorm

import torch
from torchvision import datasets,transforms
import os
import time

import matplotlib.pyplot as plt
import torchvision.utils as vutils # 暂时不知道干嘛的（处理图像用的？）

# batch_size = 128 # 训练batch
batch_size = 128# 训练batch
image_size = 28 # 训练图像size，默认64x64, MINST 28x28
nc = 1 # 输入图像通道数
nz = 100 # 生成器输入的噪声z维度, lantent vector size

ngf = 64 # 生成器特征图的数量
ndf = 64 # 判别器特征图的数量
num_epochs = 1 # 训练轮数
lr = 0.0002 # Adam优化器的学习率
beta1 = 0.5 # Adam优化器的参数
is_mnist = True
num_epochs_pre = 0 # 预训练轮数

"""判别器D网络"""
class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 输入1*28*28 MNIST
        # 1*28*28 -> 64*16*16
        self.conv1 = ConvLayer(nc, ndf, 4,4, zero_padding=1, stride=2,method='SAME', bias_required=False)
        self.lrelu1 = Activators.LeakyReLU(0.2)

        # 64*16*16 -> 128*8*8
        self.conv2 = ConvLayer(ndf, ndf*2, 4,4, zero_padding=1, stride=2, method='SAME', bias_required=False)
        self.bn1 = BatchNorm(ndf*2)
        self.lrelu2 = Activators.LeakyReLU(0.2)

        # 128*8*8 -> 256*4*4
        self.conv3 = ConvLayer(ndf*2, ndf*4, 4,4, zero_padding=1, stride=2, method='SAME', bias_required=False)
        self.bn2 = BatchNorm(ndf*4)
        self.lrelu3 = Activators.LeakyReLU(0.2)

        # 256*4*4 -> 1*1
        self.conv4 = ConvLayer(ndf*4, 1, 4,4, zero_padding=0, stride=1, method='VALID', bias_required=False)
        self.sigmoid = Activators.Sigmoid_CE()

    def forward(self, x_input):
        l1 = self.lrelu1.forward(self.conv1.forward(x_input))

        l2 = self.lrelu2.forward(self.bn1.forward(self.conv2.forward(l1)))

        l3 = self.lrelu3.forward(self.bn2.forward(self.conv3.forward(l2)))
        
        l4 = self.conv4.forward(l3)
        # print('D l1 shape: ',l1.shape)
        # print('D l2 shape: ',l2.shape)
        # print('D l3 shape: ',l3.shape)
        # print('D l4 shape: ',l4.shape)
        output_sigmoid = self.sigmoid.forward(l4)
        return output_sigmoid
    
    def backward(self, dy):
        # print('dy.shape: ', dy.shape)
        dy_sigmoid = self.sigmoid.gradient(dy)
        # print('dy_sigmoid.shape: ', dy_sigmoid.shape)
        dy_l4 = self.conv4.gradient(dy_sigmoid)
        dy_l3 = self.conv3.gradient(self.bn2.gradient(self.lrelu3.gradient(dy_l4)))
        dy_l2 = self.conv2.gradient(self.bn1.gradient(self.lrelu2.gradient(dy_l3)))
        self.conv1.gradient(self.lrelu1.gradient(dy_l2))


class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 构建反向传播网络组建
        # 输入Z=[100,]
        # 100*1 -> 512*4*4
        self.deconv1 = Deconv(nz, ngf*4, 4, zero_padding=0, stride=1, method='VALID', bias_required=False)
        self.bn1 = BatchNorm(ngf*4)
        self.relu1 = Activators.ReLU()

        self.deconv2 = Deconv(ngf*4, ngf*2, 4, zero_padding=1, stride=2, method='SAME', bias_required=False)
        self.bn2 = BatchNorm(ngf*2)
        self.relu2 = Activators.ReLU()

        self.deconv3 = Deconv(ngf*2, ngf, 4, zero_padding=1, stride=2, method='SAME', bias_required=False)
        self.bn3 = BatchNorm(ngf)
        self.relu3 = Activators.ReLU()

        self.deconv4 = Deconv(ngf, nc, 4, zero_padding=1, stride=2, method='SAME', bias_required=False)
        self.tanh = Activators.Tanh_CE()

    def forward(self, x_input):
        # print('G input shape: ',x_input.shape)
        l1 = self.relu1.forward(self.bn1.forward(self.deconv1.forward(x_input)))
        l2 = self.relu2.forward(self.bn2.forward(self.deconv2.forward(l1)))
        l3 = self.relu3.forward(self.bn3.forward(self.deconv3.forward(l2)))
        l4 = self.deconv4.forward(l3)

        # print('G l1 shape: ',l1.shape)
        # print('G l2 shape: ',l2.shape)
        # print('G l3 shape: ',l3.shape)
        # print('G l4 shape: ',l4.shape)
        output_tanh = self.tanh.forward(l4)

        return output_tanh

    def backward(self, dy):
        dy_tanh = self.tanh.gradient(dy)
        dy_l4 = self.deconv4.gradient(dy_tanh)
        dy_l3 = self.deconv3.gradient(self.bn3.gradient(self.relu3.gradient(dy_l4)))
        dy_l2 = self.deconv2.gradient(self.bn2.gradient(self.relu2.gradient(dy_l3)))
        self.deconv1.gradient(self.bn1.gradient(self.relu1.gradient(dy_l2)))


'''批量初始化网络的权重'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # 卷积层和反卷积层设置没有bias参数
        # nn.init.normal_(m.weights.data, 0.0, 0.02)
        m.weights.data = np.random.normal(0.0,0.02,size=m.weights.data.shape)
    elif classname.find('BatchNorm') != -1:
        # BN层初始化两组参数，weight=gamma，bias=beta
        # nn.init.normal_(m.weight.data, 1.0, 0.02)
        # nn.init.constant_(m.bias.data, 0)
        m.gamma.data = np.random.normal(1.0, 0.02, size=m.gamma.data.shape)
        m.beta.data = np.zeros(m.beta.data.shape)
        
def test_dcgan():
    start_time = time.time()
    """加载MNIST数据集"""
    mnist_train_dataset = datasets.MNIST('./data/',download=True,train=True,transform=transforms.Compose([
                                   transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                               ]))
    print('traindata_len: \n',len(mnist_train_dataset))
    # 构建数据集迭代器
    mnist_train_loader = torch.utils.data.DataLoader(mnist_train_dataset,batch_size=batch_size,shuffle=True)
    
    """初始化网络、参数"""
    netG = Generator()
    netG.apply(weights_init)
    print('netG: \n', netG)
    
    netD = Discriminator()
    netD.apply(weights_init)
    print('netD: \n', netD)


    """构建优化器"""
    # 二进制交叉熵损失函数
    loss = BECLoss()
    # 噪声从标准正态分布（均值为0，方差为 1，即高斯白噪声）中随机抽取一组数
    fixed_noise = np.random.normal(0.0, 1.2, size=(64,nz, 1,1)) # 用于G生成图像时固定的噪声初始化
    # 定义真假样本标签
    real_label = 1
    fake_label = 0
    # 定义Adam优化器
    optimizerD = Adam(netD.parameters(), learning_rate=lr, betas=(beta1, 0.999))
    optimizerG = Adam(netG.parameters(), learning_rate=lr, betas=(beta1, 0.999))

    """训练模型，生成数据"""
    # 存储生成的图像
    img_list = []
    # 记录G和D的损失
    G_losses = []
    D_losses = []
    iters = 0

    """加载预训练的模型"""
    '''
    pre_module_path = "./model_save/lenet_numpy_parameters-Adam-1.pkl"
    params = torch.load(pre_module_path)
    netD.load_state_dict(params['D_state_dict']) # 加载模型
    netG.load_state_dict(params['G_state_dict']) # 加载模型
    num_epochs_pre = params['epoch']
    '''
    
    print("----------start training loop----------")

    for epoch in range(num_epochs):
        # dataloader获取真实图像
        for t, (data, target) in enumerate(mnist_train_loader, 0):
            '''
                (1)先更新D Update D network: minimize -[ log(D(x)) + log(1 - D(G(z))) ]
                训练D的目标是让D更加有能力判断真假数据
            '''
            ## 使用真实数据X进行训练（计算log(D(x))）
            netD.zero_grad() # 训练更新前需要在每个batch中将梯度设置为0
            real_data = data.detach().numpy()
            ## MNIST 数据需要先从1x28x28填充到1x32x32
            if is_mnist:
                real_data = np.pad(real_data, ((0, 0), (0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
            b_size = real_data.shape[0]
            label = np.full((b_size,), real_label)
            # 计算D前向传播值
            output_d_real = netD.forward(real_data).reshape(-1)
            # 计算D真实数据交叉熵损失
            errD_real = loss.forward(output_d_real, label)
            # 计算D的梯度
            dy_errD_real = loss.gradient()
            netD.backward(dy_errD_real)
            
            ## 使用生成数据进行训练（计算log(1 - D(G(z)))）
            noise = np.random.normal(0.0, 1.2, size=(b_size, nz, 1,1)) # 训练每次单独生成噪声
            # G生成假数据
            fake_data = netG.forward(noise)
            label.fill(fake_label)
            # D识别假数据
            output_d_fake = netD.forward(fake_data).reshape(-1)
            # 计算D假数据交叉熵损失
            errD_fake = loss.forward(output_d_fake, label)
            # 计算D的梯度
            dy_errD_fake = loss.gradient()
            netD.backward(dy_errD_fake)

            # 计算总损失
            errD = errD_real+errD_fake

            # 计算D(x),D(G(z))的均值
            D_x = np.mean(output_d_real)
            D_G_z1 = np.mean(output_d_fake)

            # 更新D参数
            optimizerD.step()

            '''
                (2)更新G Update G network: minimize -log(D(G(z)))
            '''
            netG.zero_grad()
            # 填充真实标签，使得交叉熵函数可以只计算log(D(G(z))部分
            label.fill(real_label)
            output_d_fake = netD.forward(fake_data).reshape(-1)
            errG = loss.forward(output_d_fake, label)
            # 计算G的梯度
            dy_errG = loss.gradient()
            netG.backward(dy_errG)
            # 计算D(G(z))的均值
            D_G_z2 = np.mean(output_d_fake)
            # 更新G参数（不会去计算D的梯度2333）
            optimizerG.step()

            """输出训练状态"""
            # Loss_D
            # Loss_G
            # D(x)：训练中D对真实数据的平均预测输出
            # D(G(z))：训练中D对虚假数据的平均预测输出（为啥是除法？？）
            if t % 10 == 0:
                print('[%d/%d][%d/%d]\t Loss_D: %.4f\t Loss_G: %.4f\t D(x): %.4f\t D(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, t, len(mnist_train_loader), errD, errG, D_x, D_G_z1, D_G_z2))

            # 记录损失的历史，可以用作画图
            G_losses.append(errG)
            D_losses.append(errD)

            # 记录G生成的图像
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (t == len(mnist_train_loader)-1)):
                fake_img = netG.forward(fixed_noise)# 一次生成64张图
                fake_tensor = torch.tensor(fake_img)
                img_list.append(vutils.make_grid(fake_tensor, padding=2, normalize=True))
            
            iters += 1
    """绘图：记录损失"""
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    
    # 保存图片
    time_stemp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    plt.savefig('./experiment_img/gan_generate/Loss_fig-Adam'+str(num_epochs)+'('+time_stemp+').png')
    # plt.show()

    """绘图：记录G输出"""
    real_batch = next(iter(mnist_train_loader))
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))

    # 保存图片
    plt.savefig('./experiment_img/gan_generate/Real_Generate-Adam-'+str(num_epochs)+'('+time_stemp+').png')
    # plt.show()
    end_time = time.time()
    print('training time: \n', (end_time-start_time)/60)

    '''存储模型'''
    checkpoint_path = "./model_save/DCGAN_numpy_parameters-Adam"+str(num_epochs+num_epochs_pre)+".pkl"
    torch.save({'epoch':num_epochs+num_epochs_pre, 'D_state_dict':netD.state_dict(), 'G_state_dict':netG.state_dict()}, checkpoint_path)


if __name__ == "__main__":
    test_dcgan()

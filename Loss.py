'''
Loss.py用于实现损失函数
主要实现交叉熵Cross Entropy
'''
import numpy as np
from Module import Module

class NLLLoss(Module):
    # 计算多分类任务中的负对数似然损失函数，传入logsoftmax([p1,p2,...,pk])
    def __init__(self, size_average=True):
        super(NLLLoss, self).__init__()
        self.size_average = size_average

    # 计算损失函数，先计算softmax值，再使用cross entropy计算损失
    def cal_loss(self, prediction, labels):
        '''
            predict：output of predicted probability [batch, [p1,p2,...,pk]]
            labels: labels of dataset [batch, 1] , example: [2,3,8,9]表示类别
            labels可能有one-hot编码模式，[0,0,1,0]代表3
            size_average：if the loss need to be averaged
        '''
        self.labels = labels
        self.prediction = prediction
        self.batchsize = self.prediction.shape[0]
        self.loss = 0
        # 判断是否使用one-hot编码
        if labels.ndim >1: # one-hot [[p1,p2,...,pk],...]
            for i in range(self.batchsize):
                self.loss -= np.sum(self.prediction * self.labels)
        elif labels.ndim == 1: # [class_num]
            for i in range(self.batchsize):
                self.loss -= prediction[i, labels[i]]
        # 对所有样本的loss求平均，作为最终的loss输出
        if self.size_average:
            self.loss = self.loss/self.batchsize
        return self.loss

    def gradient(self):
        self.eta = self.labels.copy()
        # 求导结果为-yi
        self.eta_next = -self.eta
        return self.eta_next

def test_NLLLoss():
    print('-----NLLLoss test-----')

if __name__ == '__main__':
    test_NLLLoss()
'''
Loss.py用于实现损失函数
主要实现Cross Entropy和softmax
'''
import numpy as np

class CrossEntropy(object):
    def __init__(self, input_shape):
        # 交叉熵损失函数的输入本身就是预测每个标签的概率
        # input_shape=[batch, label_num]
        self.input_shape = input_shape
        self.batchsize = input_shape[0]
        self.cross_entropy = np.zeros(self.input_shape)

class Softmax(object):
    def __init__(self, input_shape):
        # input_shape=[batch, class_num]
        self.input_shape = input_shape
        self.batchsize = input_shape[0]
        self.eta = np.zeros(self.input_shape)

    def predict(self, prediction):
        # prediction 可以表示从FC层输出的数据 [batch, class_num]
        self.softmax = np.zeros(prediction.shape)
        # 对每个batch的数据求softmax
        for i in range(self.batchsize):
            exps_i = np.exp(prediction[i]-np.max(prediction[i]))
            self.softmax[i] = exps_i/np.sum(exps_i)
        # softmax[batch, class_num]
        return self.softmax

    # 计算损失函数，先计算softmax值，再使用cross entropy计算损失
    def cal_loss(self, prediction, labels, size_average=True):
        '''
            predict：output of predicted probability [batch, class_num]
            labels: labels of dataset [batch, 1] , example: [2,3,8,9]表示类别
            labels可能有one-hot编码模式，[0,0,1,0]代表3
            size_average：if the loss need to be averaged
        '''
        self.labels = labels
        self.prediction = prediction
        # 用于计算self.softmax的结果，计算损失的时候需要使用
        predict_softmax = self.predict(prediction)
        self.loss = 0
        # 使用softmax结果进行交叉熵计算
        for i in range(self.batchsize):
            self.loss -= np.log(predict_softmax[i, labels[i]])
        # 另一种计算方式，结合softmax和cross entropy计算公式，减少loss计算量
        # for i in range(self.batchsize):
        #     self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i][labels[i]]
        # 对所有样本的loss求平均，作为最终的loss输出
        if size_average:
            self.loss = self.loss/self.batchsize
        return self.loss

    def gradient_with_loss(self):
        self.eta = self.softmax.copy()
        # 采用cross entropy 和 softmax后的梯度公式
        # delta_oi = pi-yi
        # y 代表标签的 One-hot 编码
        for i in range(self.batchsize):
            self.eta[i, self.labels[i]] -= 1
        # eta[batchsize, class_num]
        # 需要除以batchsize用于平均该批次的影响
        self.eta = self.eta/self.batchsize
        return self.eta

def test_softmax():
    print('-----softmax test-----')
    a = np.arange(4).reshape(2,2)
    softmax_ob = Softmax(a.shape)

if __name__ == '__main__':
    test_softmax()
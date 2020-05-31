'''
    Optimizer_func.py 实现几种优化算法(继承自Optimizer类)
    SGD
    Adam
'''

from Optimizer import Optimizer
import numpy as np


"""SGD(Stochastic Gradient Descent) 随机梯度下降"""
class SGD(Optimizer):
    # 目前先只实现基础功能，和momentum版本
    def __init__(self, params, learning_rate, momentum=0):
        if learning_rate < 0.0:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))

        # 构建参数字典，加入到Optimizer类的参数列表中
        defaults = dict(learning_rate=learning_rate, momentum=momentum)
        # 将参数传递给父类Optimizer，可以在父类中统一初始化
        super(SGD, self).__init__(params, defaults)
    
    '''训练，更新参数'''
    # 重写父类函数，每个优化函数有自己的方法
    def step(self, closure=None):
        # 对params中每层的数据进行更新
        loss = None # loss目前没啥用
        if closure is not None:
            loss = closure()
        # 遍历参数集合
        for group in self.param_groups:
            momentum = group['momentum']
            learning_rate = group['learning_rate']
            # print('learning rate: ',learning_rate)
            # 遍历每组参数
            for p in group['params']:
                # print('optim SGD param_groups id: ', id(p))
                if p.grad is None:
                    continue
                d_p = p.grad
                '''梯度修正？？看看有没有用= ='''
                # d_p[np.abs(d_p)<1e-10]=0
                # d_p[np.abs(d_p) > 100]=0
                # d_p[np.abs(d_p) < -100]=-100

                buf = learning_rate*d_p # V_t = lr*d_p
                # 处理momentum的计算
                if momentum!=0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state: # 第一次迭代，没有历史momentum的时候
                        buf = buf
                    else:
                        buf = momentum*param_state['momentum_buffer'] + buf # V_t = V_(t-1)*momentum + lr*d_p
                    param_state['momentum_buffer'] = buf # 更新父类的参数没有使用in-place函数，所以需要重新给momentum_buffer赋值
                    # print('error: \n', d_p-param_state['momentum_buffer'])
                
                p_new = p.data - buf # weight = weight - V_t 更新参数
                p.set_param(p_new)
                
                '''
                # pytorch 方案 p=p-lr*(v_[t-1]*m + grad*(1-dampening=0))
                if momentum!=0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = d_p.copy()
                    else:
                        buf = param_state['momentum_buffer']
                        buf = buf*momentum+d_p
                    param_state['momentum_buffer'] = buf
                    d_p = buf

                p.data -= learning_rate*d_p
                '''

        return loss 
                    

"""Adam(Stochastic Gradient Descent) 随机梯度下降"""
class Adam(Optimizer):
    def __init__(self, params, learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        if learning_rate < 0.0:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        # 构建参数字典，加入到Optimizer类的参数列表中
        defaults = dict(learning_rate=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        '''
            m = beta1*m + (1‐beta1)*dx
            v = beta2*v + (1‐beta2)*(dx**2)
            x += ‐ learning_rate * m / (np.sqrt(v) + eps)
        '''
        loss = None # loss目前没啥用
        if closure is not None:
            loss = closure()
        # 计算Adam
        for group in self.param_groups:
            learning_rate = group['learning_rate']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0 # 计算偏差校正
                    state['exp_avg'] = np.zeros(p.data.shape)
                    state['exp_avg_sq'] = np.zeros(p.data.shape)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # 由于mt 和 vt 被初始化为 0 向量，那它们就会向 0 偏置，所以做了偏差校正，通过计算偏差校正后的 mt 和 vt 来抵消这些偏差。
                # mt_new = mt/(1-beta1**t)
                # vt_new = vt/(1-beta2**t)
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if weight_decay != 0:
                    grad += weight_decay*p.data
                
                # 计算动量
                exp_avg = beta1*exp_avg+(1-beta1)*grad
                exp_avg_sq = beta2*exp_avg_sq+(1-beta2)*(grad**2)
                # 计算校正后的值，赋给state
                exp_avg_correction = exp_avg/bias_correction1
                exp_avg_sq_correction = exp_avg_sq/bias_correction2
                state['exp_avg'] = exp_avg_correction
                state['exp_avg_sq'] = exp_avg_sq_correction

                # 计算更新
                p_new = p - learning_rate*exp_avg_correction/((exp_avg_sq_correction.sqrt)+eps)
                
                p.set_param(p_new)
        
        return loss






        
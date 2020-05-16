'''
    Parameter.py 用于实现参数类Parameter，分离地管理网络中的参数
'''

import numpy as np

class Parameter(object):
    def __init__(self, data=None, requires_grad=True):
        self.data = data
        if data is None:
            self.data = np.array([])
        # requires_grad
        self.requires_grad = requires_grad
        # grad
        self.grad = None
        if requires_grad:
            self.grad = np.zeros(self.data.shape)

    # __repr__将实例对象按照自定义的格式用字符串的形式显示出来，提高可读性。
    def __repr__(self):
        return 'Parameter containing:\n%s' % (repr(self.data))

    def set_param(self, data):
        self.data = data
    
    def set_grad(self, grad):
        self.grad = grad
    
    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros(self.data.shape)


# class Parameter(np.ndarray):
#     # 真构造函数，__new__
#     # __new__方法用于创建对象并返回对象，当返回对象时会自动调用__init__方法进行初始化。__new__方法是静态方法，而__init__是实例方法
#     # cls指当前正在实例化的类
#     # 使用__new__的意义在于，我们想让Parameter类返回ndarray（其他）类对象
#     def __new__(cls, data=None, requires_grad=False):
#         if data is None:
#             data = np.array([])
#         obj = np.asarray(data).view(cls)
#         return np.array(data, )
        # return torch.Tensor._make_subclass(cls, data, requires_grad)


if __name__ == '__main__':
    data = np.array([])
    print(data)
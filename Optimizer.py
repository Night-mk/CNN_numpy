'''
    Optimizer.py 实现几种优化算法
    SGD
    Adam
'''

import numpy as np
from collections import defaultdict
from Parameter import Parameter

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

"""Optimizer基类，所有优化方法都继承这个类"""
class Optimizer(object):
    # 参数初始化方法
    def __init__(self, params, defaults):
        # self.params = params
        self.defaults = defaults # 优化方法需要使用的参数
        if isinstance(params, Parameter):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            type(params))

        # 存储所需要的参数的状态（例如：momentum）
        # defaultdict的作用在于当字典里的 key 被查找但不存在时，返回的不是keyError而是一个默认值，此处defaultdict(dict)返回的默认值会是个空字典
        self.state = defaultdict(dict)
        # 用于存储参数的列表？
        self.param_groups = []
        param_groups = list(params)
        print('param_groups: \n', param_groups)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict): # 处理单个参数情况？？
            param_groups = [{'params': param_groups}]
    
        for param_group in param_groups:
            self.add_param_group(param_group)

    '''重新组织网络参数，将param_group放进self.param_groups中'''
    def add_param_group(self, param_group):
        # 源码是真的怪异= =
        assert isinstance(param_group, dict), "param group must be a dict" # 断言，判断pram_group必须是个dict

        params = param_group['params']
        if isinstance(params, Parameter): # 
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in '
                            'ordered collections, but the ordering of tensors in sets '
                            'will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, Parameter):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + type(param))
            # ,当requires_grad()为True时我们将会记录tensor的运算过程并为自动求导做准备,但是并不是每个requires_grad()设为True的值都会在backward的时候得到相应的grad.它还必须为leaf.这就说明. leaf成为了在 requires_grad()下判断是否需要保留 grad的前提条件
            # if not param.is_leaf:
            #     raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required "
                                 "optimization parameter " + name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))
        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group) 

    '''将参数的梯度grad设置为0'''
    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.zero_grad()

    '''优化步骤，参数更新'''
    def step(self, closure):
        # 需要在子类中被重写
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        
        .. note::
            Unless otherwise specified, this function should not modify the ``.grad`` field of the parameters.
        """
        raise NotImplementedError
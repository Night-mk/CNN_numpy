'''
    Module.py 实现模型的父类，更好地服务于模型训练（参数更新）
    重构原本的layers代码，更好地管理：layer实例，模型参数
'''

import numpy as np
from collections import OrderedDict # OrderedDict 实现了对字典对象中元素的排序
from Parameter import Parameter


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class Module(object):
    def __init__(self):
        self._parameters = OrderedDict() # 保存保存用户直接设置的parameter
        self._modules = OrderedDict() # 保存子类实例化的模型
        self._buffers = OrderedDict() # 缓存，保存一些不变的量（暂时用不到）
        self._backward_hooks = OrderedDict() # 钩子技术，提取中间变量
        self._forward_hooks = OrderedDict()
        self.training = True # 使用training值决定前向传播策略

    
    '''定义forward函数，需要被所有子类重写'''
    def forward(self, *input):
        raise NotImplementedError

    '''向self._parameters中注册参数'''
    def register_parameter(self, name, param):
        # name参数判断
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        
        elif not isinstance(name, str):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))
        # param参数更新
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(type(param), name))
        # elif param.grad_fn:
        #     raise ValueError(
        #         "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
        #         "parameters must be created explicitly. To express '{0}' "
        #         "as a function of another Tensor, compute the value in "
        #         "the forward() method.".format(name))
        else:
            self._parameters[name] = param
            # print('param type: ',type(param))
            # print('id param: ', id(param))

    '''
        设置module实例的属性来注册模块和参数
        不直接设置，通过对__serattr__方法进行裁剪来实现self._modules,self._parameters,self._buffers的更新
        
        __setattr__方法：每当属性被赋值的时候都会调用该方法
    '''
    # '''
    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]
        # print('name: ',name)
        # print('value: ',value)
        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            # print('param_name: ',name)
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules)
            # 注册parameter参数
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(type(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                # print('module_list_name: ',name)
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(type(value), name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, np):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(type(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)
    # '''

    '''
        功能：获取给定name的Module类中的成员,并返回该值
        由于在__setattr__函数中，我们为了找到全部的_parameters,_buffers,_modules之后会用remove_from将这个值del掉。
        当获取 self.__dict__ 中没有的键所对应的值的时候，就会调用这个方法
        因为 parameter, module, buffer 的键值对存在与 self._parameters, self._modules, self._buffer 中，所以，当想获取这些 值时， 就会调用这个方法。
    '''
    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    '''在网络中添加子模型module'''
    def add_module(self, name, module):
        # 照抄nn.Module的add_module
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                type(module)))
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(
                type(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module


    '''
        功能：为子模型添加fn方法（递归执行）
        将Module及其所有的SubModule传进给定的fn函数操作一遍,
        可以用这个函数来对Module的网络模型参数用指定的方法初始化
    '''
    def apply(self, fn):
        # 子模型由.children()方法获得
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def children(self):
        for name, module in self.named_children():
            yield module

    def named_children(self):
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    '''返回类名'''
    def _get_name(self):
        return self.__class__.__name__
    
    '''可以设置module对象返回的内容，形式（额外表达），需要在子module中重写'''
    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    '''
        可以设置对象的实际返回格式（类型不变，形式可以自定义），打印时调用。
        在使用内建函数 repr()时自动调用
    '''
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    ################################################################################
    
    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    '''获取网络参数'''
    # 因为Module的成员并没有直接派生自Parameter类，所以无法直接获取_parameters的值
    def parameters(self, recurse=True):
        # recurse=True时，返回当前module和所有submodule的参数
        # recurse=False时，仅返回当前module的参数
        for name, param in self.named_parameters(recurse=recurse):
            # yield返回生成器
            yield param
    
    '''可以根据参数名的前缀prefix获取相应参数'''
    def named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    '''递归地获取Module类中的各类参数'''
    def named_modules(self, memo=None, prefix=''):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.
        Yields:
            (string, Module): Tuple of name and module
        """
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    ##################################################################
    '''保存模型参数，用于模型存储'''
    '''存储各个子模型参数（基础方法）'''    
    def _save_to_state_dict(self, destination, prefix):
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix+name] = param.data

    '''存储整个模型的参数'''
    def state_dict(self, destination=None, prefix=''):
        if destination is None:
            destination = OrderedDict()
        self._save_to_state_dict(destination, prefix) # 第一次保存的是整个模型的参数
        # 递归获取全部子模型的参数
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.')
        return destination

    '''加载模型参数，用于模型的提取'''
    def _load_from_state_dict(self, name, param):
        if self._parameters[name] is not None:
            # 加载新参数覆盖原始参数
            self._parameters[name].set_param(param)
    
    def load_state_dict(self, param_dict=None, prefix=''):
        # 设置模型参数
        if param_dict is not None:
            for name, param in param_dict.items(): # [module.param (e.g conv1.weights)]
                name_list = name.split('.')
                if self._modules[name_list[0]] is not None:
                    self._modules[name_list[0]]._load_from_state_dict(name_list[1], param)
                    
    '''将模型的梯度置为0'''
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.zero_grad()
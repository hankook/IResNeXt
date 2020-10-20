from functools import reduce
import operator
import torch, torch.nn.functional as F

__all__ = ["MeasureMode", "measure", "get_state", "add_flops", "add_params"]

_functions = {}

_flops_counter = 0
_params_counter = 0
_curr_state = False

# some utility functions
def _to_tuple(x, dim=2):
    if type(x) is int:
        return (x, )*dim
    else:
        assert dim == len(x)
        return x

def add_wrapper(func):
    name = func.__name__
    assert name.endswith('_wrapper')
    name = name[:-8]
    f = F.__dict__[name]
    _functions[name] = (f, func(f))

def get_state():
    return _curr_state

def add_flops(x):
    global _flops_counter
    if get_state():
        _flops_counter += x

def add_params(x):
    global _params_counter
    if get_state():
        _params_counter += x

# wrapper functions for computing flops and parameters
@add_wrapper
def relu_wrapper(func):
    def inner_func(input, inplace=False):
        add_flops(input.numel())
        return func(input, inplace)
    return inner_func

@add_wrapper
def threshold_wrapper(func):
    def inner_func(input, threshold, value, inplace=False):
        add_flops(input.numel())
        return func(input, threshold, value, inplace)
    return inner_func

@add_wrapper
def avg_pool2d_wrapper(func):
    def inner_func(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        output = func(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
        kh, kw = _to_tuple(kernel_size, 2)
        add_flops(input.size(0) * input.size(1) * output.size(2) * output.size(3) * kh * kw)
        return output
    return inner_func

@add_wrapper
def conv2d_wrapper(func):
    def inner_func(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        output = func(input, weight, bias, stride, padding, dilation, groups)
        add_flops(input.size(0) * weight.numel() * output.size(2) * output.size(3))
        add_params(weight.numel() + (0 if bias is None else bias.numel()))
        return output
    return inner_func

@add_wrapper
def linear_wrapper(func):
    def inner_func(input, weight, bias=None):
        add_flops(input.size(0) * (weight.numel() + (0 if bias is None else bias.numel())))
        add_params(weight.numel() + (0 if bias is None else bias.numel()))
        return func(input, weight, bias)
    return inner_func

@add_wrapper
def batch_norm_wrapper(func):
    def inner_func(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
        add_flops(input.numel())
        add_params(0 if weight is None else weight.numel())
        add_params(0 if bias is None else bias.numel())
        return func(input, running_mean, running_var, weight, bias, training, momentum, eps)
    return inner_func

class MeasureMode(object):
    def __init__(self, init):
        self.init = init

    def __enter__(self):
        for name, (_, f) in _functions.items():
            F.__dict__[name] = f

        global _flops_counter, _params_counter, _curr_state
        if self.init:
            _flops_counter = 0
            _params_counter = 0

        self.old_state = _curr_state
        _curr_state = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, (f, _) in _functions.items():
            F.__dict__[name] = f

        global _curr_state
        _curr_state = self.old_state

def measure(model, *args, **kwargs):
    with MeasureMode(True):
        model(*args, **kwargs)
    return _flops_counter, _params_counter

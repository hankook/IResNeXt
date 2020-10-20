import argparse
import models
import inspect

def eval_or_str(string):
    try:
        return eval(string)
    except:
        return string

def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # common arguments
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--resume', action='store_true')

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--valsize', type=int, default=5000)

    # optimization arguments
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--nesterov', action='store_true')

    # parse arguments
    args, remaining_args = parser.parse_known_args()

    # get model arguments
    f = models.__dict__[args.model]
    if inspect.isclass(f):
        argspec = inspect.getargspec(f.__init__)
        argspec.args.pop(0)
    else:
        argspec = inspect.getargspec(f)
    assert len(argspec.args) == len(argspec.defaults)

    # parse model arguments
    model_parser = argparse.ArgumentParser()
    for argname, default in zip(argspec.args, argspec.defaults):
        model_parser.add_argument('--' + argname, type=eval_or_str, default=default)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args

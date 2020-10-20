import numpy as np

__all__ = ['Meter', 'AverageMeter', 'ClassErrorMeter']

class Meter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def __call__(self):
        return self.value()

class AverageMeter(Meter):
    def reset(self):
        self.total = 0.0
        self.count = 0

    def add(self, value, n=1):
        self.total += value*n
        self.count += n
        return value

    def value(self):
        return self.total / self.count

class ClassErrorMeter(AverageMeter):
    def add(self, pred, target):
        t = np.sum(np.argmax(pred.numpy(), axis=1) != target.numpy())
        c = pred.size(0)
        return super(ClassErrorMeter, self).add(t/float(c), n=c)

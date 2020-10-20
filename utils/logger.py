# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from __future__ import print_function
from tensorboardX import SummaryWriter
import numpy as np
import os, sys, torch, shutil, time
from datetime import datetime
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

__all__ = ['Logger']

class Logger(object):
    def __init__(self, logdir, read_only=False):
        self.logdir = logdir

        if not os.path.exists(logdir):
            os.mkdir(logdir)

        if len(os.listdir(logdir)) != 0 and not read_only:
            prompt = "'%s' is not empty. Files inside the direcotry can be over-writted. Will you proceed [YES/n]? "
            prompt = prompt%logdir
            ans = input(prompt)
            if ans != 'YES':
                exit(1)

        if not read_only:
            self.writer = SummaryWriter(logdir)

        if not read_only:
            self.logfile = open(os.path.join(logdir, 'log.txt'), 'a')
        else:
            self.logfile = None

        self.log(logdir)

    def log(self, string):
        now = datetime.now()
        if self.logfile is not None:
            self.logfile.write('[%s] %s'%(now, string) + '\n')
            self.logfile.flush()

        print('[%s] %s'%(now, string))
        sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        self.writer.add_image(tag, images, step)

    def histo_summary(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def save(self, state_dict, name):
        torch.save(state_dict, os.path.join(self.logdir, name))

    def load(self, name, map_location="cuda:0"):
        return torch.load(os.path.join(self.logdir, name), map_location=map_location)

    def save_checkpoint(self, epoch, model, optimizer, is_best):
        self.save(model.state_dict(), 'last.model')
        self.save(optimizer.state_dict(), 'last.optim')
        with open(os.path.join(self.logdir, 'last.epoch'), 'w') as f:
            f.write(str(epoch))

        if is_best:
            self.save(model.state_dict(), 'best.model')

    def load_checkpoint(self):
        model_state = self.load('last.model')
        optim_state = self.load('last.optim')
        with open(os.path.join(self.logdir, 'last.epoch')) as f:
            epoch = int(f.read())
        return model_state, optim_state, epoch

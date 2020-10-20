import torch
import torch.nn as nn, torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import models
from options import parse_args
from utils import Logger, AverageMeter, ClassErrorMeter
from datasets import get_dataset
from utils.measure_v2 import measure

import argparse

torch.backends.cudnn.benchmark = True

###########################
# DEFINE GLOBAL VARIABLES #
###########################

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--add_wd_low', type=float, default=0.0)
parser.add_argument('--add_wd_high', type=float, default=0.0)
args, model_args = parse_args(parser)

# define logger
logdir = args.logdir
logger = Logger(logdir, read_only=args.test_only)
logger.log('args: %s'%str(args))
logger.log('model args: %s'%str(model_args))

# define model
model = models.get_model(args.model, model_args).cuda()
logger.log('full-model FLOPs: %d'%measure(model, torch.zeros(1, 3, 32, 32).cuda(), k=-1)[0])

# define datasets - 0: train, 1: val, 2: test
datasets = get_dataset(args.dataset, val_size=args.valsize)
dataloaders = []
for d in datasets:
    dataloaders.append(DataLoader(d,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4))

# define loss
criterion = nn.CrossEntropyLoss().cuda()

# define optimizer
optimizer = optim.SGD(model.parameters(),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.wd,
                      nesterov=args.nesterov)

# define lr scheduler
lr_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[args.num_epochs/2, args.num_epochs*3/4],
    gamma=0.1)

####################
# DEFINE FUNCTIONS #
####################

def log_iter(epoch, mode, i, losses, errors):
    prompt = '[epoch %3d] [%s] [iter %3d] [loss %.6f %.6f] [error %.4f]'
    logger.log(prompt%(epoch, mode, i, sum(losses), losses[-1], errors[-1]))

def log_epoch(epoch, mode, losses, errors):
    prompt = '[epoch %3d] [%s] [anytime %d] [loss %.6f] [error %.4f]'
    for k, (l, e) in enumerate(zip(losses, errors)):
        logger.log(prompt%(epoch, mode, k, l(), e()))
    logger.scalar_summary('loss/%s'%mode, losses[-1](), epoch)
    logger.scalar_summary('error/%s'%mode, errors[-1](), epoch)

def train(epoch, dataloader):
    model.train()
    loss_average = [AverageMeter() for _ in range(model.anytime)]
    error_average = [ClassErrorMeter() for _ in range(model.anytime)]
    for i, (images, labels) in enumerate(dataloader):
        n = images.size()[0]

        optimizer.zero_grad()
        losses = []
        errors = []
        for k in range(model.anytime):
            y = model(Variable(images.cuda()), k)

            loss = criterion(y, Variable(labels.cuda()))
            losses.append(loss_average[k].add(loss.data[0], n))
            errors.append(error_average[k].add(y.cpu().data, labels))
            loss.backward()

        reg = 0.0
        c = model.anytime
        low, high = args.add_wd_low, args.add_wd_high
        if low < high:
            wds = torch.arange(low, high+1e-8, step=(high-low)/float(c-1))
            wds = Variable(wds.view(c, 1).cuda())
            for stage in model.stages:
                for block in stage:
                    for name in ['conv1', 'conv2']:
                        m = block._modules[name]
                        for param in m.parameters():
                            reg = reg + param.pow(2).view(c, -1).mul(wds).sum()
                    h = block.conv3.weight.size(0)
                    reg = reg + block.conv3.weight.pow(2).view(h, c, -1).mul(wds.view(1, c, 1)).sum()

                    for name in ['bn2', 'bn3']:
                        for k in range(model.anytime):
                            m = block._modules[name][k]
                            for param in m.parameters():
                                reg = reg + param.pow(2).view(k+1, -1).mul(wds[:k+1]).sum()
            reg.backward()

        optimizer.step()
        log_iter(epoch, 'train', i, losses, errors)

    log_epoch(epoch, 'train', loss_average, error_average)
    return error_average[-1]()

def test(epoch, dataloader, val=False):
    if len(dataloader) == 0:
        return 1.0

    mode = 'val' if val else 'test'
    model.eval()
    loss_average = [AverageMeter() for _ in range(model.anytime)]
    error_average = [ClassErrorMeter() for _ in range(model.anytime)]
    for i, (images, labels) in enumerate(dataloader):
        with torch.no_grad():
            n = images.size()[0]
            losses = []
            errors = []
            for k in range(model.anytime):
                y = model(Variable(images.cuda()), k)

                loss = criterion(y, Variable(labels.cuda()))
                losses.append(loss_average[k].add(loss.data[0], n))
                errors.append(error_average[k].add(y.cpu().data, labels))

            log_iter(epoch, mode, i, losses, errors)

    log_epoch(epoch, mode, loss_average, error_average)
    return error_average[-1]()

###############
# MAIN SCRIPT #
###############

if args.test_only:
    model.load_state_dict(logger.load('best.model'))
    val_error  = test(0, dataloaders[1], val=True)
    test_error = test(0, dataloaders[2], val=False)
    exit()

last_epoch = -1
if args.resume:
    model_state, optim_state, last_epoch = logger.load_checkpoint()
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)

val_best = 1.0
test_best = 1.0
for epoch in range(last_epoch+1, args.num_epochs):
    lr_scheduler.step(epoch=epoch)

    train(epoch, dataloaders[0])
    val_error  = test(epoch, dataloaders[1], val=True)
    test_error = test(epoch, dataloaders[2], val=False)

    state_dict = model.state_dict()
    is_best = False
    if (args.valsize > 0 and val_best > val_error) or (args.valsize == 0 and test_best > test_error):
        val_best = val_error
        test_best = test_error
        is_best = True
    logger.save_checkpoint(epoch, model, optimizer, is_best)
    logger.log('[epoch %3d] [best] [val %.4f] [test %.4f]'%(epoch, val_best, test_best))

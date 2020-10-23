import torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

import models
from options import parse_args
from utils import Logger, AverageMeter, ClassErrorMeter
from datasets import get_dataset
from utils.measure_v2 import measure

torch.backends.cudnn.benchmark = True

###########################
# DEFINE GLOBAL VARIABLES #
###########################

# parse arguments
args, model_args = parse_args()

# define logger
logdir = args.logdir
logger = Logger(logdir, read_only=args.test_only)
logger.log('args: %s'%str(args))
logger.log('model args: %s'%str(model_args))

# define model
model = models.__dict__[args.model](**vars(model_args)).cuda()
# logger.log('full-model FLOPs: %d' % measure(model, torch.zeros(1, 3, 32, 32).cuda())[0])

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

def log_iter(epoch, mode, i, loss, error):
    prompt = '[epoch %3d] [%s] [iter %3d] [loss %.6f] [error %.4f]'
    logger.log(prompt%(epoch, mode, i, loss, error))

def log_epoch(epoch, mode, loss, error):
    prompt = '[epoch %3d] [%s] [loss %.6f] [error %.4f]'
    logger.log(prompt%(epoch, mode, loss, error))
    logger.scalar_summary('loss/%s'%mode, loss, epoch)
    logger.scalar_summary('error/%s'%mode, error, epoch)

def train(epoch, dataloader):
    model.train()
    loss_average = AverageMeter()
    error_average = ClassErrorMeter()
    for i, (images, labels) in enumerate(dataloader):
        n = images.size()[0]
        y = model(images.cuda())

        optimizer.zero_grad()
        loss = criterion(y, labels.cuda())
        loss_average.add(loss.item(), n)
        error = error_average.add(y.cpu().data, labels)
        loss.backward()
        optimizer.step()

        log_iter(epoch, 'train', i, loss.item(), error)

    log_epoch(epoch, 'train', loss_average(), error_average())
    return error_average()

def test(epoch, dataloader, val=False):
    if len(dataloader) == 0:
        return 1.0

    mode = 'val' if val else 'test'
    model.eval()
    loss_average = AverageMeter()
    error_average = ClassErrorMeter()
    for i, (images, labels) in enumerate(dataloader):
        with torch.no_grad():
            n = images.size()[0]
            y = model(images.cuda())

            loss = criterion(y, labels.cuda())
            loss_average.add(loss.item(), n)
            error = error_average.add(y.cpu().data, labels)

            log_iter(epoch, mode, i, loss.item(), error)

    log_epoch(epoch, mode, loss_average(), error_average())
    return error_average()

###############
# MAIN SCRIPT #
###############

if args.test_only:
    model.load_state_dict(logger.load('best.model'))
    val_error  = test(0, dataloaders[1], val=True)
    test_error = test(0, dataloaders[2], val=False)
    exit()

val_best = 1.0
test_best = 1.0
for epoch in range(args.num_epochs):
    lr_scheduler.step()

    train(epoch, dataloaders[0])
    val_error  = test(epoch, dataloaders[1], val=True)
    test_error = test(epoch, dataloaders[2], val=False)

    state_dict = model.state_dict()
    logger.save(state_dict, 'last.model')
    if (args.valsize > 0 and val_best > val_error) or (args.valsize == 0 and test_best > test_error):
        val_best = val_error
        test_best = test_error
        logger.save(state_dict, 'best.model')
    logger.log('[epoch %3d] [best] [val %.4f] [test %.4f]'%(epoch, val_best, test_best))

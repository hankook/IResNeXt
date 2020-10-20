import models
from options import parse_args
from utils import Logger
from datasets import get_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True

def main(trainer_class):
    # parse arguments
    args, model_args = parse_args()

    # define logger
    logdir = args.logdir
    logger = Logger(logdir, read_only=args.test_only)
    logger.log('args: %s'%str(args))
    logger.log('model args: %s'%str(model_args))

    # define model
    model = models.__dict__[args.model](**vars(model_args)).cuda()

    # define datasets - 0: train, 1: val, 2: test
    datasets = get_dataset(args.dataset, val_size=args.valsize)
    dataloaders = []
    for d in datasets:
        dataloaders.append(DataLoader(d, batch_size=args.batch_size, shuffle=True, num_workers=4))

    # define loss
    ce_loss = nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd,
                                nesterov=args.nesterov)

    # define lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[args.num_epochs/2, args.num_epochs*3/4],
                                                        gamma=0.1)

    # train
    val_best = 100.0
    test_best = 100.0
    trainer = trainer_class(model, ce_loss, optimizer, logger)

    if args.test_only:
        model.load_state_dict(logger.load('best.model'))
        if args.val_size > 0:
            val_loss, val_error   = trainer.test(0, dataloaders[1], val=True)
        test_loss, test_error = trainer.test(0, dataloaders[2], val=False)
        return

    for epoch in range(args.num_epochs):
        lr_scheduler.step()

        trainer.train(epoch, dataloaders[0])
        if args.valsize > 0:
            val_loss, val_error   = trainer.test(epoch, dataloaders[1], val=True)
        else:
            val_error = 100.0
        test_loss, test_error = trainer.test(epoch, dataloaders[2], val=False)

        state_dict = model.state_dict()
        logger.save(state_dict, 'last.model')
        if (args.valsize > 0 and val_best > val_error) or (args.valsize == 0 and test_best > test_error):
            val_best = val_error
            test_best = test_error
            logger.save(state_dict, 'best.model')
        logger.log('[epoch %3d] [best] [val %.4f] [test %.4f]'%(epoch, val_best, test_best))

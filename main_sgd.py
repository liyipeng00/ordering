"""
Modified from https://github.com/EugeneLYC/GraB
and https://github.com/GarlGuo/CD-GraB
"""

import time
import torch
from torch.utils.data import Subset

from sim.data.datasets import build_dataset
#from sim.models.build_models import build_model
from sim.utils.record_utils import logconfig, add_log_info, add_log_debug, record_exp_result
from sim.utils.utils import setup_seed

from sgd.build_models import build_model
from sgd.utils import evaluate_dataset, train
from sgd.utils import _RANDOM_RESHUFFLING_, _SHUFFLE_ONCE_, _FLIPFLOP_SORT_, _GRAB_, _PAIRGRAB_

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', default='vgg9', type=str, help='Model')
parser.add_argument('-d', default='cifar10', type=str, help='Dataset')
parser.add_argument('--num_workers',default=0,type=int,metavar='N',help='number of data loading workers (default: 0)')
parser.add_argument('--epochs',default=200,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('--eval-every', default=1, type=int, help='Evaluate every several epochs')
parser.add_argument('--start_epoch',default=0,type=int,metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--grad_accumulation_step', default=1, type=int, metavar='N', help='gradient accumulation step in the optimization (default: 1)')
parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr-scheduler', default='exp', type=str, help='exp/multistep')
parser.add_argument('--lr-decay', default=1.0, type=float, help='Learning rate decay')
parser.add_argument('--momentum', default=0.0, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=0.0, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--start_sort', default=1, type=int, metavar='N', help='the epoch where the greedy strategy will be first used (100 in CIFAR10 case)')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='random seed used in the experiment')
parser.add_argument('--clip', default=0, type=int, help='Clip')
parser.add_argument('--log', default='', type=str, help='info/debug+log, info/debug+print, info/debug+no')
parser.add_argument('--device', default=0, type=int, help='Device')
parser.add_argument('--shuffle_type',default='random_reshuffling',type=str,help='shuffle type used for the optimization (choose from random_reshuffling, shuffle_once, stale_grad_greedy_sort, fresh_grad_greedy_sort)')

args = parser.parse_args()

eval_batch_size = 32
log_level, log_mode = args.log.split('+')
torch.set_num_threads(4)
setup_seed(args.seed)
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

def build_record_name(args):
    '''SO_R2,10_logistic_mnist_sgd0.01,0.0,0.0_exp1.0_b2,32_seed0_clip0'''
    record_name = f'{args.shuffle_type}_R{args.epochs},{args.eval_every}_{args.m}_{args.d}'\
                + f'_{args.optim}{args.lr},{args.momentum},{args.weight_decay}'\
                + f'_{args.lr_scheduler}{args.lr_decay}_b{args.batch_size},{args.grad_accumulation_step}_seed{args.seed}_clip{args.clip}'
    return record_name

record_name = build_record_name(args)


def main():
    global args, record_name, device
    logconfig(name=record_name, level=log_level, mode=log_mode)
    add_log_info('{}'.format(args), level=log_level, mode=log_mode)
    add_log_info('record_name: {}'.format(record_name), level=log_level, mode=log_mode)

    criterion = torch.nn.CrossEntropyLoss()

    model = build_model(model_name=args.m, dataset_name=args.d)
    model.to(device)
    model_dimen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    add_log_info(f"Using model: {args.m} with dimension: {model_dimen}.", level=log_level, mode=log_mode)
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 2024-12-29. It uses a constant learning rate.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1, last_epoch=args.start_epoch-1)

    original_dataset = build_dataset(dataset_name=args.d)
    trainset = original_dataset.get_trainset(transform=original_dataset.transform_test)
    #initial_order = torch.randperm(len(trainset))
    #trainset = Subset(trainset, initial_order)
    testset = original_dataset.get_testset(transform=original_dataset.transform_test)

    shuffle_flag = True if args.shuffle_type in [_RANDOM_RESHUFFLING_] else False
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=shuffle_flag)
    train_val_loader = torch.utils.data.DataLoader(trainset, batch_size=eval_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False)
    
    # Epoch-wise data ordering
    if args.shuffle_type in [_RANDOM_RESHUFFLING_, _SHUFFLE_ONCE_]:
        # 2025-07-28. SO is included: see (1) shuffle_flag and (2) shuffle_type.
        sorter = None
    else:
        # 2024-12-29. whether to use projection when doing the greedy sorting
        grad_dimen = model_dimen
        num_batches = len(list(enumerate(train_loader)))
        if args.shuffle_type == _FLIPFLOP_SORT_:
            from sgd.algo import FlipFlopSort
            sorter = FlipFlopSort(args, num_batches)
        elif args.shuffle_type == _GRAB_:
            from sgd.algo import GraBSort
            sorter = GraBSort(args, num_batches, grad_dimen, device)
        elif args.shuffle_type == _PAIRGRAB_:
            from sgd.algo import PairGraBSort
            sorter = PairGraBSort(args, num_batches, grad_dimen, device)
        else:
            raise NotImplementedError("This sorting method is not supported yet")

    start_time = time.time()
    record_exp_result(record_name, [0])
    for epoch in range(args.start_epoch, args.epochs):
        exp_result_epoch = [epoch+1]

        train(args=args, loader=train_loader, model=model, criterion=criterion, device=device, optimizer=optimizer, epoch=epoch, sorter=sorter)
        # evaluate on training set
        train_losses, train_top1, train_top5 = evaluate_dataset(model=model, data_loader=train_val_loader, criterion=criterion, device=device)
        add_log_info("Round {}'s server train  acc: {:6.2f}%, train loss: {:.4f}".format(epoch+1, train_top1.avg, train_losses.avg), level=log_level, mode=log_mode, color='red')
        exp_result_epoch.extend([train_losses.avg, train_top1.avg, train_top5.avg])

        # evaluate on test set
        test_losses, test_top1, test_top5 = evaluate_dataset(model=model, data_loader=test_loader, criterion=criterion, device=device)
        add_log_info("Round {}'s server test   acc: {:6.2f}%, test  loss: {:.4f}".format(epoch+1, test_top1.avg, test_losses.avg), level=log_level, mode=log_mode, color='blue')
        exp_result_epoch.extend([test_losses.avg, test_top1.avg, test_top5.avg])

        record_exp_result(record_name, exp_result_epoch)

    end_time = time.time()
    add_log_info("TrainingTime: {} sec".format(end_time - start_time), level=log_level, mode=log_mode)

if __name__ == '__main__':
    main()

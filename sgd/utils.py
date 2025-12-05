import torch
import copy
#from sklearn import random_projection

from sim.utils.utils import accuracy, AverageMeter

_RANDOM_RESHUFFLING_ = 'RR'
_SHUFFLE_ONCE_ = 'SO'
_FLIPFLOP_SORT_ = 'FlipFlop'
_GRAB_ = 'GraB'
_PAIRGRAB_ = 'PairGraB'


def compute_avg_grad_error(args,
                        model,
                        train_batches,
                        optimizer,
                        epoch,
                        tb_logger,
                        oracle_type='cv',
                        orders=None):
    grads = dict()
    for i in range(len(train_batches)):
        grads[i] = flatten_params(model).zero_()
    full_grad = flatten_params(model).zero_()
    if orders is None:
        orders = {i:0 for i in range(len(train_batches))}
    for j in orders.keys():
        i, batch = train_batches[j]
        if oracle_type == 'cv':
            loss, _, _ = model(batch)
            optimizer.zero_grad()
            loss.backward()
        else:
            raise NotImplementedError
        grads[i] = flatten_grad(optimizer)
        full_grad.add_(grads[i])
    cur_grad = flatten_params(model).zero_()
    index, cur_var = 0, 0
    for j in orders.keys():
        i, _ = train_batches[j]
        for p1, p2, p3 in zip(cur_grad, grads[i], full_grad):
            p1.data.add_(p2.data)
            cur_var += torch.norm(p1.data/(index+1) - p3.data/len(train_batches)).item()**2
        index += 1
    tb_logger.add_scalar('train/metric', cur_var, epoch)

def flatten_grad(optimizer):
    t = []
    for _, param_group in enumerate(optimizer.param_groups):
        for p in param_group['params']:
            if p.grad is not None: t.append(p.grad.data.view(-1))
    return torch.concat(t)

def flatten_params(model):
    t = []
    for _, param in enumerate(model.parameters()):
        if param is not None: t.append(param.data.view(-1))
    return torch.concat(t)


def train(args,loader,model,criterion, device, optimizer, epoch, sorter=None):

    model.train()
    grad_buffer = copy.deepcopy(model)
    for p in grad_buffer.parameters():
        p.data.zero_()
    train_batches = list(enumerate(loader))
    num_batches = len(train_batches)

    if sorter is not None:
        if args.shuffle_type == _FLIPFLOP_SORT_:
            orders = sorter.sort(epoch=epoch)
        elif args.shuffle_type in [_GRAB_, _PAIRGRAB_]:
            orders = sorter.sort()
        else:
            raise NotImplementedError
    else:
        orders = {i:0 for i in range(len(train_batches))}

    grad_step = 0
    cur_step = 0
    for i in orders.keys():
        grad_step += 1
        cur_step += 1
        _, batch = train_batches[i]
        input_var, target_var = batch
        input_var = input_var.to(device)
        target_var = target_var.to(device)
        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        
        for p1, p2 in zip(grad_buffer.parameters(), model.parameters()):
            # 2024-12-30. Store the cumulative gradient to `grad_buffer`.
            p1.data.add_(p2.grad.data)

        if sorter is not None and args.shuffle_type in [_GRAB_, _PAIRGRAB_]:
            sorter.step(optimizer=optimizer, batch_idx=i)
        
        if grad_step % args.grad_accumulation_step == 0 or grad_step == num_batches:
            for p1, p2 in zip(grad_buffer.parameters(), model.parameters()):
                # 2024-12-30. Return the cumulative gradient to `model`
                p1.data.mul_(1/cur_step)
                p2.grad.data.zero_().add_(p1.data)
                p1.data.zero_()
            optimizer.step()
            cur_step = 0


def evaluate_dataset(model, data_loader, criterion, device):
    '''Evaluate on the given dataset'''
    #data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for input, target in data_loader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=[1,5])
            losses.update(loss.item(), target.size(0))
            top1.update(acc1.item(), target.size(0))
            top5.update(acc5.item(), target.size(0))

        return losses, top1, top5,
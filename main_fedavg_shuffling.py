r"""Federated Averaging (FedAvg), Less memory

Sampling and Averaging Scheme:
\bar{x} = \sum_{s\in S} \frac{1}{\sum_{i\in S}p_i} p_s x_s,
where `S` clients is selected without replacement per round and `p_s=n_s/n` is the weight of client `s`. 

References:
[1] https://github.com/chandra2thapa/SplitFed
[2] https://github.com/AshwinRJ/Federated-Learning-PyTorch
[3] https://github.com/lx10077/fedavgpy/
"""
import torch
import time
import copy
import re

from sim.algorithms.fedbase import FedClient, ShufflingFedServer
from sim.data.data_utils import FedDataset
from sim.data.datasets import build_dataset
from sim.data.partition import build_partition
from sim.models.build_models import build_model
from sim.utils.record_utils import logconfig, add_log_info, add_log_debug, record_exp_result
from sim.utils.utils import setup_seed
from sim.utils.optim_utils import OptimKit, LrUpdater

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', default='vgg9', type=str, help='Model')
parser.add_argument('-d', default='cifar10', type=str, help='Dataset')
parser.add_argument('-s', default=2, type=int, help='Index of split layer')
parser.add_argument('-R', default=200, type=int, help='Number of total training rounds')
parser.add_argument('-K', default=1, type=int, help='Number of local steps')
parser.add_argument('-M', default=100, type=int, help='Number of total clients')
parser.add_argument('-P', default=100, type=int, help='Number of clients participate')
parser.add_argument('--partition', default='iid', type=str, help='Data partition')
parser.add_argument('--alpha', default=10, type=float, nargs='*', help='The parameter `alpha` of dirichlet distribution')
parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer')
parser.add_argument('--lr', default=0.0, type=float, help='Client/Local learning rate')
parser.add_argument('--lr-scheduler', default='exp', type=str, help='exp/multistep')
parser.add_argument('--lr-decay', default=1.0, type=float, help='Learning rate decay')
parser.add_argument('--momentum', default=0.0, type=float, help='Momentum of client optimizer')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay of client optimizer')
parser.add_argument('--global-lr', default=1.0, type=float, help='Server/Global learning rate')
parser.add_argument('--batch-size', default=50, type=int, help='Mini-batch size')
parser.add_argument('--seed', default=1234, type=int, help='Seed')
parser.add_argument('--clip', default=0, type=int, help='Clip')
parser.add_argument('--log', default='', type=str, help='info+log')
parser.add_argument('--eval-every', default=10, type=int, help='Number of evaluations')
parser.add_argument('--device', default=0, type=int, help='Device')
parser.add_argument('--start-round', default=0, type=int, help='Start')
parser.add_argument('--save-model', default=0, type=int, help='Whether to save model')
args = parser.parse_args()

# nohup python main_fedavg.py -m logistic -d mnist -s 1 -R 100 -K 10 -M 500 -P 10 --partition iid --alpha 2 10 --optim sgd --lr 0.1 --lr-decay 0.98 --momentum 0 --batch-size 20 --seed 1234 --eval-case 1 --log Print &

torch.set_num_threads(4)
setup_seed(args.seed)
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
if args.partition in ['exdir', 'exdirb']:
    args.alpha = [int(args.alpha[0]), args.alpha[1]] 
elif args.partition in ['shard']:
    args.alpha = [int(args.alpha[0])] 
log_level, log_mode = args.log.split('+')

def customize_record_name(args):
    '''ShufflingFedAvg_M10_P10_K2_R4_mlp_mnist_exdir2,10.0_sgd0.001,1.0,0.0,0.0001_b20_seed1234_clip0.csv'''
    if args.partition in ['exdir', 'exdirb']:
        partition = f'{args.partition}{args.alpha[0]},{args.alpha[1]}'
    elif args.partition in ['shard']:
        partition = f'{args.partition}{args.alpha[0]}'
    elif args.partition == 'iid':
        partition = f'{args.partition}'
    record_name = f'ShufflingFedAvg{args.global_lr}_M{args.M},{args.P}_K{args.K}_R{args.R},{args.eval_every}_{args.m}_{args.d}_{partition}'\
                + f'_{args.optim}{args.lr},{args.momentum},{args.weight_decay}_{args.lr_scheduler}{args.lr_decay}_b{args.batch_size}_seed{args.seed}_clip{args.clip}'
    return record_name
record_name = customize_record_name(args)

def main():
    global args, record_name, device
    logconfig(name=record_name, level=log_level, mode=log_mode)
    add_log_info('{}'.format(args), level=log_level, mode=log_mode)
    add_log_info('record_name: {}'.format(record_name), level=log_level, mode=log_mode)
    
    client = FedClient()
    server = ShufflingFedServer()

    origin_dataset = build_dataset(dataset_name=args.d)
    partition_map = build_partition(args.d, args.M, args.partition, args.alpha)
    feddataset = FedDataset(origin_dataset, partition_map)
    client.setup_feddataset(feddataset)

    global_model = build_model(args.m, args.d)
    server.setup_model(global_model.to(device))
    add_log_info('{}'.format(global_model), level=log_level, mode=log_mode)

    # construct optim kit
    client_optim_kit = OptimKit(optim_name=args.optim, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.lr_scheduler == 'exp':
        client_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater, mul=args.lr_decay)
    elif args.lr_scheduler == 'multistep':
        client_optim_kit.setup_lr_updater(LrUpdater.multistep_lr_updater, mul=args.lr_decay, total_rounds=args.R)
    client.setup_optim_kit(client_optim_kit)
    client.setup_criterion(torch.nn.CrossEntropyLoss())
    server.setup_optim_settings(lr=args.global_lr)

    start_time = time.time()
    record_exp_result(record_name, [0])
    for round in range(args.start_round, args.R):
        server.aggregate_reset()
        selected_clients = server.select_clients(args.M, args.P)
        add_log_debug('selected clients: {}'.format(selected_clients), level=log_level, mode=log_mode)
        for c_id in selected_clients:
            local_delta, local_update_log = client.local_update_step(c_id=c_id, model=copy.deepcopy(server.global_model), num_steps=args.K, device=device, clip=args.clip)
            # if local_update_log != {}:
            #     add_log('{}'.format(local_update_log.__str__()), mode=args.log) 
            server.aggregate_update(local_delta, weight=client.feddataset.get_datasetsize(c_id))
        server.aggregate_avg()
        param_vec_curr = server.global_update()
        torch.nn.utils.vector_to_parameters(param_vec_curr, server.global_model.parameters())

        client.optim_kit.update_lr(round+1)
        add_log_debug('lr={}'.format(client.optim_kit.settings['lr']), level=log_level, mode=log_mode)

        if (round+1) % max(args.eval_every, 1) == 0:
            exp_result_round = [round+1]
            
            # evaluate on train dataset (should not use data augmentation)
            train_losses, train_top1, train_top5 = client.evaluate_dataset(model=server.global_model, dataset=client.feddataset.get_eval_trainset(), device=args.device)
            add_log_info("Round {}'s server train  acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train_top1.avg, train_losses.avg), level=log_level, mode=log_mode, color='red')
            exp_result_round.extend([train_losses.avg, train_top1.avg, train_top5.avg])

            # evaluate on test dataset
            test_losses, test_top1, test_top5 = client.evaluate_dataset(model=server.global_model, dataset=client.feddataset.get_eval_testset(), device=args.device)
            add_log_info("Round {}'s server test   acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg), level=log_level, mode=log_mode, color='blue')
            exp_result_round.extend([test_losses.avg, test_top1.avg, test_top5.avg])
            
            record_exp_result(record_name, exp_result_round)
    
    if args.save_model == 1:
        torch.save({'model': torch.nn.utils.parameters_to_vector(server.global_model.parameters())}, './save_model/{}.pt'.format(record_name))

    end_time = time.time()
    add_log_info("TrainingTime: {} sec".format(end_time - start_time), level=log_level, mode=log_mode)

if __name__ == '__main__':
    main()
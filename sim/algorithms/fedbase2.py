import numpy as np
import torch
import copy
from torch.utils.data import DataLoader
from sim.utils.utils import AverageMeter, accuracy

eval_batch_size = 32

###### CLIENT ######
class FedClient():
    def __init__(self):
        pass
    
    def setup_criterion(self, criterion):
        self.criterion = criterion

    def setup_train_dataset(self, dataset):
        self.train_feddataset = dataset
    
    def setup_test_dataset(self, dataset):
        self.test_dataset = dataset

    def setup_optim_kit(self, optim_kit):
        self.optim_kit = optim_kit
    
    #client.local_update_step(model=copy.deepcopy(server.global_model), dataset=client.train_feddataset.get_dataset(c_id), num_steps=args.K, device=device, clip=args.clip)
    def local_update_step(self, model, dataset, num_steps, device, **kwargs):
        data_loader = DataLoader(dataset, batch_size=self.optim_kit.batch_size, shuffle=True)
        optimizer = self.optim_kit.optim(model.parameters(), **self.optim_kit.settings)
        
        model.train()
        step_count = 0
        while(True):
            for input, target in data_loader:
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = self.criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                if 'clip' in kwargs.keys() and kwargs['clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=kwargs['clip'])
                optimizer.step()
                step_count += 1
                if (step_count >= num_steps):
                    break
            if (step_count >= num_steps):
                break
        with torch.no_grad():
            return torch.nn.utils.parameters_to_vector(model.parameters())

    def local_update_epoch(self, client_model,data, epoch, batchsize):
        pass

    def evaluate_dataset(self, model, dataset, device):
        '''Evaluate on the given dataset'''
        data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for input, target in data_loader:
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = self.criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=[1,5])
                losses.update(loss.item(), target.size(0))
                top1.update(acc1.item(), target.size(0))
                top5.update(acc5.item(), target.size(0))

            return losses, top1, top5,

###### SERVER ######
class FedServer():
    '''It collects all the client model parameters in a list. This can waste GPU memory.
    So, this code is reserved for some special algorithms that may need all the client models.
    Date: 2025-07-25
    '''
    def __init__(self):
        pass
    
    def setup_model(self, model):
        self.global_model = model

    def select_clients(self, num_clients, num_clients_per_round):
        '''https://github.com/lx10077/fedavgpy/blob/master/src/trainers/base.py'''
        num_clients_per_round = min(num_clients_per_round, num_clients)
        return np.random.choice(num_clients, num_clients_per_round, replace=False)

    def aggregate(self, local_params):
        '''https://github.com/Divyansh03/FedExP/blob/main/main.py'''
        with torch.no_grad():
            param_avg = torch.zeros_like(local_params[0][0])
            weights_sum = torch.tensor(0)
            for param, weight in local_params:
                param_avg.add_(weight * param)
                weights_sum.add_(weight)
            return param_avg.div_(weights_sum)

    
'''Version 2
We consider the strongly convex functions.
'''

import torch
import torch.nn as nn
from torch.optim import SGD

import random
import numpy as np
import copy
import os
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='quad0', type=str, help="Task")
parser.add_argument('--alg', default='cSGD', type=str, help='Algorithm')
#parser.add_argument('--alghp', default=10, type=float, nargs='*', help='Hyperparameter of the algorithm')
parser.add_argument('-R', default=200, type=int, help='Number of total training epochs') # Note!!! This gives the number of "epochs", rather than "rounds".
parser.add_argument('-K', default=1, type=int, help='Number of local steps')
parser.add_argument('-M', default=100, type=int, help='Number of total clients')
parser.add_argument('-P', default=100, type=int, help='Number of participating clients')
parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer')
parser.add_argument('--lr', default=0.0, type=float, help='Client/Local learning rate')
parser.add_argument('--lr-scheduler', default='exp', type=str, help='exp/multistep')
parser.add_argument('--lr-decay', default=1.0, type=float, help='Learning rate decay')
parser.add_argument('--momentum', default=0.0, type=float, help='Momentum of client optimizer')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay of client optimizer')
parser.add_argument('--global-lr', default=1.0, type=float, help='Server/Global learning rate')
parser.add_argument('--seed', default=0, type=int, help='Seed')
parser.add_argument('--seeds', default=0, type=int, nargs='*', help='Seed')
parser.add_argument('--eval-every', default=1, type=int, help='Number of evaluations')
parser.add_argument('--device', default=0, type=int, help='Device')
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
args.device = device

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # Both settings are acceptable (yipeng, 2023-11-16)
    # https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.benchmark
    # torch.backends.cudnn.benchmark = False

def build_record_name(args):
    record_name = f'fl_{args.task}_{args.alg}_M{args.M},{args.P}_K{args.K}_R{args.R},{args.eval_every}'\
                + f'_{args.optim}{args.lr},{args.momentum},{args.weight_decay}_{args.lr_scheduler}{args.lr_decay}_seed{args.seed}'
    print(record_name)
    return record_name

def record_exp_result(filename, result):
    savepath = './save/'
    filepath = '{}/{}.csv'.format(savepath, filename)
    round, result = result[0], result[1:]
    if round == 0:
        if (os.path.exists(filepath)):
            os.remove(filepath)
        with open (filepath, 'a+') as f:
            f.write('{},{}\n'.format('round', ','.join(['param_error', 'order_error'])))
    else:
        with open (filepath, 'a+') as f:
            f.write('{},{}\n'.format(round, ','.join(['{:.32f}'.format(i) for i in result])))

class QuadraticFunc(nn.Module):
    def __init__(self, in_dim=1):
        global args
        super(QuadraticFunc, self).__init__()
        self.L = 0.5
        #self.x = torch.nn.Parameter(torch.randn((in_dim))[0])
        self.x = torch.nn.Parameter(torch.tensor(10, dtype=float))

    def forward(self, data=[1,1,1]):
        #out = data[0] * self.x**2 + data[1] * self.x + data[2]
        out = data[0] * self.x**2 + self.L * nn.functional.relu(-self.x) ** 2 + data[1] * self.x + data[2]
        return out

def read_gradient_vector(model):
    grad_list = []
    for param in model.parameters():
        if param.grad is not None:
            grad_list.append(param.grad.view(-1))  # Flatten each gradient tensor
    return torch.cat(grad_list)

def reorder(order, signs):
    positive_indices = []
    negative_indices = []

    for i in range(len(signs)):
        if signs[i] == +1:
            positive_indices.append(order[i])
        elif signs[i] == -1:
            negative_indices.append(order[i])
    new_order = positive_indices + negative_indices[::-1]
    return new_order

def balance(vectors):
    s = torch.zeros_like(vectors[0])
    
    size = len(list(vectors))
    signs = []
    for i in range(size):
        vec = vectors[i]
        #condition = torch.norm(s+vec) <= torch.norm(s-vec)
        condition = torch.max(torch.abs(s+vec)) <= torch.max(torch.abs(s-vec))
        if condition:
            s.add_(vec)
            signs.append(+1)
        else:
            s.sub_(vec)
            signs.append(-1)
    return signs

def basic_BR(order, vectors, vec_mean=None):
    # Center the vectors (a list)
    if vec_mean == None:
        vec_mean = torch.mean(torch.stack(vectors), dim = 0)
    centered_vectors = [vec.sub_(vec_mean) for vec in vectors]
    # Balance
    signs = balance(centered_vectors)
    # Reorder
    new_order = reorder(order, signs)
    return new_order

def pair_BR(order, vectors):
    # Centering is not required
    
    # Balance
    # Compute the difference
    diffs = []
    for i in range(len(vectors)//2):
        d = vectors[2*i] - vectors[2*i+1]
        diffs.append(d)
    # Assign the signs to the difference
    diff_signs = balance(diffs)
    # Assign the signs to the vectors
    signs = []
    for i in range(len(vectors)//2):
        signs.extend([diff_signs[i], -diff_signs[i]])
    
    # Reorder
    new_order = reorder(order, signs)
    return new_order
    
def compute_order_error(gradients):
    # Compute the order error
    max_error = 0.0
    s = None # partial sum
    num_clients = len(gradients)
    grad_mean = torch.mean(torch.stack(gradients), dim = 0)

    s = torch.zeros_like(gradients[0])
    for i in range(num_clients):   
        s.add_(gradients[i] - grad_mean)
        curr_error = float(torch.max(torch.abs(s)).cpu().numpy())
        #curr_curr_error = float(torch.norm(s).cpu().numpy())
        if  curr_error >= max_error:
            max_error = curr_error
    return max_error


def train_GraB_proto(args):
    args.alg = "GraB-proto"
    setup_seed(args.seed)
    
    device = args.device
    num_local_steps = args.K
    num_clients = args.M
    num_clients_per_round = args.P
    num_epochs = args.R
    global_lr = args.global_lr

    clients = get_clients(args.task, num_clients)
    clients.to(device)
    mean = torch.mean(clients, axis=0)
    optimum = - 0.5 * (mean[1] / mean[0])

    record_name = build_record_name(args)

    model = QuadraticFunc()
    model.train()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    record_exp_result(record_name, [0])
    #curr_indices = [i for i in range(num_clients)]
    curr_indices = np.random.choice(num_clients, size=num_clients, replace=False)
    for epoch in range(num_epochs):
        # Get the train_clients according to the order
        train_clients = clients[curr_indices]

        # Obtain the client gradients; compute the mean of the client gradients
        client_gradients = []
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for i in range(num_clients):
            objective = model(train_clients[i])
            optimizer.zero_grad()
            objective.backward()
            curr_gradient = read_gradient_vector(model)
            client_gradients.append(curr_gradient)
        
        # Compute the order eror
        order_error = compute_order_error(client_gradients)

        # Obtain the client gradients; compute the mean of the client gradients
        client_gradients = []
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for i in range(num_clients):
            objective = model(train_clients[i])
            optimizer.zero_grad()
            objective.backward()
            curr_gradient = read_gradient_vector(model)
            client_gradients.append(curr_gradient)
        
        # Produce the new order
        curr_indices = basic_BR(order=curr_indices, vectors=client_gradients)

        # Update the parameters
        round_model = copy.deepcopy(model)
        pseudo_gradients = []
        for i in range(num_clients):
            local_model = copy.deepcopy(round_model)
            local_optimizer = SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            for k in range(num_local_steps):
                local_objective = local_model(train_clients[i])
                local_optimizer.zero_grad()
                local_objective.backward()
                local_optimizer.step()
            pseudo_gradient = torch.nn.utils.parameters_to_vector(round_model.parameters()) - \
                                    torch.nn.utils.parameters_to_vector(local_model.parameters())
            # 2024-01-02. Note!! We need to detach it.
            # https://discuss.pytorch.org/t/runtimeerror-only-tensors-created-explicitly-by-the-user-graph-leaves-support-the-deepcopy-protocol-at-the-moment-when-deepcopying-a-model/142876/2
            # https://stackoverflow.com/q/56590886/16304578
            pseudo_gradients.append(pseudo_gradient.detach())
            if (i+1) % num_clients_per_round == 0:
                round_model_vector = torch.nn.utils.parameters_to_vector(round_model.parameters())
                round_model_vector = round_model_vector \
                                    - ( sum(pseudo_gradients[i+1-num_clients_per_round:i+1]) / num_clients_per_round )
                torch.nn.utils.vector_to_parameters(round_model_vector, round_model.parameters())
        model_vector = torch.nn.utils.parameters_to_vector(model.parameters())
        model_vector = model_vector - global_lr * ( sum(pseudo_gradients) / num_clients_per_round )
        torch.nn.utils.vector_to_parameters(model_vector, model.parameters())
        
        # Compute the param error
        with torch.no_grad():
            param_error = float(torch.norm(model.x.data - optimum).cpu().numpy())
            record_exp_result(record_name, [epoch+1, param_error, order_error])


def train_PairGraB_proto(args):
    args.alg = "PairGraB-proto"
    setup_seed(args.seed)
    
    device = args.device
    num_local_steps = args.K
    num_clients = args.M
    num_clients_per_round = args.P
    num_epochs = args.R
    global_lr = args.global_lr

    clients = get_clients(args.task, num_clients)
    clients.to(device)
    mean = torch.mean(clients, axis=0)
    optimum = - 0.5 * (mean[1] / mean[0])

    record_name = build_record_name(args)

    model = QuadraticFunc()
    model.train()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    record_exp_result(record_name, [0])
    #curr_indices = [i for i in range(num_clients)]
    curr_indices = np.random.choice(num_clients, size=num_clients, replace=False)
    for epoch in range(num_epochs):
        # Get the train_clients according to the order
        train_clients = clients[curr_indices]

        # Obtain the client gradients; compute the mean of the client gradients
        client_gradients = []
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for i in range(num_clients):
            objective = model(train_clients[i])
            optimizer.zero_grad()
            objective.backward()
            curr_gradient = read_gradient_vector(model)
            client_gradients.append(curr_gradient)
        
        # Compute the order eror
        order_error = compute_order_error(client_gradients)

        # Obtain the client gradients; compute the mean of the client gradients
        client_gradients = []
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for i in range(num_clients):
            objective = model(train_clients[i])
            optimizer.zero_grad()
            objective.backward()
            curr_gradient = read_gradient_vector(model)
            client_gradients.append(curr_gradient)
        
        # Produce the new order
        curr_indices = pair_BR(order=curr_indices, vectors=client_gradients)

        # Update the parameters
        round_model = copy.deepcopy(model)
        pseudo_gradients = []
        for i in range(num_clients):
            local_model = copy.deepcopy(round_model)
            local_optimizer = SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            for k in range(num_local_steps):
                local_objective = local_model(train_clients[i])
                local_optimizer.zero_grad()
                local_objective.backward()
                local_optimizer.step()
            pseudo_gradient = torch.nn.utils.parameters_to_vector(round_model.parameters()) - \
                                    torch.nn.utils.parameters_to_vector(local_model.parameters())
            pseudo_gradients.append(pseudo_gradient.detach())
            if (i+1) % num_clients_per_round == 0:
                round_model_vector = torch.nn.utils.parameters_to_vector(round_model.parameters())
                round_model_vector = round_model_vector \
                                    - ( sum(pseudo_gradients[i+1-num_clients_per_round:i+1]) / num_clients_per_round )
                torch.nn.utils.vector_to_parameters(round_model_vector, round_model.parameters())
        model_vector = torch.nn.utils.parameters_to_vector(model.parameters())
        model_vector = model_vector - global_lr * ( sum(pseudo_gradients) / num_clients_per_round )
        torch.nn.utils.vector_to_parameters(model_vector, model.parameters())
        
        # Compute the param error
        with torch.no_grad():
            param_error = float(torch.norm(model.x.data - optimum).cpu().numpy())
            record_exp_result(record_name, [epoch+1, param_error, order_error])




def train_GraB(args):
    args.alg = "GraB"
    setup_seed(args.seed)
    
    device = args.device
    num_local_steps = args.K
    num_clients = args.M
    num_clients_per_round = args.P
    num_epochs = args.R
    global_lr = args.global_lr

    clients = get_clients(args.task, num_clients)
    clients.to(device)
    mean = torch.mean(clients, axis=0)
    optimum = - 0.5 * (mean[1] / mean[0])

    record_name = build_record_name(args)

    model = QuadraticFunc()
    model.train()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    record_exp_result(record_name, [0])
    #curr_indices = [i for i in range(num_clients)]
    curr_indices = np.random.choice(num_clients, size=num_clients, replace=False)
    for epoch in range(num_epochs):
        # Get the train_clients according to the order
        train_clients = clients[curr_indices]

        # Obtain the client gradients; compute the mean of the client gradients
        client_gradients = []
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for i in range(num_clients):
            objective = model(train_clients[i])
            optimizer.zero_grad()
            objective.backward()
            curr_gradient = read_gradient_vector(model)
            client_gradients.append(curr_gradient)
        
        # Compute the order eror
        order_error = compute_order_error(client_gradients)

        # Update the parameters
        round_model = copy.deepcopy(model)
        pseudo_gradients = []
        for i in range(num_clients):
            local_model = copy.deepcopy(round_model)
            local_optimizer = SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            for k in range(num_local_steps):
                local_objective = local_model(train_clients[i])
                local_optimizer.zero_grad()
                local_objective.backward()
                local_optimizer.step()
            pseudo_gradient = torch.nn.utils.parameters_to_vector(round_model.parameters()) - \
                                    torch.nn.utils.parameters_to_vector(local_model.parameters())
            pseudo_gradients.append(pseudo_gradient.detach())
            if (i+1) % num_clients_per_round == 0:
                round_model_vector = torch.nn.utils.parameters_to_vector(round_model.parameters())
                round_model_vector = round_model_vector \
                                    - ( sum(pseudo_gradients[i+1-num_clients_per_round:i+1]) / num_clients_per_round )
                torch.nn.utils.vector_to_parameters(round_model_vector, round_model.parameters())
        model_vector = torch.nn.utils.parameters_to_vector(model.parameters())
        model_vector = model_vector - global_lr * ( sum(pseudo_gradients) / num_clients_per_round )
        torch.nn.utils.vector_to_parameters(model_vector, model.parameters())
        
        if epoch == 0:
            stale_gradient_mean = torch.zeros_like(pseudo_gradients[0])
        
        # Produce the new order
        # Note!! We use `copy.deepcopy()`. We use in-place centering in `basic_BR()`, 
        # which will change the values of the input `client_gradients`
        # Thus, we keep a copy of `client_gradients` to compute the stale mean in the next step.
        curr_indices = basic_BR(order=curr_indices, vectors=copy.deepcopy(pseudo_gradients), vec_mean=stale_gradient_mean)
        
        # Update the stale mean
        stale_gradient_mean = torch.mean(torch.stack(pseudo_gradients), dim = 0)
        
        # Compute the param error
        with torch.no_grad():
            param_error = float(torch.norm(model.x.data - optimum).cpu().numpy())
            record_exp_result(record_name, [epoch+1, param_error, order_error])


def train_PairGraB(args):
    # PairGraB
    args.alg = "PairGraB"
    setup_seed(args.seed)
    
    device = args.device
    num_local_steps = args.K
    num_clients = args.M
    num_clients_per_round = args.P
    num_epochs = args.R
    global_lr = args.global_lr

    clients = get_clients(args.task, num_clients)
    clients.to(device)
    #mean = torch.mean(clients, axis=0)
    #optimum = - 0.5 * (mean[1] / mean[0])
    optimum = 0

    record_name = build_record_name(args)

    model = QuadraticFunc()
    model.train()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    record_exp_result(record_name, [0])
    #curr_indices = [i for i in range(num_clients)]
    curr_indices = np.random.choice(num_clients, size=num_clients, replace=False)
    for epoch in range(num_epochs):
        # Get the train_clients according to the order
        train_clients = clients[curr_indices]

        # Obtain the client gradients; compute the mean of the client gradients
        client_gradients = []
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for i in range(num_clients):
            objective = model(train_clients[i])
            optimizer.zero_grad()
            objective.backward()
            curr_gradient = read_gradient_vector(model)
            client_gradients.append(curr_gradient)
        
        # Compute the order eror
        order_error = compute_order_error(client_gradients)        

        # Update the parameters
        round_model = copy.deepcopy(model)
        pseudo_gradients = []
        for i in range(num_clients):
            local_model = copy.deepcopy(round_model)
            local_optimizer = SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            for k in range(num_local_steps):
                local_objective = local_model(train_clients[i])
                local_optimizer.zero_grad()
                local_objective.backward()
                local_optimizer.step()
            pseudo_gradient = torch.nn.utils.parameters_to_vector(round_model.parameters()) - \
                                    torch.nn.utils.parameters_to_vector(local_model.parameters())
            pseudo_gradients.append(pseudo_gradient.detach())
            if (i+1) % num_clients_per_round == 0:
                round_model_vector = torch.nn.utils.parameters_to_vector(round_model.parameters())
                round_model_vector = round_model_vector \
                                    - ( sum(pseudo_gradients[i+1-num_clients_per_round:i+1]) / num_clients_per_round )
                torch.nn.utils.vector_to_parameters(round_model_vector, round_model.parameters())
        model_vector = torch.nn.utils.parameters_to_vector(model.parameters())
        model_vector = model_vector - global_lr * ( sum(pseudo_gradients) / num_clients_per_round )
        torch.nn.utils.vector_to_parameters(model_vector, model.parameters())
        
        # Produce the new order
        curr_indices = pair_BR(order=curr_indices, vectors=pseudo_gradients)

        # Compute the param error
        with torch.no_grad():
            param_error = float(torch.norm(model.x.data - optimum).cpu().numpy())
            record_exp_result(record_name, [epoch+1, param_error, order_error])

def train_RR(args):
    args.alg = "RR"
    setup_seed(args.seed)

    device = args.device
    num_local_steps = args.K
    num_clients = args.M
    num_clients_per_round = args.P
    num_epochs = args.R
    global_lr = args.global_lr

    clients = get_clients(args.task, num_clients)
    clients.to(device)
    #mean = torch.mean(clients, axis=0)
    #optimum = - 0.5 * (mean[1] / mean[0])
    optimum = 0

    record_name = build_record_name(args)

    model = QuadraticFunc()
    model.train()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    record_exp_result(record_name, [0])
    for epoch in range(num_epochs):
        curr_indices = np.random.choice(num_clients, size=num_clients, replace=False)
        train_clients = clients[curr_indices]

        # Obtain the client gradients; compute the mean of the client gradients
        client_gradients = []
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for i in range(num_clients):
            objective = model(train_clients[i])
            optimizer.zero_grad()
            objective.backward()
            curr_gradient = read_gradient_vector(model)
            client_gradients.append(curr_gradient)
        
        # Compute the order eror
        order_error = compute_order_error(client_gradients)

        # Update the parameters
        round_model = copy.deepcopy(model)
        pseudo_gradients = []
        for i in range(num_clients):
            local_model = copy.deepcopy(round_model)
            local_optimizer = SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            for k in range(num_local_steps):
                local_objective = local_model(train_clients[i])
                local_optimizer.zero_grad()
                local_objective.backward()
                local_optimizer.step()
            pseudo_gradient = torch.nn.utils.parameters_to_vector(round_model.parameters()) - \
                                    torch.nn.utils.parameters_to_vector(local_model.parameters())
            pseudo_gradients.append(pseudo_gradient.detach())
            if (i+1) % num_clients_per_round == 0:
                round_model_vector = torch.nn.utils.parameters_to_vector(round_model.parameters())
                round_model_vector = round_model_vector \
                                    - ( sum(pseudo_gradients[i+1-num_clients_per_round:i+1]) / num_clients_per_round )
                torch.nn.utils.vector_to_parameters(round_model_vector, round_model.parameters())
        model_vector = torch.nn.utils.parameters_to_vector(model.parameters())
        model_vector = model_vector - global_lr * ( sum(pseudo_gradients) / num_clients_per_round )
        torch.nn.utils.vector_to_parameters(model_vector, model.parameters())
        
        # Compute the param error
        with torch.no_grad():
            param_error = float(torch.norm(model.x.data - optimum).cpu().numpy())
            record_exp_result(record_name, [epoch+1, param_error, order_error])


def train_RR_FF(args):
    args.alg = "RR-FF"
    setup_seed(args.seed)

    device = args.device
    num_local_steps = args.K
    num_clients = args.M
    num_clients_per_round = args.P
    num_epochs = args.R
    global_lr = args.global_lr

    clients = get_clients(args.task, num_clients)
    clients.to(device)
    #mean = torch.mean(clients, axis=0)
    #optimum = - 0.5 * (mean[1] / mean[0])
    optimum = 0

    record_name = build_record_name(args)

    model = QuadraticFunc()
    model.train()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    record_exp_result(record_name, [0])
    for epoch in range(num_epochs):
        if epoch % 2 == 0:
            indices = np.random.choice(num_clients, size=num_clients, replace=False)
            train_clients = clients[indices]
        else:
            reverse_indices = indices[::-1].copy()
            train_clients = clients[reverse_indices]

        # Obtain the client gradients; compute the mean of the client gradients
        client_gradients = []
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for i in range(num_clients):
            objective = model(train_clients[i])
            optimizer.zero_grad()
            objective.backward()
            curr_gradient = read_gradient_vector(model)
            client_gradients.append(curr_gradient)
        
        # Compute the order eror
        order_error = compute_order_error(client_gradients)

        # Update the parameters
        round_model = copy.deepcopy(model)
        pseudo_gradients = []
        for i in range(num_clients):
            local_model = copy.deepcopy(round_model)
            local_optimizer = SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            for k in range(num_local_steps):
                local_objective = local_model(train_clients[i])
                local_optimizer.zero_grad()
                local_objective.backward()
                local_optimizer.step()
            pseudo_gradient = torch.nn.utils.parameters_to_vector(round_model.parameters()) - \
                                    torch.nn.utils.parameters_to_vector(local_model.parameters())
            pseudo_gradients.append(pseudo_gradient.detach())
            if (i+1) % num_clients_per_round == 0:
                round_model_vector = torch.nn.utils.parameters_to_vector(round_model.parameters())
                round_model_vector = round_model_vector \
                                    - ( sum(pseudo_gradients[i+1-num_clients_per_round:i+1]) / num_clients_per_round )
                torch.nn.utils.vector_to_parameters(round_model_vector, round_model.parameters())
        model_vector = torch.nn.utils.parameters_to_vector(model.parameters())
        model_vector = model_vector - global_lr * ( sum(pseudo_gradients) / num_clients_per_round )
        torch.nn.utils.vector_to_parameters(model_vector, model.parameters())
        
        # Compute the param error
        with torch.no_grad():
            param_error = float(torch.norm(model.x.data - optimum).cpu().numpy())
            record_exp_result(record_name, [epoch+1, param_error, order_error])


def train_SO(args):
    args.alg = "SO"
    setup_seed(args.seed)

    device = args.device
    num_local_steps = args.K
    num_clients = args.M
    num_clients_per_round = args.P
    num_epochs = args.R
    global_lr = args.global_lr

    clients = get_clients(args.task, num_clients)
    clients.to(device)
    #mean = torch.mean(clients, axis=0)
    #optimum = - 0.5 * (mean[1] / mean[0])
    optimum = 0

    record_name = build_record_name(args)

    model = QuadraticFunc()
    model.train()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    record_exp_result(record_name, [0])
    curr_indices = np.random.choice(num_clients, size=num_clients, replace=False)
    for epoch in range(num_epochs):
        train_clients = clients[curr_indices]

        # Obtain the client gradients; compute the mean of the client gradients
        client_gradients = []
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for i in range(num_clients):
            objective = model(train_clients[i])
            optimizer.zero_grad()
            objective.backward()
            curr_gradient = read_gradient_vector(model)
            client_gradients.append(curr_gradient)
        
        # Compute the order eror
        order_error = compute_order_error(client_gradients)

        # Update the parameters
        round_model = copy.deepcopy(model)
        pseudo_gradients = []
        for i in range(num_clients):
            local_model = copy.deepcopy(round_model)
            local_optimizer = SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            for k in range(num_local_steps):
                local_objective = local_model(train_clients[i])
                local_optimizer.zero_grad()
                local_objective.backward()
                local_optimizer.step()
            pseudo_gradient = torch.nn.utils.parameters_to_vector(round_model.parameters()) - \
                                    torch.nn.utils.parameters_to_vector(local_model.parameters())
            pseudo_gradients.append(pseudo_gradient.detach())
            if (i+1) % num_clients_per_round == 0:
                round_model_vector = torch.nn.utils.parameters_to_vector(round_model.parameters())
                round_model_vector = round_model_vector \
                                    - ( sum(pseudo_gradients[i+1-num_clients_per_round:i+1]) / num_clients_per_round )
                torch.nn.utils.vector_to_parameters(round_model_vector, round_model.parameters())
        model_vector = torch.nn.utils.parameters_to_vector(model.parameters())
        model_vector = model_vector - global_lr * ( sum(pseudo_gradients) / num_clients_per_round )
        torch.nn.utils.vector_to_parameters(model_vector, model.parameters())
        
        # Compute the param error
        with torch.no_grad():
            param_error = float(torch.norm(model.x.data - optimum).cpu().numpy())
            record_exp_result(record_name, [epoch+1, param_error, order_error])


def train_IG(args):
    args.alg = "IG"
    setup_seed(args.seed)

    device = args.device
    num_local_steps = args.K
    num_clients = args.M
    num_clients_per_round = args.P
    num_epochs = args.R
    global_lr = args.global_lr

    clients = get_clients(args.task, num_clients)
    clients.to(device)
    #mean = torch.mean(clients, axis=0)
    #optimum = - 0.5 * (mean[1] / mean[0])
    optimum = 0

    record_name = build_record_name(args)

    model = QuadraticFunc()
    model.train()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    record_exp_result(record_name, [0])
    curr_indices = np.array([i for i in range(num_clients)])
    for epoch in range(num_epochs):
        train_clients = clients[curr_indices]

        # Obtain the client gradients; compute the mean of the client gradients
        client_gradients = []
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for i in range(num_clients):
            objective = model(train_clients[i])
            optimizer.zero_grad()
            objective.backward()
            curr_gradient = read_gradient_vector(model)
            client_gradients.append(curr_gradient)
        
        # Compute the order eror
        order_error = compute_order_error(client_gradients)

        # Update the parameters
        round_model = copy.deepcopy(model)
        pseudo_gradients = []
        for i in range(num_clients):
            local_model = copy.deepcopy(round_model)
            local_optimizer = SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            for k in range(num_local_steps):
                local_objective = local_model(train_clients[i])
                local_optimizer.zero_grad()
                local_objective.backward()
                local_optimizer.step()
            pseudo_gradient = torch.nn.utils.parameters_to_vector(round_model.parameters()) - \
                                    torch.nn.utils.parameters_to_vector(local_model.parameters())
            pseudo_gradients.append(pseudo_gradient.detach())
            if (i+1) % num_clients_per_round == 0:
                round_model_vector = torch.nn.utils.parameters_to_vector(round_model.parameters())
                round_model_vector = round_model_vector \
                                    - ( sum(pseudo_gradients[i+1-num_clients_per_round:i+1]) / num_clients_per_round )
                torch.nn.utils.vector_to_parameters(round_model_vector, round_model.parameters())
        model_vector = torch.nn.utils.parameters_to_vector(model.parameters())
        model_vector = model_vector - global_lr * ( sum(pseudo_gradients) / num_clients_per_round )
        torch.nn.utils.vector_to_parameters(model_vector, model.parameters())
        
        # Compute the param error
        with torch.no_grad():
            param_error = float(torch.norm(model.x.data - optimum).cpu().numpy())
            record_exp_result(record_name, [epoch+1, param_error, order_error])


def train_cSGD(args):
    # Basic
    args.alg = "cSGD"
    setup_seed(args.seed)

    device = args.device
    num_local_steps = args.K
    num_clients = args.M
    num_clients_per_round = args.P
    num_epochs = args.R
    global_lr = args.global_lr

    clients = get_clients(args.task, num_clients)
    clients.to(device)
    #mean = torch.mean(clients, axis=0)
    #optimum = - 0.5 * (mean[1] / mean[0])
    optimum = 0

    record_name = build_record_name(args)

    model = QuadraticFunc()
    model.train()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    record_exp_result(record_name, [0])
    for epoch in range(num_epochs):
        curr_indices = np.random.choice(num_clients, size=num_clients, replace=True)
        train_clients = clients[curr_indices]

        # Obtain the client gradients; compute the mean of the client gradients
        client_gradients = []
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for i in range(num_clients):
            objective = model(train_clients[i])
            optimizer.zero_grad()
            objective.backward()
            curr_gradient = read_gradient_vector(model)
            client_gradients.append(curr_gradient)
        
        # Compute the order eror
        order_error = compute_order_error(client_gradients)

        # Update the parameters
        round_model = copy.deepcopy(model)
        pseudo_gradients = []
        for i in range(num_clients):
            local_model = copy.deepcopy(round_model)
            local_optimizer = SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            for k in range(num_local_steps):
                local_objective = local_model(train_clients[i])
                local_optimizer.zero_grad()
                local_objective.backward()
                local_optimizer.step()
            pseudo_gradient = torch.nn.utils.parameters_to_vector(round_model.parameters()) - \
                                    torch.nn.utils.parameters_to_vector(local_model.parameters())
            pseudo_gradients.append(pseudo_gradient.detach())
            if (i+1) % num_clients_per_round == 0:
                round_model_vector = torch.nn.utils.parameters_to_vector(round_model.parameters())
                round_model_vector = round_model_vector \
                                    - ( sum(pseudo_gradients[i+1-num_clients_per_round:i+1]) / num_clients_per_round )
                torch.nn.utils.vector_to_parameters(round_model_vector, round_model.parameters())
        model_vector = torch.nn.utils.parameters_to_vector(model.parameters())
        model_vector = model_vector - global_lr * ( sum(pseudo_gradients) / num_clients_per_round )
        torch.nn.utils.vector_to_parameters(model_vector, model.parameters())
        
        # Compute the param error
        with torch.no_grad():
            param_error = float(torch.norm(model.x.data - optimum).cpu().numpy())
            record_exp_result(record_name, [epoch+1, param_error, order_error])


def dump_clients(name, clients):
    path = 'maps/'
    clients = clients.cpu().numpy()
    clients = clients.tolist()
    path = os.path.join(path, name)
    with open(path, 'w') as f:
        json.dump(clients, f)

def load_clients(name):
    path = 'maps/'
    path = os.path.join(path, name)
    with open(path, 'r') as f:
        clients = json.load(f)
    clients = torch.tensor(clients)
    return clients

def gen_clients(task, num_clients):
    if task == 'quadh0': # hessian change
        mean = 0; std_dev = 1
        coef1 = np.random.normal(0.5, std_dev, num_clients)
        coef2 = np.array([1 for i in range(num_clients//2)] + [-1 for i in range(num_clients//2)])
    elif task == 'quadh1': # hessian change
        mean = 0; std_dev = 1
        coef1 = np.random.normal(0.5, std_dev, num_clients)
        temp = np.random.normal(mean, std_dev, num_clients//2)
        coef2 = np.concatenate((temp, -temp))
        coef2 = np.sort(coef2)
    else:
        raise ValueError
    
    clients = []
    for i in range(num_clients):
        client = torch.tensor([coef1[i], coef2[i], 0])
        clients.append(client)
    clients = torch.stack(clients, dim=0)
    file = "{}_M{}".format(task, num_clients)
    dump_clients(file, clients)
    
    return clients
    
def get_clients(task, num_clients):
    path = 'maps/'
    file = "{}_M{}".format(task, num_clients)
    if os.path.exists(os.path.join(path, file)):
        print('Load the clients.')
        clients = load_clients(file)
    else:
        print('No existing clients. So we generate one.')
        clients = gen_clients(task, num_clients)
    return clients

def main():
    global args

    #lrs = [0.0001, 0.000316, 0.001, 0.00316, 0.01, 0.0316, 0.1, 0.316, 1.0]
    lrs = [0.0001, 0.000316]
    seeds = args.seeds

    for lr in lrs:
        args.lr = lr
        for seed in seeds:
            args.seed = seed    
            train_cSGD(args)
            train_IG(args)
            train_SO(args)
            train_RR(args)
            train_RR_FF(args)
            train_GraB_proto(args)
            train_PairGraB_proto(args)
            train_GraB(args)
            train_PairGraB(args)

if __name__ == '__main__':
    main()
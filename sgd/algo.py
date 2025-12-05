import torch
import copy
import random
#from sklearn import random_projection
from .utils import flatten_grad

class Sort:
    def sort(self, orders):
        raise NotImplementedError

'''2025-01-06. Comments on `orders`.
The function of `orders`. A map of the batch id to its order.
For instance, let `orders = { 2:1, 0:2, 1:3, 3:0 }`.
When it performs `sort`, it will be { 3:0, 2:1, 0:2, 1:3 }, it uses the keys [3,2,0,1] as the training order.
'''
class GraBSort(Sort):
    """
    Implementation of the GraB algorithm, which uses stale gradient to sort the examples
        via minimizing the discrepancy bound. The details can be found in:
        https://arxiv.org/abs/2205.10733.
        
    """
    def __init__(self,args,num_batches,grad_dimen,device):
        self.args = args
        self.num_batches = num_batches
        self.grad_dimen = grad_dimen
        self.avg_grad = torch.zeros(grad_dimen).to(device)
        self.cur_sum = torch.zeros_like(self.avg_grad)
        self.next_epoch_avg_grad = torch.zeros_like(self.avg_grad)
        self.orders = {i:0 for i in range(self.num_batches)}
        self.first = 0
        self.last = self.num_batches
    
    def _skip_sort_this_epoch(self, epoch):
        return epoch <= self.args.start_sort
    
    def sort(self):
        self.orders = {k: v for k, v in sorted(self.orders.items(), key=lambda item: item[1], reverse=False)}
        self.avg_grad.copy_(self.next_epoch_avg_grad)
        self.next_epoch_avg_grad.zero_()
        self.cur_sum.zero_()
        self.first = 0
        self.last = self.num_batches
        return self.orders

    def step(self, optimizer, batch_idx):
        cur_grad = flatten_grad(optimizer)
        self.next_epoch_avg_grad.add_(cur_grad / self.num_batches)
        cur_grad.add_(-1 * self.avg_grad)
        # The balancing algorithm used here is described in Algorithm 5 in 
        #   https://arxiv.org/abs/2205.10733. We can always replace it with other balancing variants.
        if torch.norm(self.cur_sum + cur_grad) <= torch.norm(self.cur_sum - cur_grad):
            self.orders[batch_idx] = self.first
            self.first += 1
            self.cur_sum.add_(cur_grad)
        else:
            self.orders[batch_idx] = self.last
            self.last -= 1
            self.cur_sum.add_(-1 * cur_grad)


class PairGraBSort(Sort):
    """
    PairGraB. See https://github.com/EugeneLYC/GraB/tree/main.
    """
    def __init__(self,args,num_batches,grad_dimen,device):
        self.args = args
        self.num_batches = num_batches
        self.grad_dimen = grad_dimen
        self.prev_grad = torch.zeros(grad_dimen).to(device) # store the gradient of the previous batch temporarily
        self.prev_batch_idx = 0 # store the index of the previous batch temporarily
        self.cur_sum = torch.zeros_like(self.prev_grad)
        self.orders = {i:0 for i in range(self.num_batches)}
        self.batch_counter = 0
        self.first = 0
        self.last = self.num_batches
    
    def _skip_sort_this_epoch(self, epoch):
        return epoch <= self.args.start_sort
    
    def sort(self):
        # sort according to the value of the second item, in ascending order. 
        sorted_orders = sorted(self.orders.items(), key=lambda item: item[1], reverse=False)
        self.orders = {k: v for k, v in sorted_orders}
        self.cur_sum.zero_()
        self.batch_counter = 0
        self.first = 0
        self.last = self.num_batches
        return self.orders

    def step(self, optimizer, batch_idx):
        cur_grad = flatten_grad(optimizer)
        # 2024-01-06. When the number of batches is odd, there is only one position left.
        # Thus, the position of the last batch is determined automatically.
        if self.first == self.last:
            self.orders[batch_idx] = self.first
        if self.batch_counter % 2 == 0:
            self.prev_grad.copy_(cur_grad)
            self.prev_batch_idx = batch_idx
            self.batch_counter += 1
        else:
            diff_grad = self.prev_grad - cur_grad
            if torch.norm(self.cur_sum + diff_grad) <= torch.norm(self.cur_sum - diff_grad):
                sign = +1
                self.orders[self.prev_batch_idx] = self.first
                self.orders[batch_idx] = self.last
                
            else:
                sign = -1
                self.orders[self.prev_batch_idx] = self.last
                self.orders[batch_idx] = self.first
            self.first += 1
            self.last  -= 1
            self.cur_sum.add_(diff_grad * sign)
            self.batch_counter += 1


class FlipFlopSort(Sort):
    def __init__(self,args,num_batches):
        self.args = args
        self.num_batches = num_batches
        self.orders = {i:0 for i in range(self.num_batches)}
    
    def sort(self, epoch):
        '''2025-07-28. One example.
        if epoch % 2 == 0, then,
        self.orders = { 2:0, 0:1, 1:2, 3:3 }, original
        self.orders = { 2:0, 0:1, 1:2, 3:3 }, after sorted
        training order = 2,0,1,3

        if epoch % 2 != 0, then,
        self.orders = { 2:0, 0:1, 1:2, 3:3 }, original
        self.orders = { 3:3, 1:2, 0:1, 2:0 }, after sorted
        training order = 3,1,0,2
        '''
        if epoch % 2 == 0:
            idx_list = [i for i in range(self.num_batches)]
            idx_list_copy = [i for i in range(self.num_batches)]
            random.shuffle(idx_list)
            self.orders = {i:j for i, j in zip(idx_list, idx_list_copy)}
            self.orders = {k: v for k, v in sorted(self.orders.items(), key=lambda item: item[1], reverse=False)}
        else:
            self.orders = {k: v for k, v in sorted(self.orders.items(), key=lambda item: item[1], reverse=True)}
        return self.orders


import torch

class OptimKit(object):
    def __init__(self, optim_name, batch_size, **settings):
        if optim_name == 'sgd':
            setattr(self, 'optim', torch.optim.SGD)
        elif optim_name == 'adam':
            setattr(self, 'optim', torch.optim.SGD)
        self.batch_size = batch_size
        self.settings = settings
        self.lr_updater = None

    def setup_lr_updater(self, lr_updater, **kwargs):
        self.lr_updater = lr_updater
        self.lr_updater_settings = kwargs
    
    def update_lr(self, round):
        if self.lr_updater != None:
            self.settings['lr'] = self.lr_updater(self.settings['lr'], round, **self.lr_updater_settings)


class LrUpdater():
    @staticmethod
    def exponential_lr_updater(lr, round, mul):
        return lr * mul
    
    @staticmethod
    def multistep_lr_updater(lr, round, mul, total_rounds):
        if round == int(total_rounds*0.5):
            return lr * mul
        elif round == int(total_rounds*0.8):
            return lr * mul
        else:
            return lr
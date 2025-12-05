import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset

class BaseDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target


class FedDataset(object):
    def __init__(self, origin_dataset, partition_map):
        self.origin_dataset = origin_dataset
        self.map = partition_map
        self.num_datasets = len(self.map)
        self.fedsetsizes = [len(self.map[i]) for i in range(self.num_datasets)]
        self.totalsize = sum(self.fedsetsizes)
        
        # federated training set
        self.fedsets = []
        trainset = self.origin_dataset.get_trainset(transform=self.origin_dataset.transform_train)
        for i in range(len(self.map)):
            self.fedsets.append(Subset(trainset, self.map[i]))
        
        # evaluating set
        trainset = self.origin_dataset.get_trainset(transform=self.origin_dataset.transform_test)
        testset = self.origin_dataset.get_testset(transform=self.origin_dataset.transform_test)
        self.eval_trainset = BaseDataset([trainset[j] for j in range(len(trainset))])
        self.eval_testset = BaseDataset([testset[j] for j in range(len(testset))])
    
    def get_map(self, id):
        return self.map[id]
    
    def get_dataset(self, id):
        return self.fedsets[id]
    
    def get_datasetsize(self, id):
        return self.fedsetsizes[id]
    
    def get_eval_trainset(self):
        return self.eval_trainset

    def get_eval_testset(self):
        return self.eval_testset
    
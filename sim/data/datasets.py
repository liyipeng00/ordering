'''Datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, CINIC-10
Note: In FL, it is impossible to get the mean and std of the whole training set (yipeng, 2023-11-16)
'''
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageFolder
import torchvision.transforms as transforms

def build_dataset(dataset_name='mnist', dataset_dir = '../datasets/'):
    if dataset_name == 'mnist':
        origin_dataset = OriginMNIST()
    elif dataset_name == 'fashionmnist':
        origin_dataset = OriginFashionMNIST()
    elif dataset_name == 'cifar10':
        origin_dataset = OriginCIFAR10()
    elif dataset_name == 'cifar100':
        origin_dataset = OriginCIFAR100()
    elif dataset_name == 'cinic10':
        origin_dataset = OriginCINIC10()
    return origin_dataset


class OriginMNIST():
    '''
    https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/utils.py
    '''
    def __init__(self, root='../datasets/'):
        self.root = root
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    
    def get_trainset(self, transform=None):
        trainset = MNIST(root=self.root, train=True, download=True, transform=transform)
        return trainset
    
    def get_testset(self, transform=None):
        testset = MNIST(root=self.root, train=False, download=True, transform=transform)
        return testset


class OriginFashionMNIST():
    '''
    Calculate the mean and std of the training dataset manually
    Some links also use `mean=0.1307, std=0.3081` (MNIST) [1] or `mean=0.5, std=0.5` [2]
    [1] https://github.com/IBM/fl-arbitrary-participation/blob/main/dataset/dataset.py
    [2] https://github.com/Divyansh03/FedExP/blob/main/util_data.py
    '''
    def __init__(self, root='../datasets/'):
        self.root = root
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
            ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
            ])
    
    def get_trainset(self, transform=None):
        trainset = FashionMNIST(root=self.root, train=True, download=True, transform=transform)
        return trainset
    
    def get_testset(self, transform=None):
        testset = FashionMNIST(root=self.root, train=False, download=True, transform=transform)
        return testset
    

class OriginCIFAR10():
    '''https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/cifar10/data_loader.py'''
    def __init__(self, root='../datasets/'):
        self.root = root
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    def get_trainset(self, transform=None):
        trainset = CIFAR10(root=self.root, train=True, download=True, transform=transform)
        return trainset
    
    def get_testset(self, transform=None):
        testset = CIFAR10(root=self.root, train=False, download=True, transform=transform)
        return testset


class OriginCIFAR100():
    '''https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/cifar100/data_loader.py'''
    def __init__(self, root='../datasets/'):
        self.root = root
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    
    def get_trainset(self, transform=None):
        trainset = CIFAR100(root=self.root, train=True, download=True, transform=transform)
        return trainset
    
    def get_testset(self, transform=None):
        testset = CIFAR100(root=self.root, train=False, download=True, transform=transform)
        return testset


class OriginCINIC10():
    '''
    https://github.com/BayesWatch/cinic-10
    https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/cinic10/data_loader.py
    '''
    def __init__(self, root='../datasets/'):
        self.root = root
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def get_trainset(self, transform=None):
        trainset = ImageFolder('{}/{}'.format(self.root, '/CINIC-10/train/'), transform=transform)
        return trainset
    
    def get_testset(self, transform=None):
        testset = ImageFolder('{}/{}'.format(self.root, '/CINIC-10/test/'), transform=transform)
        return testset





if __name__ == '__main__':
    pass
    # for i in range(0, 3):
    #     # to judge if the sample sequence is the same at different times
    #     train_dataset, test_dataset = dataset_mnist('../datasets/')
    #     print(train_dataset.targets[:30])
   
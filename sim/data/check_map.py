'''Three partition strategies are included: IID, Dir, ExDir.
'''
import os
from partition import Partitioner, build_partition

# python sim/data/check_map.py -d mnist -n 10 --partition exdir -C 1 --alpha 1.0 
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='mnist', help='dataset name')
    parser.add_argument('-n', type=int, default=100, help='divide into n clients')
    parser.add_argument('--partition', type=str, default='iid', help='iid')
    parser.add_argument('--balance', type=bool, default=True, help='balanced or imbalanced')
    parser.add_argument('--alpha', type=float, default=1.0, help='the alpha of dirichlet distribution')
    parser.add_argument('-C', type=int, default=1, help='the classes of pathological partition')
    args = parser.parse_args()
    print(args)
    
    dataset_dir = '../datasets/' # the directory path of datasets
    output_dir = 'maps/raw/' # the directory path of outputs
    dataset_name = args.d # the name of the dataset
    num_clients = args.n # number of clients
    partition = args.partition # partition way
    balance = args.balance
    alpha = args.alpha
    C = args.C

    # Prepare the dataset
    num_class_dict = { 'mnist': 10, 'fashionmnist': 10, 'cifar10': 10, 'cifar100': 100, 'cinic10': 10, 'test': 4 }
    origin_dataset = build_dataset(dataset_name=dataset_name, dataset_dir=dataset_dir)
    train_dataset = origin_dataset.get_trainset()
    # if partitioning the trainining set merely
    labels = list(train_dataset.targets) # Note: `train_dataset.targets` is a list in cifar10/100 , but a tensor in mnist (yipeng, 2023-04-26)
    # if partitioning the whold set (including training set and test set)
    #label_list = list(train_dataset.targets) + list(test_dataset.targets)
    num_classes = num_class_dict[dataset_name]

    dataidx_map = build_partition(dataset_name=dataset_name, num_clients=num_clients, partition=partition, alpha=[C, alpha])
    
    Partitioner.check_dataidx_map(dataidx_map, labels, num_clients, num_classes)

if __name__ == '__main__':
    from datasets import build_dataset
    main()
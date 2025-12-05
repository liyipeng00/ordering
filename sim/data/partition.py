'''Three partition strategies are included: IID, Dir, ExDir.
'''
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

CHECK = True

def build_partition(dataset_name='mnist', num_clients=10, partition='iid', alpha=[]):
    r"""Build local data distributions to assign data sample indices to clients.
    Args:
        num_clients (int): number of clients
        partition (str): partition way, e.g., iid, dir and exdir
        alpha (list): parameters of partition ways
    
    Returns:
        dataidx_map (dict): { client id (int): data indices (numpy.ndarray) }, e.g., {0: [0,1,4], 1: [2,3,5]}
    """
    map_dir = './maps/' # map directory
    if partition == 'iid':
        map_path = "{}/{}_M{}_{}.txt".format(map_dir, dataset_name, num_clients, partition)
    elif partition == 'dir':
        alpha = alpha[0]
        map_path = "{}/{}_M{}_{}{}.txt".format(map_dir, dataset_name, num_clients, partition, alpha)
    elif partition == 'exdir':
        C = int(alpha[0]); alpha = alpha[1]
        map_path = "{}/{}_M{}_{}{},{}.txt".format(map_dir, dataset_name, num_clients, partition, C, alpha)
    elif partition == 'exdirb':
        C = int(alpha[0]); alpha = alpha[1]
        map_path = "{}/{}_M{}_{}{},{}.txt".format(map_dir, dataset_name, num_clients, partition, C, alpha)
    elif partition == 'shard':
        C = int(alpha[0])
        map_path = "{}/{}_M{}_{}{}.txt".format(map_dir, dataset_name, num_clients, partition, C)
    else:
        raise ValueError
    dataidx_map = Partitioner.read_dataidx_map(map_path)
    return dataidx_map


class Partitioner():
    def __init__(self):
        pass

    def partition_data(self):
        r"""Partition data indices to clients.
        Returns:
            dataidx_map (dict): { client id (int): data indices (numpy.ndarray) }, e.g., {0: [0,1,4], 1: [2,3,5]}
        """
        pass
    
    def gen_dataidx_map(self, labels, num_clients, num_classes, map_dir):
        r"""Generate dataidx_map"""
        dataidx_map = self.partition_data(labels, num_clients, num_classes)

        # Check the dataidx_map
        if CHECK == True:
            self.check_dataidx_map(dataidx_map, labels, num_clients, num_classes)
        map_path = "{}/{}_M{}_{}.txt".format(map_dir, self.dataset_name, num_clients, self.output_name)
        self.dumpmap(dataidx_map, map_path)

    @classmethod
    def read_dataidx_map(self, map_path):
        dataidx_map = self.loadmap(map_path)
        return dataidx_map
    
    @classmethod
    def check_dataidx_map(cls, dataidx_map=None, labels=None, num_clients=10, num_classes=10):
        r"""Check whether the map is reasonable by displaying some map information.
        Args:
            labels (numpy.ndarray, list): labels of the whole dataset
        """
        # Count the number of data samples per class per client
        n_sample_per_class_per_client = { cid: [] for cid in range(num_clients) } # cid: client id
        for cid in range(num_clients):
            # number of data samples per class of any one client
            n_sample_per_class_one_client = [ 0 for _ in range(num_classes) ]
            for j in range(len(dataidx_map[cid])):
                n_sample_per_class_one_client[int(labels[dataidx_map[cid][j]])] += 1
            n_sample_per_class_per_client[cid] = n_sample_per_class_one_client
        print("\n***** the number of samples per class per client *****")
        print(n_sample_per_class_per_client)

        # Count the number of samples per client
        n_sample_per_client = []
        for cid in range(num_clients):
            n_sample_per_client.append(sum(n_sample_per_class_per_client[cid]))
        n_sample_per_client = np.array(n_sample_per_client)
        print("\n***** the number of samples per client *****")
        #print(n_sample_per_client.mean(), n_sample_per_client.std())
        print(n_sample_per_client)

        # Count the number of samples per label
        n_sample_per_label = []
        n_client_per_label = []
        for i in range(num_classes):
            n_s = 0 # number of samples of any one label
            n_c = 0 # number of clients of any one label
            for j in range(num_clients):
                n_s = n_s + n_sample_per_class_per_client[j][i]
                n_c = n_c + int(n_sample_per_class_per_client[j][i] != 0)
            n_sample_per_label.append(n_s)
            n_client_per_label.append(n_c)
        n_sample_per_label = np.array(n_sample_per_label)
        n_client_per_label = np.array(n_client_per_label)
        print("\n*****the number of samples per label*****")
        print(n_sample_per_label)
        print("\n*****the number of clients per label*****")
        #print(n_client_per_label.mean(), n_client_per_label.std())
        print(n_client_per_label)
        
        cls.bubble(n_sample_per_class_per_client, num_clients, num_classes)
        #cls.heatmap(n_sample_per_class_per_client, num_clients, num_classes)

    @classmethod
    def bubble(cls, n_sample_per_class_per_client, num_clients, num_classes):
        r"""Draw bubble chart to display the local data distribution.
        Args:
            n_sample_per_class_per_client (set): { client id: [number of samples of Class 0, number of samples of Class 1, ...] } 
        """
        x = []
        for i in range(num_clients):
            x.extend([i for _ in range(num_classes)])

        y = []
        for i in range(num_clients):
            y.extend([j for j in range(num_classes)])

        size = []
        for i in range(len(x)):
            size.append(n_sample_per_class_per_client[x[i]][y[i]])
        size = [i*0.2 for i in size]

        plt.figure()
        plt.scatter(x, y, s=size, alpha=1)
        #plt.title(title)
        plt.xlabel("Client ID")
        plt.ylabel("Label")
        #plt.savefig('./raw_partition/{}/{}.png'.format(dataset, title))
        plt.show()
    
    @classmethod
    def heatmap(cls, n_sample_per_class_per_client, num_clients, num_classes):
        r"""Draw heat map to display the local data distribution"""
        num_sample_per_client = []
        heatmap_data = np.zeros((num_classes, num_clients), int)
        for i in range(num_clients):
            heatmap_data[:,i] = np.array(n_sample_per_class_per_client[i])
            num_sample_per_client.append(sum(n_sample_per_class_per_client[i]))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = sns.heatmap(heatmap_data, ax=ax, annot=True, fmt="d", linewidths=.9, cmap="YlGn",) #
        ax.set_xticklabels(['{}'.format(i) for i in range(num_clients)], rotation=0)
        #ax.set_xticklabels(['{} ({})'.format(i, num_sample_per_client[i]) for i in range(num_clients)], rotation=0)
        ax.set_yticklabels([str(i) for i in range(num_classes)], rotation=0)
        ax.set_xlabel("Client ID", fontsize=15)
        ax.set_ylabel("Label", fontsize=15)
        #ax.set_title(title, fontsize=16)
        #plt.savefig('./raw_partition/{}/{}.png'.format(dataset, title), bbox_inches='tight')
        plt.show()
    
    @classmethod
    def dumpmap(cls, dataidx_map, map_path):
        for i in range(len(dataidx_map)):   
            if isinstance(dataidx_map[i], list) == False:
                dataidx_map[i] = dataidx_map[i].tolist()
        with open(map_path, 'w') as f:
            json.dump(dataidx_map, f)
    
    @classmethod
    def loadmap(cls, map_path):
        with open(map_path, 'r') as f:
            temp = json.load(f)
        # Since `json.load` will form dict{ '0': [] }, instead of dict{ 0: [] },
        # we need to turn dict{ '0': [] } to dict{ 0: [] }
        dataidx_map = dict()
        for i in range(len(temp)):
            dataidx_map[i] = np.array(temp[str(i)]) 
        return dataidx_map


class IIDPartitioner(Partitioner):
    r"""https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py"""
    def __init__(self, dataset_name='mnist'):
        super(IIDPartitioner, self).__init__()
        self.name = 'iid'
        self.dataset_name = dataset_name
        self.output_name = self.name
    
    def partition_data(self, labels, num_clients, num_classes):
        # Note: now 'balance' is ready, 'unbalance' is not completed (yipeng, 2023-11-14)
        num_labels = len(labels)
        idxs = np.random.permutation(num_labels)
        client_idxs = np.array_split(idxs, num_clients)
        dataidx_map = { cid: client_idxs[cid] for cid in range(num_clients) }
        return dataidx_map


class DirPartitioner(Partitioner):
    r"""The implementation of Dir-paritition way is from
    https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
    """
    def __init__(self, dataset_name='mnist', alpha=10.0):
        super(DirPartitioner, self).__init__()
        self.name = 'dir'
        self.dataset_name = dataset_name
        self.alpha = alpha
        self.output_name = '{}{}'.format(self.name, self.alpha)
        
    def partition_data(self, labels, num_clients, num_classes):
        alpha = self.alpha
        min_size = 0
        min_require_size = 10 # the minimum size of samples per client is required to be 10 
        num_labels = len(labels)
        labels = np.array(labels) # Note: to make `np.where(labels == k)[0]` succesful, turn labels to `np.ndarray` (yipeng, 2023-11-14)
        
        while min_size < min_require_size:
            idx_per_client = [[] for _ in range(num_clients)] # data sample indices per client
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0] # data sample indices of class k
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Note: Balance the number of data samples of clients.
                # Don't assign samples to client j when its number of data samples is larger than the average (yipeng, 2023-11-14)
                proportions = np.array([p * (len(idx_j) < num_labels / num_clients) for p, idx_j in zip(proportions, idx_per_client)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_per_client = [idx_j + idx.tolist() for idx_j, idx in zip(idx_per_client, np.split(idx_k, proportions))] 
                min_size = min([len(idx_j) for idx_j in idx_per_client])

        dataidx_map = {}
        for j in range(num_clients):
            np.random.shuffle(idx_per_client[j])
            dataidx_map[j] = idx_per_client[j]
        return dataidx_map


class ExDirPartitioner(Partitioner):
    def __init__(self, dataset_name='mnist', C=10, alpha=10.0):
        super(ExDirPartitioner, self).__init__()
        self.name = 'exdir'
        self.dataset_name = dataset_name
        self.C, self.alpha = C, alpha
        self.output_name = '{}{},{}'.format(self.name, self.C, self.alpha)
        
    def allocate_classes(self, num_clients, num_classes, p_classes):
        '''Allocate `C` classes to each client
        Returns:
            clientidx_map (dict): { class id (int): client indices (list) }
        '''
        min_size_per_class = 0
        #min_require_size_per_class = max(self.C * num_clients // num_classes // 5, 1)
        min_require_size_per_class = 1
        while min_size_per_class < min_require_size_per_class:
            clientidx_map = { k: [] for k in range(num_classes) }
            for cid in range(num_clients):
                slected_classes = np.random.choice(range(num_classes), self.C, replace=False, p=p_classes)
                for k in slected_classes:
                    clientidx_map[k].append(cid)
            min_size_per_class = min([len(clientidx_map[k]) for k in range(num_classes)])
        return clientidx_map
    
    def partition_data(self, labels, num_clients, num_classes):
        C, alpha = self.C, self.alpha
        labels = np.array(labels)
        min_size = 0
        min_require_size = 1 # 10
        num_examples = len(labels)
        
        """2024-11-08. Dealing with Logistic Regression in JMLR.
        If the classes are allocated with equal probability,
        the number of examples will vary largely,
        when meeting the probability of each class varies largely
        """
        # p_classes = [] # compute the probability of each class
        # for i in range(num_classes):
        #     p = len(np.where(labels == i)[0])
        #     p_classes.append(p / num_examples)
        # print(p_classes)
        # we provide the previous setting as a choice. 2024-11-10.
        p_classes = [1/num_classes for _ in range(num_classes)]

        clientidx_map = self.allocate_classes(num_clients, num_classes, p_classes)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[cid]) for cid in range(num_classes)])

        while min_size < min_require_size:
            idx_per_client = [[] for _ in range(num_clients)]
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Case 1 (original case in Dir): Balance
                proportions = np.array([p * (len(idx_j) < num_examples / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_per_client))])
                # Case 2: Don't balance
                #proportions = np.array([p * (j in label_netidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_per_client))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                '''Note: Process the remainder data samples (yipeng, 2023-11-14).
                There are some cases that the samples of class k are not allocated completely, i.e., proportions[-1] < len(idx_k)
                In these cases, the remainder data samples are assigned to the last client in `clientidx_map[k]`.
                '''
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], num_clients-1):
                        proportions[w] = len(idx_k)
                
                idx_per_client = [idx_j + idx.tolist() for idx_j, idx in zip(idx_per_client, np.split(idx_k, proportions))] 
                min_size = min([len(idx_j) for idx_j in idx_per_client])
        
        dataidx_map = {}
        for j in range(num_clients):
            np.random.shuffle(idx_per_client[j])
            dataidx_map[j] = idx_per_client[j]
        return dataidx_map


class ExDirBPartitioner(Partitioner):
    def __init__(self, dataset_name='mnist', C=10, alpha=10.0):
        super(ExDirBPartitioner, self).__init__()
        self.name = 'exdirb'
        self.dataset_name = dataset_name
        self.C, self.alpha = C, alpha
        self.output_name = '{}{},{}'.format(self.name, self.C, self.alpha)
        
    def allocate_classes(self, num_clients, num_classes, p_classes):
        '''Allocate `C` classes to each client
        Returns:
            clientidx_map (dict): { class id (int): client indices (list) }
        '''
        min_size_per_class = 0
        #min_require_size_per_class = max(self.C * num_clients // num_classes // 5, 1)
        min_require_size_per_class = 1
        while min_size_per_class < min_require_size_per_class:
            clientidx_map = { k: [] for k in range(num_classes) }
            for cid in range(num_clients):
                slected_classes = np.random.choice(range(num_classes), self.C, replace=False, p=p_classes)
                for k in slected_classes:
                    clientidx_map[k].append(cid)
            min_size_per_class = min([len(clientidx_map[k]) for k in range(num_classes)])
        return clientidx_map
    
    def partition_data(self, labels, num_clients, num_classes):
        C, alpha = self.C, self.alpha
        labels = np.array(labels)
        min_size = 0
        min_require_size = 1 # 10
        num_examples = len(labels)
        
        """2024-11-08. Dealing with Logistic Regression in JMLR.
        If the classes are allocated with equal probability,
        the number of examples will vary largely,
        when meeting the probability of each class varies largely
        """
        p_classes = [] # compute the probability of each class
        for i in range(num_classes):
            p = len(np.where(labels == i)[0])
            p_classes.append(p / num_examples)
        print(p_classes)
        #p_classes = [1/num_classes for _ in range(num_classes)]

        clientidx_map = self.allocate_classes(num_clients, num_classes, p_classes)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[cid]) for cid in range(num_classes)])

        while min_size < min_require_size:
            idx_per_client = [[] for _ in range(num_clients)]
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Case 1 (original case in Dir): Balance
                proportions = np.array([p * (len(idx_j) < num_examples / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_per_client))])
                # Case 2: Don't balance
                #proportions = np.array([p * (j in label_netidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_per_client))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                '''Note: Process the remainder data samples (yipeng, 2023-11-14).
                There are some cases that the samples of class k are not allocated completely, i.e., proportions[-1] < len(idx_k)
                In these cases, the remainder data samples are assigned to the last client in `clientidx_map[k]`.
                '''
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], num_clients-1):
                        proportions[w] = len(idx_k)
                
                idx_per_client = [idx_j + idx.tolist() for idx_j, idx in zip(idx_per_client, np.split(idx_k, proportions))] 
                min_size = min([len(idx_j) for idx_j in idx_per_client])
        
        # Make the examples equal among all the clients
        base_examples_per_user = int(num_examples // num_clients)
        remaining_examples = num_examples % num_clients
        examples_per_user = [base_examples_per_user] * num_clients

        for i in range(remaining_examples):
            examples_per_user[i] += 1
        
        idx_clients = []
        for c in range(num_clients):
            idx_clients.extend(idx_per_client[c])

        dataidx_map = {}
        cum = np.concatenate((np.array([0]), np.cumsum(examples_per_user)))
        for c in range(num_clients):
            dataidx_map[c] = np.random.permutation(idx_clients[cum[c]: cum[c+1]])

        return dataidx_map


class ShardPartitioner(Partitioner):
    r"""The implementation of Shard-paritition way (https://arxiv.org/abs/1602.05629) is from
    https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
    """
    def __init__(self, dataset_name='mnist', C=1):
        super(ShardPartitioner, self).__init__()
        self.name = 'shard'
        self.dataset_name = dataset_name
        self.C = C
        self.output_name = '{}{}'.format(self.name, self.C)
        
    def partition_data(self, labels, num_clients, num_classes):
        targets = labels
        if self.C > num_classes:
            print("C > num_classes")
            raise ValueError
        num_shards = num_clients * min(self.C, num_classes)

        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)
        num_samples = targets.shape[0]

        size_shard = int(num_samples / num_shards)
        if num_samples % num_shards != 0:
            print("Length of dataset isn't divided exactly by num_shards: {} % {} != 0. "
                    "Some samples will be dropped.".format(num_samples, num_shards))
            raise ValueError

        shards_per_client = int(num_shards / num_clients)
        if num_shards % num_clients != 0:
            print("warning: num_shards isn't divided exactly by num_clients. "
                    "Some shards will be dropped.")
            raise ValueError

        indices = np.arange(num_samples)
        # sort sample indices according to labels
        indices_targets = np.vstack((indices, targets))
        indices_targets = indices_targets[:, indices_targets[1, :].argsort()]
        # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
        sorted_indices = indices_targets[0, :]
        print(len(sorted_indices))

        # permute shards idx, and slice shards_per_client shards for each client
        rand_perm = np.random.permutation(num_shards)
        num_client_shards = np.ones(num_clients) * shards_per_client
        # sample index must be int
        num_cumsum = np.cumsum(num_client_shards).astype(int)
        # shard indices for each client
        client_indices_pairs = [(cid, idxs) for cid, idxs in
                            enumerate(np.split(rand_perm, num_cumsum)[:-1])]
        client_shards_dict = dict(client_indices_pairs)

        # map shard idx to sample idx for each client
        client_dict = dict()
        for cid in range(num_clients):
            shards_set = client_shards_dict[cid]
            current_indices = [
                sorted_indices[shard_id * size_shard: (shard_id + 1) * size_shard]
                for shard_id in shards_set]
            client_dict[cid] = np.concatenate(current_indices, axis=0)

        return client_dict


# python sim/data/partition.py -d mnist -n 10 --partition exdir -C 1 --alpha 1.0 
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

    if partition == 'iid':
        p = IIDPartitioner(dataset_name=dataset_name)
        p.gen_dataidx_map(labels=labels, map_dir=output_dir, num_clients=num_clients, num_classes=num_classes)
    elif partition == 'dir':
        p = DirPartitioner(dataset_name=dataset_name, alpha=alpha)
        p.gen_dataidx_map(labels=labels, map_dir=output_dir, num_clients=num_clients, num_classes=num_classes)
    elif partition == 'exdir':
        p = ExDirPartitioner(dataset_name=dataset_name, C=C, alpha=alpha)
        p.gen_dataidx_map(labels=labels, map_dir=output_dir, num_clients=num_clients, num_classes=num_classes)
    elif partition == 'exdirb':
        p = ExDirBPartitioner(dataset_name=dataset_name, C=C, alpha=alpha)
        p.gen_dataidx_map(labels=labels, map_dir=output_dir, num_clients=num_clients, num_classes=num_classes)
    elif partition == 'shard':
        p = ShardPartitioner(dataset_name=dataset_name, C=C)
        p.gen_dataidx_map(labels=labels, map_dir=output_dir, num_clients=num_clients, num_classes=num_classes)

if __name__ == '__main__':
    from datasets import build_dataset
    main()
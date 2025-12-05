from .lenet import LeNet
from .resnet import resnet20

def build_model(model_name='lenet5', dataset_name='mnist'):
    if model_name == 'lenet5':
        if dataset_name in [ "cifar10", "cifar100", "cinic10" ]:
            model = LeNet(dataset=dataset_name)
        else:
            raise ValueError
    elif model_name == 'resnet10':
        model = resnet20()
    else:
        raise ValueError
    return model
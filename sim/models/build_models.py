import importlib
import re

MODElNAMES = {
    'logistic': 'Logistic', 
    'mlp'     : 'MLP',
    'lenet5'  : 'LeNet5',
    'cnn': 'CNN'
    }
dataset_num_classes = { 'mnist': 10, 'fashionmnist': 10, 'cifar10': 10, 'cifar100': 100, 'cinic10': 10,}

def check(model, dataset):
    model_pattern = r'logistic|mlp|lenet5|cnn|wvgg|resnet|resnetii|resnetgnii|wrn'
    model_result = re.search(model_pattern, model)
    model_result = model_result.group(0) if model_result != None else model_result
    dataset_pattern = r'mnist|cifar|cinic10'
    dataset_result = re.search(dataset_pattern, dataset)
    dataset_result = dataset_result.group(0) if dataset_result != None else dataset_result
    print(model_result, dataset_result)
    return model_result, dataset_result


def build_model(model_name='lenet5', dataset_name='mnist'):
    '''https://github.com/lx10077/fedavgpy/blob/master/main.py (lines 100-103)'''
    check_result = check(model_name, dataset_name)
    #print(check_result)
    assert check_result[0] != None and check_result[1] != None

    # module
    module_dict = { 'mnist'  : 'mnist',
                    'cifar'  : 'cifar',
                    'cinic10': 'cifar',
                    }
    module = importlib.import_module('.{}_{}'.format(module_dict[check_result[1]], check_result[0]), 'sim.models')
    # class or function to construct models
    if model_name in MODElNAMES.keys():
        model_class = getattr(module, MODElNAMES[model_name])
    else:
        model_class = getattr(module, model_name)
    # model
    num_classes = dataset_num_classes[dataset_name]
    if dataset_name in ['mnist', 'fashionmnist']:
        model = model_class()
    else:
        print(2)
        print(model_class)
        model = model_class(num_classes=num_classes)
    return model
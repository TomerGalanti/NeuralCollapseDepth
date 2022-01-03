""" helper function
author baiyu
"""
import os
import sys
import re
import datetime
import operator
import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(settings):
    """ return given network
    """

    if settings.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(settings)
    elif settings.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(settings)
    elif settings.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(settings)
    elif settings.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(settings)
    elif settings.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(settings)
    elif settings.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161(settings)
    elif settings.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169(settings)
    elif settings.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201(settings)
    elif settings.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet(settings)
    elif settings.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3(settings)
    elif settings.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4(settings)
    elif settings.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2(settings)
    elif settings.net == 'xception':
        from models.xception import xception
        net = xception(settings)
    elif settings.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(settings)
    elif settings.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(settings)
    elif settings.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(settings)
    elif settings.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(settings)
    elif settings.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(settings)
    elif settings.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18(settings)
    elif settings.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34(settings)
    elif settings.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50(settings)
    elif settings.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101(settings)
    elif settings.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152(settings)
    elif settings.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50(settings)
    elif settings.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101(settings)
    elif settings.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152(settings)
    elif settings.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet(settings)
    elif settings.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(settings)
    elif settings.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet(settings)
    elif settings.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet(settings)
    elif settings.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(settings)
    elif settings.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet(settings)
    elif settings.net == 'attention56':
        from models.attention import attention56
        net = attention56(settings)
    elif settings.net == 'attention92':
        from models.attention import attention92
        net = attention92(settings)
    elif settings.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18(settings)
    elif settings.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34(settings)
    elif settings.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50(settings)
    elif settings.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101(settings)
    elif settings.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152(settings)
    elif settings.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet(settings)
    elif settings.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18(settings)
    elif settings.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34(settings)
    elif settings.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50(settings)
    elif settings.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101(settings)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    net = net.to(settings.device)

    return net

def get_training_dataloader(dataset_name, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    """
    dataset = operator.attrgetter(dataset_name)(torchvision.datasets)

    if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
        im_size = 28
        padded_im_size = 32
        transform_train = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)]
                                       )
        training_set = dataset(root='./data', train=True, download=True, transform=transform_train)

    elif dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        training_set = dataset(root='./data', train=True, download=True, transform=transform_train)

    elif dataset_name == 'STL10' or dataset_name == 'SVHN':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        training_set = dataset(root='./data', split='train', download=True, transform=transform_train)

    training_loader = DataLoader(training_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return training_loader

def get_test_dataloader(dataset_name, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return test dataloader
    """
    dataset = operator.attrgetter(dataset_name)(torchvision.datasets)

    if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
        im_size = 28
        padded_im_size = 32
        transform_test = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)]
                                       )
        test_set = dataset(root='./data', train=False, download=True, transform=transform_test)

    elif dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_set = dataset(root='./data', train=False, download=True, transform=transform_test)

    elif dataset_name == 'STL10' or dataset_name == 'SVHN':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_set = dataset(root='./data', split='test', download=True, transform=transform_test)

    test_loader = DataLoader(
        test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]
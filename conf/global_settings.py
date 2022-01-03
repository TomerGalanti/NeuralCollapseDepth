""" configurations for this project
author baiyu
"""
import os
from datetime import datetime

dataset_name = 'CIFAR10' # MNIST, FashionMNIST, STL10, CIFAR10, CIFAR100, SVHN

if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
    num_output_classes = 10
    num_input_channels = 1
    mean = 0.1307
    std = 0.3081

elif dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100':
    num_output_classes = int(dataset_name[5:])
    num_input_channels = 3
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

elif dataset_name == 'STL10':
    num_output_classes = 10
    num_input_channels = 3
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

elif dataset_name == 'SVHN':
    num_output_classes = 10
    num_input_channels = 3
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 600
MILESTONES = [60, 120, 160, 300]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# saving save
SAVE_EPOCH = 10
directory = './results'
resume = False

# network architecture
net = 'resnet18'
top_layers_type = 'res' # 'res' or 'fc'
top_depth = 2

# device
device = 'cuda'

# training hyperparameters
batch_size = 128
warm = 1
lr = 0.1
top_lr = 0.1
import os

import torch
import torchvision
import torchvision.transforms as transforms

import torch.multiprocessing
import h5py
import os
import numpy as np
from os.path import join as ospj
import sys
torch.multiprocessing.set_sharing_strategy("file_system")

np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

class CIFAR10:
    def __init__(self, args):
        super(CIFAR10, self).__init__()

        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # sys.path.append(ospj(PROJECT_ROOT, args.data))
        data_root = ospj(PROJECT_ROOT, f'{args.data}/CIFAR10/')

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        print("=> Preparing CIFAR10 dataset")
        train_dataset = torchvision.datasets.CIFAR10(root=data_root,
                                           train=True,
                                           transform=transform_train,
                                           download=True)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           **kwargs)
        
        test_dataset = torchvision.datasets.CIFAR10(root=data_root,
                                           train=False,
                                           transform=transform_test,
                                           download=True)

        self.val_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=100,
                                           shuffle=False,
                                           **kwargs)
        
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')
        
class CIFAR100:
    def __init__(self, args):
        super(CIFAR100, self).__init__()

        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # sys.path.append(ospj(PROJECT_ROOT, args.data))
        data_root = ospj(PROJECT_ROOT, f'{args.data}/CIFAR100/')

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
        print("=> Preparing CIFAR100 dataset")
        train_dataset = torchvision.datasets.CIFAR100(root=data_root,
                                           train=True,
                                           transform=transform_train,
                                           download=True)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           **kwargs)
        
        test_dataset = torchvision.datasets.CIFAR100(root=data_root,
                                           train=False,
                                           transform=transform_test,
                                           download=True)

        self.val_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=100,
                                           shuffle=False,
                                           **kwargs)
        
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')
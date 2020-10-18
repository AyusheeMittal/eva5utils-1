from eva5utils.utils import DEVICE, IS_CUDA
import torch
import torchvision
from .TinyImageNet import download_images, class_names, TinyImageNet
from .DatasetFromSubset import DatasetFromSubset
from torch.utils.data import DataLoader, Dataset, random_split

def load_cifar10(train_transform, test_transform, cuda_batch_size, cpu_batch_size=32, num_workers=2):
    dataloader_args = dict(shuffle=True, batch_size=cuda_batch_size, num_workers=num_workers, pin_memory=True) \
        if IS_CUDA else dict(shuffle=True, batch_size=cpu_batch_size)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)  # batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)  # batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return (trainloader, testloader, classes)


def load_tiny_imagenet(train_transform, test_transform, train_split, cuda_batch_size, cpu_batch_size=32, num_workers=2):
    download_images()
    classes = class_names()
    print(len(classes))

    dataset = TinyImageNet(classes)
    print('Dataset length: ', len(dataset))
    train_split = len(dataset) * train_split //100
    test_split = len(dataset) - train_split
    train_data, test_data = random_split(dataset, [train_split, test_split])
    print('Trainset length: ', len(train_data))
    print('Testset length: ', len(test_data))

    # Apply transformations on train and test data separately
    train_data = DatasetFromSubset(train_data, train_transform)
    test_data = DatasetFromSubset(test_data, test_transform)

    dataloader_args = dict(shuffle=True, batch_size=cuda_batch_size, num_workers=num_workers,
                           pin_memory=True) if IS_CUDA else dict(shuffle=True, batch_size=cpu_batch_size)

    trainloader = torch.utils.data.DataLoader(train_data, **dataloader_args)
    testloader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    return (trainloader, testloader, classes)

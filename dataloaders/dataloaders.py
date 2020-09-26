from utils import DEVICE, IS_CUDA
import torch
import torchvision

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
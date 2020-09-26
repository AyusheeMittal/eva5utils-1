from model import Model7
from utils.helpers import show_model_summary, DEVICE
import dataloaders
from transforms import model7_transforms
from utils import plot_samples
from train import train_loop
from test import test_loop
import torch.optim as optim
import torch.nn as nn



model = Model7()
show_model_summary(model.to(DEVICE), (3, 32, 32))

# Constants, put in config
epochs = 50
cuda_batch_size=128
cpu_batch_size = 4
num_workers = 4

# ToDo: Create separate transforms for train and test...
transforms = model7_transforms()
(train_loader, test_loader, classes) = \
    dataloaders.load_cifar10(transforms, transforms, cuda_batch_size, cpu_batch_size, num_workers)

plot_samples(train_loader)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.009, momentum=0.9)

train_loop(epochs, train_loader, model, DEVICE, optimizer, criterion)
test_loop(test_loader, model, DEVICE, criterion)

import torchvision.transforms as transforms

from albumentations.pytorch import transforms as P
from albumentations.augmentations import transforms as A
from albumentations.core import composition as C
import numpy as np


def model7_transforms():
    transform = transforms.Compose(
        [transforms.RandomCrop(30, padding = 2),
         transforms.ToTensor(),
         transforms.RandomHorizontalFlip(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform


def model9_resnet_train_transforms():
  transforms = C.Compose([
    A.HorizontalFlip(),
    A.RandomCrop(height=30, width=30, p=5.0),
    #A.Cutout(num_holes=1, max_h_size=16, max_w_size=16),
    P.ToTensor(dict (mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    ])
  return lambda img: transforms(image = np.array(img))["image"]


def model9_resnet_test_transforms():
  transforms = C.Compose([
    P.ToTensor(dict (mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    ])
  return lambda img: transforms(image = np.array(img))["image"]


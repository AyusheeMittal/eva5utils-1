import torchvision.transforms as transforms

from albumentations.pytorch import transforms as P
from albumentations.augmentations import transforms as A
from albumentations.core import composition as C
import numpy as np
import cv2

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


def model10_resnet_train_transforms():
  transforms = C.Compose([
    A.HorizontalFlip(),
    #A.RandomCrop(height=30, width=30, p=5.0),
    A.Cutout(num_holes=1, max_h_size=16, max_w_size=16),
    P.ToTensor(dict (mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    ])
  return lambda img: transforms(image = np.array(img))["image"]


def model11_davidnet_train_transforms():
  transform = C.Compose([
    A.PadIfNeeded(min_height=36, min_width=36, border_mode=cv2.BORDER_CONSTANT,
        value=0.5),
    A.RandomCrop(height=32, width=32, p=1),
    A.HorizontalFlip(p=0.5),
    A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, p=1),
    P.ToTensor(dict (mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    ])
  return lambda img: transform(image = np.array(img))["image"]


def model12_train_transforms():
  transform = C.Compose([
    A.PadIfNeeded(min_height=70, min_width=70, border_mode=cv2.BORDER_CONSTANT,
         value=0.5),
    A.RandomCrop(height=64, width=64),
    A.HorizontalFlip(p=0.5),
    A.Cutout(num_holes=1, max_h_size=32, max_w_size=32, p=1),
    P.ToTensor(dict (mean=(0.4802, 0.4481, 0.3975), std=(0.2302, 0.2265, 0.2262)))
    ])
  return lambda img: transform(image = np.array(img))["image"]

def model12_test_transforms():
  transform = C.Compose([
    P.ToTensor(dict (mean=(0.4802, 0.4481, 0.3975), std=(0.2302, 0.2265, 0.2262)))
    ])
  return lambda img: transform(image = np.array(img))["image"]

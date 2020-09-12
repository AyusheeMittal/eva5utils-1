import torchvision.transforms as transforms



def model7_transforms():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomHorizontalFlip(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform
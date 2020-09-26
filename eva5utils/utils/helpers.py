
import torch
from torchsummary import summary
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

from gradcam import gradcam as G
from gradcam import utils as U

IS_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if IS_CUDA else "cpu")

# def get_device():
#     cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if cuda else "cpu")
#     return device

def show_model_summary(model, input_size):
    #model = Net().to(device)
    #summary(model, input_size=(3, 32, 32))
    result = summary(model, input_size=input_size)
    print(result)


def accuracy_per_class(model, classes, testloader, device):
    class_len = len(classes)
    class_correct = list(0. for i in range(class_len))
    class_total = list(0. for i in range(class_len))
    with torch.no_grad():
        for data, label in testloader:
            images, labels = data.to(device), label.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def show_gradcam(model, model_type, layer, testloader, classes, samples=5):
    #config = dict(model_type='resnet', arch=model, layer_name='layer4')
    config = dict(model_type=model_type, arch=model, layer_name=layer)
    gradcam = G.GradCAM.from_config(**config)

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    outputs = model(images.to(DEVICE))
    _, predicted = torch.max(outputs.data, 1)

    for i in range(samples):
        imagestodisplay = []
        mask, _ = gradcam(images[i][np.newaxis, :].to(DEVICE))
        heatmap, result = U.visualize_cam(mask, images[i][np.newaxis, :])
        imagestodisplay.extend([images[i].cpu(), heatmap, result])
        grid_image = make_grid(imagestodisplay, nrow=3)
        plt.figure(figsize=(20, 20))
        plt.imshow(np.transpose(grid_image, (1, 2, 0)))
        plt.show()
        print(f"Prediction : {classes[predicted[i]]}, Actual : {classes[labels[i]]}")


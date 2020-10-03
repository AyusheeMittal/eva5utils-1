import matplotlib.pyplot as plt
import numpy as np
import torchvision
from gradcam import GradCAM
from .gradcam_utils import visualize_cam
from .helpers import DEVICE
import torch

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_samples(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))


def plot_misclassified_gradcam(model, incorrect_indexes, classes, layer, model_type='resnet'):
    # {23: {'actual': 1, 'predicted': 4}}
    gradcam = GradCAM.from_config(model_type=model_type, arch=model, layer_name=layer)

    x = 0
    y = 0
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    plt.setp(axs, xticks=[], yticks=[])
    fig.subplots_adjust(wspace=0.7)
    images = list(incorrect_indexes.items())[:25]
    for index, results in images:
        img = results['data']
        img = torch.from_numpy(img)

        actual_class = classes[results['actual']]
        predicted_class = classes[results['predicted']]

        mask, _ = gradcam(img[np.newaxis, :].to(DEVICE))
        heatmap, result = visualize_cam(mask, img[np.newaxis, :])
        result = np.transpose(result.cpu().numpy(), (1, 2, 0))

        axs[x, y].imshow(result)
        axs[x, y].set_title('Actual Class:' + str(actual_class) + "\nPredicted class: " + str(predicted_class))

        if y == 4:
            x += 1
            y = 0
        else:
            y += 1

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from .gradcam import GradCAM
from .gradcam_utils import visualize_cam
from .helpers import DEVICE
import torch
import math

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
    fig, axs = plt.subplots(5, 5, figsize=(20, 20))
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


def plot_train_vs_test(train_list, test_list, label):
  train_range = range(0, len(train_list))
  test_range = range(0, len(test_list))

  fig, ax = plt.subplots()
  ax.plot(train_range, train_list, label='Training ' + label, color='blue')
  ax.plot(test_range, test_list, label='Test ' + label, color = 'red')
  legend = ax.legend(loc='center right', fontsize='x-large')
  plt.xlabel('epoch')
  plt.ylabel(label)
  plt.title('Training and Test ' + label)
  plt.show()


def get_cycle_lr(iterations, step_size, lr_max, lr_min):
  cycle = math.floor(1 + iterations / (2.0 * step_size))
  x = math.fabs((iterations / step_size) - (2 * cycle) + 1)
  lr = lr_min + (lr_max - lr_min) * (1 - x)
  print("cycle=", str(cycle), ", x=", str(x), ", lr=", lr)
  return lr


def plot_cycle_lr(samples, batch_size, epochs, lr_max, lr_min):
  iterations = math.ceil(samples/batch_size) # 98
  step_size = epochs/2.0
  y_value = []
  #
  for i in range(iterations):
      lr = get_cycle_lr(i, step_size, lr_max, lr_min)
      print("i=", i, ", lr = ", lr)
      y_value.append(lr)
  #
  fig, ax = plt.subplots()
  ax.plot(y_value)
  plt.xlabel('epoch')
  plt.ylabel('LR')
  plt.title('Cyclic LR')
  plt.show()


#plot_cycle_lr(60000, 512, 24, 3, 0.3)
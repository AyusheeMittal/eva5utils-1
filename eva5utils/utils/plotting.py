import matplotlib.pyplot as plt
import numpy as np
import torchvision


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


def plot_misclassified(incorrect_indexes):
    # {23: {'actual': 1, 'predicted': 4}}

    x = 0
    y = 0
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    plt.setp(axs, xticks=[], yticks=[])
    fig.subplots_adjust(wspace=0.7)
    images = list(incorrect_indexes.items())[:25]
    for index, results in images:
        # print(index)
        img = results['data']
        img = np.squeeze(img)
        actual_class = results['actual']
        predicted_class = results['predicted']

        #plt.savefig("misclassified.png")
        #files.download("misclassified.png")
        axs[x, y].imshow(img)
        axs[x, y].set_title('Actual Class:' + str(actual_class) + "\nPredicted class: " + str(predicted_class))

        if y == 4:
            x += 1
            y = 0
        else:
            y += 1

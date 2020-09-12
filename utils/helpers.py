
import torch
from torchsummary import summary

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
import torch


def test_loop(testloader, model, device, criterion):
    correct = 0
    total = 0
    test_loss = 0
    running_loss = 0
    with torch.no_grad():
        for data, label in testloader:
            images, labels = data.to(device), label.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss = criterion(outputs, labels)
            running_loss += test_loss.item()

    print('Accuracy of the network on the 10000 test images: %d %%, Test loss:' % (
            100 * correct / total), running_loss)
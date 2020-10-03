import torch


def test_loop(testloader, model, device, criterion):
    correct = 0
    total = 0
    test_loss = 0
    running_loss = 0
    loss_accumulator = []
    acc_accumulator = []

    with torch.no_grad():
        for data, label in testloader:
            images, labels = data.to(device), label.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            test_loss += loss
            running_loss += test_loss.item()
            loss_accumulator.append(loss)
            acc_accumulator.append(100 * correct / total)
    print('Accuracy of the network on the 10000 test images: %d %%, Test loss:' % (
            100.0 * correct / total), running_loss)
    return loss_accumulator, acc_accumulator
import torch

def train_loop(epochs, trainloader, model, device, optimizer, criterion, scheduler=None, stepWithLoss=False):
    loss_accumulator = []
    acc_accumulator = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_corrects = 0
        processed = 0
        for i, (data, labels) in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            processed += len(data)

        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))

        loss_accumulator.append(running_loss)
        #acc_accumulator.append(running_corrects.double() / len(trainloader.dataset))
        acc_accumulator.append(100.0*running_corrects/processed)

        if scheduler:
            if stepWithLoss:
                scheduler.step(loss)
            else:
                scheduler.step()
    return loss_accumulator, acc_accumulator
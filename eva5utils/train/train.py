

def train_loop(epochs, trainloader, model, device, optimizer, criterion, scheduler=None, stepWithLoss=false):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
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

        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))

        if scheduler:
            if stepWithLoss:
                scheduler.step(loss)
            else:
                scheduler.step()



def train_loop(epochs, trainloader, model, device, optimizer, criterion, scheduler):
    for epoch in range(epochs):  # loop over the dataset multiple times
        # total=0
        running_loss = 0.0
        for i, (data, labels) in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data.to(device), labels.to(device)
            # print(inputs.shape)
            # print(labels.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # total += labels.size(0)
            # print(total)
            # if i % 40 == 39:    # print every 2000 mini-batches
            # print('[%d, %5d] loss: %.3f' %
            # (epoch + 1, i + 1, running_loss / 40))
            # running_loss = 0.0
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))
        
        scheduler.step()

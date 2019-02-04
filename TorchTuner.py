import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

## TODO
## Need a better name for accuracy function. 

def evaluateModel(model=None,
                  train_dataset=None,
                  val_dataset=None,
                  optimizer_func=None, 
                  optimizer_param=None, 
                  criterion_func=None,
                  epochs=9,
                  batch_size=4,
                  accuracy_func=None):

    ## Needs to return the model itself, with its weights etc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    # Training metrics
    tr_loss = []
    tr_acc = []

    # Validation metrics
    val_loss = []
    val_acc = []

    # Move to GPU
    model = model.to(device)

    optimizer = optimizer_func(model.parameters(), **optimizer_param)
    criterion = criterion_func()

    for e in range(epochs):
        running_acc = 0
        running_loss = 0.0

        model.train(True)

        print('Training')
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(data)
            output_label = torch.topk(output, 1)[1].view(-1)
            output = output.view(batch_size, -1)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy_func(output_label, label)

        running_loss /= (len(train_loader) * batch_size)
        running_acc /= (len(train_loader) * batch_size)
        running_acc *= 100

        tr_loss.append(running_loss)
        tr_acc.append(running_acc)

        print('Training Loss : ', running_loss)
        print('Training Accuracy : ', running_acc)

        running_loss = 0.0
        running_acc= 0

        model.eval()
        print('Validation')
        with torch.no_grad():
            for data, label in tqdm(val_loader):
                data = data.to(device)
                label = label.to(device)

                output = model(data)
                output_label = torch.topk(output, 1)[1].view(-1)
                output = output.view(batch_size, -1)

                loss = criterion(output, label)

                running_loss += loss.item()
                running_acc += accuracy_func(output_label, label)

        running_loss /= (len(val_loader) * batch_size)
        running_acc /= (len(val_loader) * batch_size)
        running_acc *= 100

        val_loss.append(running_loss)
        val_acc.append(running_acc)

        print('Validation Loss : ', running_loss)
        print('Validation Accuracy : ', running_acc)
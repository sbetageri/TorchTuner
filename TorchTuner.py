import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

## TODO
## Need a better name for accuracy function. 

def evaluateModel(model=None,
                  train_dataset=None,
                  val_dataset=None,
                  optimizer_func=None
                  optimizer_param=None, 
                  criterion_func=None,
                  epochs=9,
                  batch_size=4,
                  accuracy_func=None,
                  verbose=False):

    ## Needs to return the model itself, with its weights etc

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optimizer(model.parameters(), **optimizer_param)

    # Training metrics
    tr_loss = []
    tr_acc = []

    # Validation metrics
    val_loss = []
    val_acc = []

    for e in range(epochs):
        running_acc = 0
        running_loss = 0.0

        model.train(True)

        print('Training')
        for data, label in tqdm(train_loader):
            optimizer.zero_grad()

            output = model(data)
            output = output.view(batch_size, -1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += accuracy_func(output, label)

        running_loss /= len(train_loader)
        running_accuracy /= len(train_loader)

        tr_losses.append(running_loss)
        tr_acc.append(running_accuracy)

        print('Training Loss : ', running_loss)
        print('Training Accuracy : ', running_accuracy)

        running_loss = 0.0
        running_accuracy = 0

        model.eval(True)
        print('Validation')
        for data, label in tqdm(test_loader):
            output = model(data)
            output = output.view(batch_size, -1)
            loss = criterion(output, label)

            running_loss += loss.item()
            running_accuracy += accuracy_func(output, label)

        running_loss /= len(train_loader)
        running_accuracy /= len(train_loader)

        val_loss.append(running_loss)
        val_acc.append(running_accuracy)

        print('Validation Loss : ', running_loss)
        print('Validation Accuracy : ', running_accuracy)
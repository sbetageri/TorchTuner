import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

class TorchTuner:
    ## Names for parameters
    _OPTIM_FUNC = 'optimizer_func'
    _OPTIM_PARAM = 'optimizer_param'
    _EPOCHS = 'epochs'
    _BATCH_SIZE = 'batch_size'
    _PARAM_IDX = 'param_id'

    def __init__(self, 
                 model=None, 
                 criterion_func=None,
                 accuracy_func=None,
                 train_dataset=None,
                 val_dataset=None):
        '''Initialise torch tuner
        
        :param model: Model to be tested, defaults to None
        :param model: Custom PyTorch model, optional
        :param criterion_func: Criterion Function, defaults to None
        :param criterion_func: torch.nn, optional
        :param accuracy_func: Accuracy funciton, defaults to None
        :param accuracy_func: Custom function, optional
        :param train_dataset: Training dataset, defaults to None
        :param train_dataset: torch.utils.data.Dataset, optional
        :param val_dataset: Validation dataset, defaults to None
        :param val_dataset: torch.utils.data.Dataset, optional
        '''

        self.params = []
        self.param_name_prefix = 'param_'
        self.model = model
        self.criterion = criterion_func()
        self.accuracy_func = accuracy_func
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.results = {}

    def evaluateModel(self,
                      param_id = None,
                      optimizer_func=None, 
                      optimizer_param=None, 
                      epochs=9,
                      batch_size=4):
        '''Evaluates a model and returns loss and accuracy
        
        Builds and executes the entire training and validation pipeline.
        Runs implicitly on the GPU
        
        :param optimizer_func: function to obtain optimizer, defaults to None
        :param optimizer_func: torch.optim, optional
        :param optimizer_param: parameters for optimizer, defaults to None
        :param optimizer_param: dict, optional
        :param epochs: number of epochs, defaults to 9
        :param epochs: int, optional
        :param batch_size: size of batch, defaults to 4
        :param batch_size: int, optional
        :return: Log of evaluation metrics
        :rtype: Dictionary
        '''

        ## Needs to return the model itself, with its weights etc
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, shuffle=True)

        # Training metrics
        tr_loss = []
        tr_acc = []

        # Validation metrics
        val_loss = []
        val_acc = []

        # Move to GPU
        model = self.model.to(device)

        optimizer = optimizer_func(model.parameters(), **optimizer_param)
        criterion = self.criterion

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
                running_acc += self.accuracy_func(output_label, label)

            running_loss /= (len(train_loader) * batch_size)
            running_acc /= (len(train_loader) * batch_size)
            running_acc *= 100

            tr_loss.append(running_loss)
            tr_acc.append(running_acc)

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
                    running_acc += self.accuracy_func(output_label, label)

                running_loss /= (len(val_loader) * batch_size)
                running_acc /= (len(val_loader) * batch_size)
                running_acc *= 100

                val_loss.append(running_loss)
                val_acc.append(running_acc)

        return {
            'train_loss' : tr_loss,
            'train_acc' : tr_acc,
            'val_loss' : val_loss,
            'val_acc' : val_acc
        }

    def addModel(self, 
                model=None,
                criterion_func=None,
                accuracy_func=None):
        '''Change model
        
        Change the underlying model for which the hyperparameters need to be tested
        
        :param model: Pytorch model, defaults to None
        :param model: Custom model, optional
        :param criterion_func: Loss function, defaults to None
        :param criterion_func: torch.nn, optional
        :param accuracy_func: Evaluation metric function, defaults to None
        :param accuracy_func: function, optional
        '''

        self.model = model
        self.criterion_func = criterion_func
        self.accuracy_func = accuracy_func

    def addHyperparameters(self,
                           optimizer_func=None,
                           optimizer_param=None,
                           epochs=9,
                           batch_size=8):
        '''Add hyperparams for evaluation
        
        :param optimizer_func: Optimizer, defaults to None
        :param optimizer_func: torch.optim, optional
        :param optimizer_param: Parameters to optimizer, defaults to None
        :param optimizer_param: Dict of params, optional
        :param epochs: Number of epochs to run evaluation metric on, defaults to 9
        :param epochs: int, optional
        :param batch_size: Number of data-points to consider during evaluation, defaults to 8
        :param batch_size: int, optional
        '''

        param_idx = self.param_name_prefix + str(len(self.params) + 1)
        param = {
            TorchTuner._PARAM_IDX : param_idx,
            TorchTuner._OPTIM_FUNC : optimizer_func,
            TorchTuner._OPTIM_PARAM : optimizer_param,
            TorchTuner._EPOCHS : epochs
        }
        self.params.append(param)

    def evaluateHyperparams(self):
        '''Evaluate hyperparameters
        
        Evaluate hyperparams and log results
        
        '''

        for param in self.params:
            result = self.evaluateModel(**param)
            name = param[TorchTuner._PARAM_IDX]
            self.results[name] = result
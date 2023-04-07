 
# # Centralized Learning to Federated Learning
# 
# COGONI Guillaume (p1810070)

 
# # Import libraries


# Torch
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
from Preprocessing import Prepropressing

# Time
import time

# Random
import random

# Tqdm
from tqdm import tqdm
from tqdm import trange

# Copy
import copy

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

# Pandas
import pandas as pd

# Datetime
from datetime import datetime, timedelta

# Matplotlib
import matplotlib.pyplot as plt

import warnings

from ArchitectureCentralize import ArchitectureCentralize
warnings.filterwarnings('ignore')


dataset_tensor = Prepropressing.first_exemple()#torch.utils.data.TensorDataset(X_tensor, y_tensor)


len(dataset_tensor)

 
# # Models


class MyNet(nn.Module):
    def __init__(self, _Input, _Output):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(_Input, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, _Output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# # Centralized Version


# ## Separation of the data (train and validation)


train_size = int(0.6 * len(dataset_tensor))
val_size = len(dataset_tensor) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset_tensor, [train_size, val_size])


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3200, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=3200, shuffle=True)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes= {'train': len(train_dataset), 'val': len(val_dataset)}

 
# # Train and Test function


def train_and_test_nn(model, criterion, optimizer, dataloaders, batch_size, num_epochs=25):

    since = time.time()
    # Instantiate the neural network and the optimizer
    model = model
    optimizer = optimizer
    criterion = criterion
    best_acc_avg = 0.0

    #pbar = trange(num_epochs, unit="carrots")

    # Train the neural network
    for epoch in range(num_epochs):
        print("\n")
        print("_________________________Epoch %d / %d ____________________" % (epoch+1, num_epochs))
        print("\n")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            correct = 0
            precision = 0.0
            recall = 0.0
            i = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item()
                    correct += accuracy_score(preds, labels)
                    precision += precision_score(labels, preds)
                    recall += recall_score(labels, preds)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                i+=1
            
            ##Statistics

            # Calculate the average loss
            loss_avg = running_loss / (i+1)

            # Calculate the average accuracy
            accuracy_avg = correct / (i+1)

            # Calculate the average precision and recall
            precision_avg = precision / (i+1)

            # Calculate the average recall
            recall_avg = recall / (i+1)


            # Print the average loss, accuracy, precision, recall for once for train and val per epoch
            print('PHASE %s:  [AVG loss: %.3f || AVG Accuracy: %.4f] [AVG Precision: %.3f || AVG Recall: %.3f]' % 
                    (phase, loss_avg, accuracy_avg, precision_avg, recall_avg))
            

            # deep copy the model
            if phase == 'val' and accuracy_avg > best_acc_avg:
                best_acc_avg = accuracy_avg
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print("\n")
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc_avg))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model = MyNet(dataset_tensor.tensors[0].shape[1], dataset_tensor.tensors[1].shape[1])
epochs = 20

criterion = nn.BCEWithLogitsLoss()
optimizer_ft = Adam(model.parameters(), lr=0.001)

# decay LR by a factor of 0.1 every 5 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

model_ft = ArchitectureCentralize().start_classification(model, criterion, optimizer_ft, dataset_tensor, train_batch_size=3200, valid_batch_size=3200, num_epochs=10)
#model_ft = train_and_test_nn(model, criterion, optimizer_ft, dataloaders, 1, num_epochs=10)

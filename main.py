from Model import MyNet
from Node import Node
from Global_Node import Global_Node
from Architecture import Architecture
from Preprocessing import Prepropressing
from utility import train_valid_test

import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss

import warnings
warnings.filterwarnings('ignore')

def main():
    architecture = Architecture()
    
    # Preprocessing on the data that return a new dataset
    dataset = Prepropressing.first_exemple()
    
    train_dataset, valid_dataset, test_dataset = train_valid_test(dataset, 0.8, 0.2, 0.0)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3200, shuffle=True) 
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=3200, shuffle=True)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes= {'train': len(train_dataset), 'val': len(valid_dataset)}
    
    model = MyNet(dataset.tensors[0].shape[1:].numel(), dataset.tensors[1].shape[1:].numel())
    
    criterion = CrossEntropyLoss()
    
    optimizer_ft = Adam(model.parameters(), lr=0.001)

    model_ft = architecture.train(model, criterion, optimizer_ft, dataloaders, dataset_sizes, 1, num_epochs=10)


if __name__ == "__main__":
    main()

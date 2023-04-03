from Model import MyNet
from Node import Node
from Global_Node import Global_Node
from ArchitectureCentralize import ArchitectureCentralize
from ArchitectureFederated import ArchitectureFederated
from Preprocessing import Prepropressing

import torch
from torch.optim import Adam, lr_scheduler, SGD
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss

import warnings
warnings.filterwarnings('ignore')

def main():
    architectureCentralize = ArchitectureCentralize()
    architectureFederated = ArchitectureFederated()
    
    # Preprocessing on the data that return a datasetTensor
    dataset = Prepropressing.first_exemple()

    model = MyNet(dataset.tensors[0].shape[1:].numel(), dataset.tensors[1].shape[1:].numel())
    criterion = BCEWithLogitsLoss()
    optimizer_ft = Adam(model.parameters(), lr=0.001)
    
    architectureCentralize.start_classification(model, criterion, optimizer_ft, dataset, train_batch_size=3200, valid_batch_size=3200, num_epochs=10)
    #architectureFederated.start_classification(model, dataset, 3, 100)


if __name__ == "__main__":
    main()

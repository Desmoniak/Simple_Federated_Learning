from Model import MyNet
from Node import Node
from Global_Node import Global_Node
from Architecture import Architecture
from Preprocessing import Prepropressing


from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss

import warnings
warnings.filterwarnings('ignore')

def main():
    print("Hello World")
    
    architecture = Architecture()
    
    example_1 = Prepropressing.first_exemple()
    
    model = MyNet(example_1["input"], example_1["output"])
    
    criterion = CrossEntropyLoss()
    
    optimizer_ft = Adam(model.parameters(), lr=0.001)

    # decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    model_ft = architecture.train(model, criterion, optimizer_ft, example_1["dataloaders"],example_1["dataset_sizes"], 1, num_epochs=10)


if __name__ == "__main__":
    main()

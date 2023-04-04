from Model import MyNet
from ArchitectureCentralize import ArchitectureCentralize
from ArchitectureFederated import ArchitectureFederated
from Preprocessing import Prepropressing

import torch

import warnings
warnings.filterwarnings('ignore')

def main():
    architectureCentralize = ArchitectureCentralize()
    architectureFederated = ArchitectureFederated()
    
    # Preprocessing on the data that return a datasetTensor
    dataset = Prepropressing.first_exemple()

    #architectureCentralize.start_classification(dataset)
    architectureFederated.start_classification(dataset)


if __name__ == "__main__":
    main()

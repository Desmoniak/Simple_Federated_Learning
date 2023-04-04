# Time
import time

# Torch
import torch
from torch.optim import Adam, lr_scheduler, SGD
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss

# Sklearn
from sklearn.metrics import precision_score, recall_score, accuracy_score
from Model import MyNet

from utility import train_valid_test

# Copy
import copy

from sklearn import svm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ArchitectureCentralize():
    def __init__(self):
        pass

###############################################################################
# train_and_valid_classif
###############################################################################
    def train_and_valid_classif(self, model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25):

        since = time.time()
        # Instantiate the neural network and the optimizer
        model = model
        optimizer = optimizer
        criterion = criterion
        best_acc_avg = 0.0

        # Train the neural network
        for epoch in range(num_epochs):
            print("\n")
            print("_________________________Epoch %d / %d ____________________" % (epoch+1, num_epochs))
            print("\n")
            
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                correct = 0.0
                i = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        running_loss += loss.item()
                        correct += torch.sum(labels.squeeze() == preds)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    i+=1
                
                ##Statistics

                # Calculate the average loss
                loss_avg = running_loss / (i+1)

                # Calculate the average accuracy
                accuracy_avg = correct.double() / dataset_sizes[phase]

                # Print the average loss, accuracy, precision, recall for once for train and val per epoch
                print('PHASE %s:  [AVG loss: %.3f || AVG Accuracy: %.4f]' % 
                (phase, loss_avg, accuracy_avg))
                

                # deep copy the model
                if phase == 'valid' and accuracy_avg > best_acc_avg:
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

###############################################################################
# test
###############################################################################
    def test(self, model, test_loader, test_dataset_size):
        print("\n")
        print("_________________________TEST PHASE____________________")
        print("\n")
        
        correct = 0.0

        # Iterate over data.
        for inputs, labels in test_loader:
                model.eval()
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(labels.squeeze() == preds)

        ##Statistics
        # Calculate the average accuracy
        accuracy_avg = correct.double() / test_dataset_size
        


        # Print the average accuracy, precision, recall for once for train and val per epoch
        print('AVG Accuracy: %.4f' % 
        (accuracy_avg))
        
###############################################################################
# train_and_valid_classif_SVM
###############################################################################
    def train_and_valid_classif_SVM(self, model, dataset):
        
        # Split the dataset into train and test sets
        train_size = 60000
        test_size = 10000
        other_size = len(dataset) - (train_size + test_size)
        train_dataset, test_dataset, _ = torch.utils.data.random_split(dataset, [train_size, test_size, other_size])

        # Create a DataLoader from the train set
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))

        # Create an SVM model with a linear kernel
        svm_model = svm.SVC(kernel='linear')

        # Train the model on the train set
        for inputs, targets in train_dataloader:
            svm_model.fit(inputs, targets)

        # Make predictions on the test set
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        y_true = []
        y_pred = []
        for inputs, targets in test_dataloader:
            y_true.append(targets.item())
            y_pred.append(svm_model.predict(inputs.numpy())[0])

        # Evaluate the model's accuracy on the test set
        accuracy = accuracy_score(y_true, y_pred)
        print('Accuracy:', accuracy)

###############################################################################
# start_regression
###############################################################################
    def start_regression(self):
        pass

###############################################################################
# start_classification
###############################################################################
    def start_classification(self, dataset):
        
        model = MyNet(dataset.tensors[0].shape[1:].numel(), dataset.tensors[1].shape[1:].numel())
        criterion = BCEWithLogitsLoss()
        optimizer = SGD(model.parameters(), lr=0.1)
        
        train_dataset, valid_dataset, test_dataset = train_valid_test(dataset, 0.6, 0.2, 0.2)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3200, shuffle=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=3200, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

        dataloaders = {'train': train_loader, 'valid': val_loader}
        dataset_sizes= {'train': len(train_dataset), 'valid': len(valid_dataset)}

        # With a neural network
        model_final = self.train_and_valid_classif(model, criterion, optimizer, dataloaders, dataset_sizes, 5)
        self.test(model_final, test_loader, len(test_dataset))
        
        # With SVM
        #self.train_and_valid_classif_SVM(model, dataset)
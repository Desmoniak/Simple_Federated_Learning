# Time
import time

# Torch
import torch

# Sklearn
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

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
                accuracy = 0.0
                precision = 0.0
                recall = 0.0
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
                        accuracy += accuracy_score(labels, preds)
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
                accuracy_avg = accuracy / (i+1)

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

###############################################################################
# test
###############################################################################
    def test(self, model, test_loader, test_dataset_size):
        print("\n")
        print("_________________________TEST PHASE____________________")
        print("\n")
        
        accuracy = 0
        precision = 0.0
        recall = 0.0
        i = 0

        # Iterate over data.
        for inputs, labels in test_loader:
                model.eval()
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                accuracy += accuracy_score(labels, preds)
                precision += precision_score(labels, preds)
                recall += recall_score(labels, preds)


                i+=1

        ##Statistics
        # Calculate the average accuracy
        accuracy_avg = accuracy / (i+1)
        # Calculate the average precision and recall
        precision_avg = precision / (i+1)
        # Calculate the average recall
        recall_avg = recall / (i+1)


        # Print the average accuracy, precision, recall for once for train and val per epoch
        print('AVG Accuracy: %.4f || AVG Precision: %.3f || AVG Recall: %.3f' % 
        (accuracy_avg, precision_avg, recall_avg))
        
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
    def start_classification(self, model, criterion, optimizer, dataset, train_batch_size, valid_batch_size, num_epochs):
        model_to_train = model
        
        train_dataset, valid_dataset, test_dataset = train_valid_test(dataset, 0.6, 0.2, 0.2)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

        dataloaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes= {'train': len(train_dataset), 'val': len(valid_dataset)}

        model_final = self.train_and_valid_classif(model_to_train, criterion, optimizer, dataloaders, dataset_sizes, num_epochs)
        self.test(model_final, test_loader, len(test_dataset))
        #self.train_and_valid_classif_SVM(model, dataset)
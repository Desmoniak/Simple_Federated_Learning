# Time
import time

# Torch
import torch

# Sklearn
from sklearn.metrics import precision_score, recall_score

# Copy
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_and_test_nn_centralize(model, criterion, optimizer, dataloaders, dataset_sizes, batch_size, num_epochs=25):

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
                    correct += torch.sum(preds == torch.argmax(labels,1))
                    precision += precision_score(torch.argmax(labels,1), preds)
                    recall += recall_score(torch.argmax(labels,1), preds)

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

class Architecture():
    def __init__(self):
        pass

    def train(self, model, criterion, optimizer, dataloaders, dataset_sizes, batch_size, num_epochs=25):
        train_and_test_nn_centralize(model, criterion, optimizer, dataloaders, dataset_sizes, batch_size, num_epochs)
        
    def valid(self, x):
        pass
    
    def test(self, x):
        pass
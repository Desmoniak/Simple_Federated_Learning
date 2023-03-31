from sklearn.metrics import precision_score, recall_score
from Model import MyNet
from Node import Node
from Global_Node import Global_Node
from Architecture import Architecture

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

# Pandas
import pandas as pd

# Copy
import copy

# Time
import time

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
@Param value: a string value with this format "Hour(s):Minute(s):Second(s)" 
'''
def convert_to_seconds(value):
        if not any(char.isdigit() for char in value):
            return pd.NaT
        hour, minute, second = map(int, value.split(':'))
        if hour >= 15:
            hour = 0
        if minute >= 60:
            minute = 0
            hour += 1
        if second >= 60:
            second = 0
            minute += 1
        return hour*3600+minute*60+second
    
def train_and_test_nn(model, criterion, optimizer, dataloaders, dataset_sizes, batch_size, num_epochs=25):

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



def main():
    print("Hello World")
    
    dataset = pd.read_csv("./202207-divvy-tripdata.csv") # Replace by your dataset
    data = dataset.copy() # Copy

    data_prep = data.copy()

    # Convert the Series "ride_length" to second
    data_prep["ride_length"] = data_prep["ride_length"].apply(convert_to_seconds)

    # dropNaN
    data_prep = data_prep.dropna()
    
    data_prep['ride_length'] = data_prep['ride_length'].astype('int64')
    
    features = ['start_lat', 
            'start_lng',
            'end_lat',
            'end_lng',
            'member_casual',
            'ride_length',
            'day_of_week',
            'ride_id']
            
    target = ["rideable_type"]

    data_prep = data_prep.loc[(data_prep["rideable_type"] == "classic_bike") | (data_prep["rideable_type"] == "electric_bike")]
    X = data_prep[features]
    y = data_prep[target]

    y_OH = OneHotEncoder(handle_unknown='ignore', sparse=False).fit_transform(y)

    num_cols = ["start_lat", "start_lng", "end_lat", "end_lng", "day_of_week", "ride_length"]
    categorical_cols_less_values = ["member_casual"]


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat_less_values', OrdinalEncoder(), categorical_cols_less_values)
        ])

    X_prep = preprocessor.fit_transform(X)
    
    X_tensor = torch.tensor(X_prep, dtype=torch.float32)
    y_tensor = torch.tensor(y_OH, dtype=torch.float32)
    
    dataset_tensor = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    
    train_size = int(0.6 * len(dataset_tensor))
    val_size = len(dataset_tensor) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_tensor, [train_size, val_size])
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3200, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=3200, shuffle=True)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes= {'train': len(train_dataset), 'val': len(val_dataset)}

    model = MyNet(X_tensor.shape[1], y_tensor.shape[1])
    epochs = 20
    model = model.to(device)

    criterion = CrossEntropyLoss()
    optimizer_ft = Adam(model.parameters(), lr=0.001)

    # decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    model_ft = train_and_test_nn(model, criterion, optimizer_ft, dataloaders, dataset_sizes, 1, num_epochs=10)


if __name__ == "__main__":
    main()

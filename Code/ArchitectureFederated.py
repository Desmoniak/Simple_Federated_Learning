# Time
import time

# Torch
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler, SGD
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss


# Sklearn
from sklearn.metrics import precision_score, recall_score, accuracy_score
from Model import MyNet

from utility import train_valid_test

# Copy
import copy

# Random 
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ArchitectureFederated():
    def __init__(self):
        pass

###############################################################################
# train_and_test_node
###############################################################################
    def train_and_test_node(self, node, criterion, optimizer, batch_size_train, batch_size_test, num_epochs=25):
    
        train_loader = torch.utils.data.DataLoader(node['data']['train_data'], batch_size=batch_size_train, shuffle=True)
        val_loader = torch.utils.data.DataLoader(node['data']['valid_data'], batch_size=batch_size_test, shuffle=True)

        data_size = {
            'train': len(node['data']['train_data']), 
            'valid': len(node['data']['valid_data'])
        }
        dataloaders = {'train': train_loader, 'valid': val_loader}

        since = time.time()
        # Instantiate the neural network and the optimizer
        model = node['model']
        best_acc_avg = 0.0

        # Train the neural network
        for _ in range(num_epochs):
            """ print("\n")
            print("_________________________Epoch %d / %d ____________________" % (epoch+1, num_epochs))
            print("\n") """
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                correct = 0
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
                accuracy_avg = correct.double() / data_size[phase]

                # Print the average loss, accuracy, precision, recall for once for train and val per epoch
                """ print('PHASE %s:  [AVG loss: %.3f || AVG Accuracy: %.4f]' % 
                    (phase, loss_avg, accuracy_avg)) """


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
        return {'model': model, 'node_best_acc_avg': best_acc_avg}
    
##############################################################################
# send_global_model_to_node
###############################################################################
    def send_global_model_to_node(self, global_model, node):
        """Send the parameters of the global model to a local model of a node

        Args:
            global_model (_type_): _description_
            node (_type_): {'model': #, 'train_data': #, 'valid_data': #, 'test_data': #}

        Returns:
            _type_: _description_
        """
        node['model'].load_state_dict(global_model.state_dict())
    
###############################################################################
# selection_nodes
###############################################################################
    def selection_nodes(self, nb_nodes, nodes):
    
        nb_nodes_names = len(nodes.keys())
        if(nb_nodes > nb_nodes_names) : nb_nodes = nb_nodes_names
        elif(nb_nodes < 1) : nb_nodes = 1
        return random.sample(list(nodes.keys()), nb_nodes)
    
###############################################################################
# split_data_nodes
###############################################################################
    def split_data_nodes(self, nodes):
    
        for node in nodes.keys():
            train_dataset, valid_dataset, test_dataset = train_valid_test(nodes[node]['data'], 0.6, 0.2, 0.2)
            nodes[node]['data'] = {'train_data': train_dataset, 'valid_data': valid_dataset, 'test_data': test_dataset}
        return nodes

###############################################################################
# test
###############################################################################
    def test(self, node):
        test_loader = torch.utils.data.DataLoader(node['data']['test_data'], batch_size=len(node['data']['test_data']), shuffle=True)

        model = node['model']
        model.eval() # Set model to evaluate mode

        corrects = 0

        # Iterate over data.
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)


            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                corrects += torch.sum(labels.squeeze() == preds)

            # Print the average loss, accuracy, precision, recall for once for train and val per epoch
            """ print('AVG Accuracy: %.4f' % 
                (accuracy_avg)) """
            return corrects / len(node['data']['test_data'])

##############################################################################
# send_local_model_for_agg
###############################################################################
    def send_local_model_for_agg(self, global_model, nodes, node_in_training_mode):
        temp = node_in_training_mode.copy()
        name_params = nodes[temp[0]]['model'].named_parameters()
        state_dict = nodes[temp.pop(0)]['model'].state_dict()
        for name, _ in name_params:
            for node in temp:
                state_dict[name] = state_dict[name] + nodes[node]["model"].state_dict()[name]
            state_dict[name] = state_dict[name] / (len(node_in_training_mode))
        global_model.load_state_dict(state_dict)

###############################################################################
# train_and_test_classif
###############################################################################
    def train_and_test_classif(self, nodes, global_model, nb_round, nb_epoch, nb_nodes_selectioned):
        nodes_selectioned = self.selection_nodes(nb_nodes_selectioned, nodes)
        nodes_best_avg = {}
        node_before_after_agg = {}

        # We send the main model to the selectioned nodes. 
        for node in nodes_selectioned:
            self.send_global_model_to_node(global_model, nodes[node])

        for k in range(nb_round):
            nodes_best_avg[k] = {}
            node_before_after_agg[k] = {}
            print("\n")
            print("############################################################")
            print("_________________________Round %d / %d ____________________" % (k+1, nb_round))
            print("############################################################")
            print("\n")
            for node in nodes_selectioned:
                print(f"_________________________TRAINING PHASE of {node}____________________")
                criterion = nn.BCEWithLogitsLoss()
                optimizer = SGD(nodes[node]['model'].parameters(), lr=0.001)
                model_best_acc_avg = self.train_and_test_node(nodes[node], criterion, optimizer, 3200, 3200, num_epochs=nb_epoch)
                nodes_best_avg[k][node] = model_best_acc_avg['node_best_acc_avg']


            for node in nodes_selectioned:
                node_before_after_agg[k][node] = {"before_agg": self.test(nodes[node])}
            
            self.send_local_model_for_agg(global_model, nodes, nodes_selectioned)

            for node in nodes_selectioned:
                self.send_global_model_to_node(global_model, nodes[node])
                node_before_after_agg[k][node]["after_agg"] = self.test(nodes[node])

        for k in range(nb_round):
            print("_____________________________________________________________________")
            print(f"_________________________Results for round {k+1} ____________________")
            print("_____________________________________________________________________")
            for node in nodes_selectioned:
                print(f'Results for {node}')
                print("Best Accuracy")
                print(nodes_best_avg[k][node])
                print("\n")
                print("Comparaison before and after aggregation")
                print(node_before_after_agg[k][node])
                print("\n")

###############################################################################
# start_regression
###############################################################################
    def start_regression(self):
        pass

###############################################################################
# start_classification
###############################################################################
    def start_classification(self,dataset):
        global_model = MyNet(dataset.tensors[0].shape[1], dataset.tensors[1].shape[1])
        local_model = MyNet(dataset.tensors[0].shape[1], dataset.tensors[1].shape[1])
        
        node_1_data, node_2_data, node_3_data, node_4_data, _ = torch.utils.data.random_split(dataset, [10000, 10000, 10000, 10000, len(dataset) - 40000])


        nodes = {
            'node_1': {'model': copy.deepcopy(local_model), 'data': node_1_data},
            'node_2': {'model': copy.deepcopy(local_model), 'data': node_2_data},
            'node_3': {'model': copy.deepcopy(local_model), 'data': node_3_data},
            'node_4': {'model': copy.deepcopy(local_model), 'data': node_4_data}
        }
        
        nodes = self.split_data_nodes(nodes)
        model_global_final = self.train_and_test_classif(nodes, global_model, 3 , 100, 4)

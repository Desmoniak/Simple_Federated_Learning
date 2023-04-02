# Time
import time

# Torch
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss


# Sklearn
from sklearn.metrics import precision_score, recall_score

from utility import train_valid_test

# Copy
import copy

# Random 
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ArchitectureFederated():
    def __init__(self):
        pass

##############################################################################
# train_and_test_node
###############################################################################
    def train_and_test_node(self, node, criterion, optimizer, batch_size_train, batch_size_test, num_epochs=25):
    
        train_loader = torch.utils.data.DataLoader(node['data']['train_data'], batch_size=batch_size_train, shuffle=True)
        val_loader = torch.utils.data.DataLoader(node['data']['valid_data'], batch_size=batch_size_test, shuffle=True)

        dataloaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes= {'train': len(node['data']['train_data']), 'val': len(node['data']['train_data'])}

        since = time.time()
        # Instantiate the neural network and the optimizer
        model = node['model']
        optimizer = optimizer
        criterion = criterion
        best_acc_avg = 0.0

        #pbar = trange(num_epochs, unit="carrots")

        # Train the neural network
        for epoch in range(num_epochs):
            """ print("\n")
            print("_________________________Epoch %d / %d ____________________" % (epoch+1, num_epochs))
            print("\n") """
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
                """ print('PHASE %s:  [AVG loss: %.3f || AVG Accuracy: %.4f] [AVG Precision: %.3f || AVG Recall: %.3f]' % 
                    (phase, loss_avg, accuracy_avg, precision_avg, recall_avg)) """
                

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
        node['model'].load_state_dict(copy.deepcopy(global_model.state_dict()))
        return node
    
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
            train_dataset, valid_dataset, test_dataset = train_valid_test(nodes[node]['data'], 0.6, 0.3, 0.1)
            nodes[node]['data'] = {'train_data': train_dataset, 'valid_data': valid_dataset, 'test_data': test_dataset}
        return nodes

###############################################################################
# test
###############################################################################
    def test(self, node):
        test_loader = torch.utils.data.DataLoader(node['data']['test_data'], batch_size=len(node['data']['test_data']), shuffle=True)

        model = node['model']
        model.eval() # Set model to evaluate mode

        correct = 0
        precision = 0.0
        recall = 0.0
        i = 0

        # Iterate over data.
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            
            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                correct += torch.sum(preds == torch.argmax(labels,1))
                precision += precision_score(torch.argmax(labels,1), preds)
                recall += recall_score(torch.argmax(labels,1), preds)
                
                i+=1
            
            ##Statistics

            # Calculate the average accuracy
            accuracy_avg = correct.double() / len(node['data']['test_data'])

            # Calculate the average precision and recall
            precision_avg = precision / (i+1)

            # Calculate the average recall
            recall_avg = recall / (i+1)


            # Print the average loss, accuracy, precision, recall for once for train and val per epoch
            print('AVG Accuracy: %.4f || AVG Precision: %.3f || AVG Recall: %.3f' % 
                (accuracy_avg, precision_avg, recall_avg))
            return accuracy_avg

##############################################################################
# send_local_model_for_agg
###############################################################################
    def send_local_model_for_agg(self, global_model, nodes, node_in_training_mode):
        temp = node_in_training_mode.copy()
        state_dict = nodes[temp.pop(0)]['model'].state_dict()
        for node in node_in_training_mode:
            for name, param in nodes[node]["model"].named_parameters():            
                state_dict[name] = state_dict[name] + nodes[node]["model"].state_dict()[name]
            state_dict[name] = state_dict[name]/(len(node_in_training_mode)+1)
            global_model.load_state_dict(state_dict)
        return global_model

###############################################################################
# train_and_test_classif
###############################################################################
    def train_and_test_classif(self, nodes, global_model, nb_round, nb_epoch, nb_nodes_selectioned):
        nodes_selectioned = self.selection_nodes(nb_nodes_selectioned, nodes)
        nodes_best_avg = {}
        node_before_after_agg = {}
        # We send the main model to the selectioned nodes. 
        for node in nodes_selectioned:
            nodes[node] = self.send_global_model_to_node(global_model, nodes[node])
            
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
                criterion = nn.CrossEntropyLoss()
                optimizer = Adam(nodes[node]['model'].parameters(), lr=0.001)
                model_best_acc_avg = self.train_and_test_node(nodes[node], criterion, optimizer, 100, 100, num_epochs=nb_epoch)
                nodes[node]['model'] = model_best_acc_avg['model']
                nodes_best_avg[k][node] = model_best_acc_avg['node_best_acc_avg']
                
            
            for node in nodes_selectioned :
                node_before_after_agg[k][node] = {}
                print(f"_________________________TEST PHASE AVANT AGGREGATION {node}____________________")
                print("\n")
                node_before_after_agg[k][node]["before_agg"] = self.test(nodes[node])
                    
            global_model = self.send_local_model_for_agg(global_model, nodes, nodes_selectioned)
                
            for node in nodes_selectioned:
                self.send_global_model_to_node(global_model, nodes[node])
                
            for node in nodes_selectioned:
                print(f"_________________________TEST PHASE APRES AGGREGATION {node}____________________")
                node_before_after_agg[k][node]["after_agg"] = self.test(nodes[node])
                print("\n")
        print(nodes_best_avg)
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
    def start_classification(self, model, dataset):
        global_model = copy.deepcopy(model)
        local_model = copy.deepcopy(model)
        
        node_1_data, node_2_data, node_3_data, node_4_data, _ = torch.utils.data.random_split(dataset, [10000, 10000, 10000, 10000, len(dataset) - 40000])
        
        nodes = {
            'node_1': {'model': copy.deepcopy(local_model), 'data': node_1_data},
            'node_2': {'model': copy.deepcopy(local_model), 'data': node_2_data},
            'node_3': {'model': copy.deepcopy(local_model), 'data': node_3_data},
            'node_4': {'model': copy.deepcopy(local_model), 'data': node_4_data}
        }
        
        nodes = self.split_data_nodes(nodes)
        model_global_final = self.train_and_test_classif(nodes, global_model, 3, 100, 4)

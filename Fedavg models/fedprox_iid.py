import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn_extra.cluster import KMedoids
import pickle
import math
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 10, 3)
        self.fc1 = nn.Linear(24*24*10, 32)
        self.fc2 = nn.Linear(32, classes)
        self.flatten = nn.Flatten()
        self.track_layers = {'conv1': self.conv1, 'conv2': self.conv2, 'fc1': self.fc1, 'fc2': self.fc2}

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

    def get_track_layers(self):
        return self.track_layers

    def apply_parameters(self, parameters_dict):
        with torch.no_grad():
            for layer_name in parameters_dict:
                self.track_layers[layer_name].weight.data *= 0
                self.track_layers[layer_name].bias.data *= 0
                self.track_layers[layer_name].weight.data += parameters_dict[layer_name]['weight']
                self.track_layers[layer_name].bias.data += parameters_dict[layer_name]['bias']

    def get_parameters(self):
        parameters_dict = dict()
        for layer_name in self.track_layers:
            parameters_dict[layer_name] = {
                'weight': self.track_layers[layer_name].weight.data,
                'bias': self.track_layers[layer_name].bias.data
            }
        return parameters_dict

    def batch_accuracy(self, outputs, labels):
        with torch.no_grad():
            _, predictions = torch.max(outputs, dim=1)
            return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

    def _process_batch(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        accuracy = self.batch_accuracy(outputs, labels)
        return (loss, accuracy)

    def fit(self, dataset, epochs, lr, batch_size=128, opt=torch.optim.SGD, mu=0.01, global_params=None):
        dataloader = DeviceDataLoader(DataLoader(dataset, batch_size, shuffle=True), device)
        optimizer = opt(self.parameters(), lr)
        history = []
        for epoch in range(epochs):
            losses = []
            accs = []
            for batch in dataloader:
                loss, acc = self._process_batch(batch)
                
                if global_params is not None:
                    # FedProx proximal term
                    proximal_term = 0.0
                    for param, global_param in zip(self.parameters(), global_params):
                        proximal_term += (mu / 2) * torch.norm(param - global_param) ** 2
                    loss += proximal_term
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss.detach()
                losses.append(loss)
                accs.append(acc)
            avg_loss = torch.stack(losses).mean().item()
            avg_acc = torch.stack(accs).mean().item()
            history.append((avg_loss, avg_acc))
        return history

    def evaluate(self, dataset, batch_size=128):
        dataloader = DeviceDataLoader(DataLoader(dataset, batch_size), device)
        losses = []
        accs = []
        with torch.no_grad():
            for batch in dataloader:
                loss, acc = self._process_batch(batch)
                losses.append(loss)
                accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        return (avg_loss, avg_acc)

class DeviceDataLoader(DataLoader):
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)

class Client:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset
        self.net = to_device(ConvNet(), device)

    def get_dataset_size(self):
        return len(self.dataset)

    def get_client_id(self):
        return self.client_id

    def get_weights(self):
        return self.net.get_parameters()

    def train(self, parameters_dict, mu):
        self.net.apply_parameters(parameters_dict)
        global_params = self.net.parameters()
        #print(type(global_params))
        train_history = self.net.fit(self.dataset, epochs_per_client, learning_rate, batch_size, mu=mu, global_params=global_params)
        print('{}: Loss = {}, Accuracy = {}'.format(self.client_id, round(train_history[-1][0], 4), round(train_history[-1][1], 4)))
        return self.net.get_parameters()

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

train_data = MNIST(
    root='data',
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
test_data = MNIST(
    root='data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

total_train_size = len(train_data)
total_test_size = len(test_data)
classes = 10
input_dim = 784

clip_value = 0.2
epsilon_dp = 0.2
num_clients = 50
rounds = 50
batch_size = 128
epochs_per_client = 3
learning_rate = 2e-2
mu = 0.1  # FedProx proximal term parameter

device = get_device()

examples_per_client = total_train_size // num_clients
client_datasets = random_split(train_data, [min(i + examples_per_client, total_train_size) - i for i in range(0, total_train_size, examples_per_client)])
clients = [Client('client_' + str(i), client_datasets[i]) for i in range(num_clients)]
client_frac = [client.get_dataset_size() / total_train_size for client in clients]

global_net = to_device(ConvNet(), device)
history = []
train_acc_round = []
test_acc_round = []

for r in range(rounds):
    print('Start Round {} ...'.format(r + 1))
    curr_parameters = global_net.get_parameters()
    new_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in curr_parameters])
    for client in clients:
        client_parameters = client.train(curr_parameters, mu)
        fraction = client.get_dataset_size() / total_train_size
        for layer_name in client_parameters:
            new_parameters[layer_name]['weight'] += fraction * client_parameters[layer_name]['weight']
            new_parameters[layer_name]['bias'] += fraction * client_parameters[layer_name]['bias']
    global_net.apply_parameters(new_parameters)

    train_loss, train_acc = global_net.evaluate(train_data)
    train_acc_round.append(train_acc)

    test_loss, test_acc = global_net.evaluate(test_data)
    print("DP {} Test_accuracy: {}\n".format(epsilon_dp, test_acc))
    test_acc_round.append(test_acc)

    print('After round {}, train_loss = {}, train_acc= {}, test_acc = {}\n'.format(r + 1, round(train_loss, 4),
            round(train_acc, 4), round(test_acc, 4)))
    history.append(train_loss)

pickle.dump(global_net.get_parameters(), open("fedprox_model_after_{}_rounds.p".format(rounds), "wb"))
pickle.dump(history, open("fedprox_history_after_{}_rounds.p".format(rounds), "wb"))
pickle.dump(test_acc_round, open("fedprox_test_accuracy_after_{}_rounds.p".format(rounds), "wb"))
pickle.dump(train_acc_round, open("fedprox_train_accuracy_after_{}_rounds.p".format(rounds), "wb"))

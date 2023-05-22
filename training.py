import math
from unittest import result
import torch
from torch import device, nn, no_grad
import torch.nn.functional as F
import numpy as np
from models import *
import matplotlib.pyplot as plt
import pickle
import argparse
from data_generate import MyLinearDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment to train a neural network.")
    parser.add_argument("--hidden_layers", type=int, default=100, help="Number of hidden layer")
    parser.add_argument("--patterns", choices=["LU", "full"], default="LU")
    parser.add_argument("--n_iters", type=int, default=1000, help='Number of training iterations')
    return parser.parse_args()

def data_generate(matrix, batch_size):
    out_size, in_size = matrix.size()
    input = torch.rand((batch_size, in_size)) * 2 - 1
    #input = torch.where(input > 0, 1.0, -1.0)
    output = F.linear(input, matrix)
    return (input, output)

def LU_support(n):
    matrix = np.tensordot(np.arange(n), np.ones(n), axes = 0) - np.tensordot(np.ones(n), np.arange(n), axes=0)
    return np.where(matrix >= 0, 1, 0), np.where(matrix <= 0, 1, 0)

def singularLU(n):
    result = np.zeros((n,n))
    for i in range(n):
        result[i, n - i - 1] = 1.0
    return torch.from_numpy(result).float()

def training(model, gt_matrix, n_epoch = 5000, learning_rate = 0.01, batch_size = 256, optim_algo = 'SGD', \
            momentum = 0.9, interval_log = 10, weight_decay = 5e-4, gpu = False, data_size = 100000):
    if optim_algo == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum, weight_decay=weight_decay)
    elif optim_algo == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    
    in_size, out_size = gt_matrix.size()
    datapoints = torch.rand((data_size, in_size)) * 2 - 1
    dataset = MyLinearDataset(gt_matrix, datapoints)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    normfc1 = []
    normfc2 = []
    emp_loss = []
    true_loss = []
    active_neuron = []
    for k in range(1, n_epoch + 1):
        cumulative_loss = 0
        active_neurons = 0
        time_step = 0
        for (input, output) in data_loader:
            if gpu:
                input = input.to(device)
                output = output.to(device)
            loss = F.mse_loss(model(input), output)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                emp_loss.append((loss / torch.mean(torch.square(output))).item())
                true_loss.append(model.compare(gt_matrix.to(device)))
                cumulative_loss += emp_loss[-1] * len(input)
                active_neurons += model.fc1.active_neuron_cal(input)
                time_step += 1
        normfc1.append(model.fc1.weight_norm())
        normfc2.append(model.fc2.weight_norm())
        
        if k % interval_log == 0:
            print("Iteration ", k, ": ")
            print("Empirical loss: ", cumulative_loss / data_size)
            print("True loss: ", true_loss[-1])
            active_neuron.append(active_neurons / time_step)
            print(active_neuron[-1])
            print(normfc1[-1], normfc2[-1])
    return normfc1, normfc2, emp_loss, true_loss, active_neuron

if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available():
        gpu = True
    else:
        gpu = False

    dict = {}
    dict['normfc1'] = []
    dict['normfc2'] = []
    dict['emp_loss'] = []
    dict['true_loss'] = []
    dict['active_neuron'] = []
    
    n = args.hidden_layers
    if args.patterns == "LU":
        support1, support2 = LU_support(n)
    else:
        support1 = np.ones((n,n))
        support2 = np.ones((n,n))
    
    print(np.sum(support1), np.sum(support2))
    gt_matrix = singularLU(n)

    # Without weight decay
    # for i in range(10):    
    #     model = NNTwoLayerFixedSupport(support1, support2)
    #     if gpu:
    #         model = model.to(device)
    #     normfc1, normfc2, emp_loss, true_loss, active_neuron = training(model, gt_matrix, n_epoch = args.n_iters, learning_rate = 0.1, batch_size = 3000, optim_algo='SGD', weight_decay = 0, gpu = gpu)
    #     dict['normfc1'].append(normfc1)
    #     dict['normfc2'].append(normfc2)
    #     dict['emp_loss'].append(emp_loss)
    #     dict['true_loss'].append(true_loss)
    #     dict['active_neuron'].append(active_neuron)

    # with open('training_evo_' + args.patterns +'.pickle', 'wb') as handle:
    #     pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # handle.c:wqlose()

    dict = {}
    dict['normfc1'] = []
    dict['normfc2'] = []
    dict['emp_loss'] = []
    dict['true_loss'] = []
    dict['active_neuron'] = []

    # With weight decay
    for i in range(10):
        model = NNTwoLayerFixedSupport(support1, support2)
        if gpu:
            model = model.to(device)

        normfc1, normfc2, emp_loss, true_loss, active_neuron = training(model, gt_matrix, n_epoch = args.n_iters, learning_rate = 0.1, batch_size = 3000, optim_algo='SGD', gpu = gpu)
        dict['normfc1'].append(normfc1)
        dict['normfc2'].append(normfc2)
        dict['emp_loss'].append(emp_loss)
        dict['true_loss'].append(true_loss)
        dict['active_neuron'].append(active_neuron)
    
    with open('training_evo_regularisation_' + args.patterns + '.pickle', 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

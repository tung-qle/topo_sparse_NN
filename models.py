import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FixedSupportLinear(nn.Module):
    def __init__(self, support, bias = True):
        super().__init__()
        self.in_size = support.shape[0]
        self.out_size = support.shape[1]
        self.register_buffer("support", torch.from_numpy(support.astype(np.float32)))
        self.weight = nn.Parameter(torch.randn(support.shape) / np.sqrt(self.in_size))
        if bias:
            bias_shape = (self.out_size, ) 
            self.bias = nn.Parameter(torch.Tensor(*bias_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        x = F.linear(input, self.support * self.weight)
        if self.bias is not None:
            return x + self.bias
        return x 
    
    def weight_max(self):
        return np.max(np.abs((self.weight * self.support).detach().to("cpu").numpy()))
    
    def weight_norm(self):
        return np.linalg.norm((self.weight * self.support).detach().to('cpu').numpy())

    def return_weight(self):
        return (self.weight * self.support).detach().to('cpu').numpy()

    def return_bias(self):
        return self.bias.detach().to('cpu').numpy()

    def active_neuron_cal(self, input):
        return (torch.sum(torch.where(F.linear(input, self.weight * self.support, self.bias) > 0.0, 1.0, 0.0)) / (self.in_size * input.size()[0])).detach().to('cpu').numpy()

class NNTwoLayerFixedSupport(nn.Module):
    def __init__(self, support1, support2, bias = True):
        if support1.shape[0] != support2.shape[1]:
            raise Exception("Incompatible matrix size")
        super().__init__()
        self.fc1 = FixedSupportLinear(support1, bias)
        self.fc2 = FixedSupportLinear(support2, bias)
    
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        return x 
    
    def show_weight(self):
        print(self.fc2.weight * self.fc2.support)
        print(self.fc1.weight * self.fc1.support)

    def show_bias(self):
        print(self.fc1.bias)
        print(self.fc2.bias)

    def compare(self, gt_matrix):
        return (torch.norm(torch.matmul(self.fc2.weight * self.fc2.support, self.fc1.weight * self.fc1.support) - gt_matrix) / torch.linalg.norm(gt_matrix)).detach().to('cpu').numpy()
        
        
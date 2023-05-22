import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class MyLinearDataset(Dataset):
    def __init__(self, matrix, points):
        super(Dataset, self).__init__()
        self.points = points
        self.values = F.linear(points, matrix)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.points[index], self.values[index]
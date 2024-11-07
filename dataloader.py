import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32) 
        self.y = torch.tensor(y, dtype=torch.float32)  

    def __getitem__(self, index):
        # 返回一个样本及其标签
        return self.X[index], self.y[index]

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.X)

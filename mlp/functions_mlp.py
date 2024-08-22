# Libs

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Pandas option to dysplay all columns in dataframe

pd.set_option('display.max_columns', None)

# Make device agnostic code

def cuda_is_available():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return device

class ModelV0(nn.Module):
    
    def __init__(self, n_features, neurons_hidden, n_class_pred):
        
        super(ModelV0, self).__init__()
        
        self.layer_1 = nn.Linear(in_features = n_features, out_features = neurons_hidden, bias = True) # 20 features (dataset) to a next layer with 30 neurons
        self.layer_2 = nn.Linear(in_features = neurons_hidden, out_features = n_class_pred, bias = True) # layer with 30 neurons and define a next layer with 20 neurons
        
    def forward(self, x): # Define a forward method containing the forward pass computation
        
        x = torch.relu(self.layer_1(x))
        x = self.layer_2(x)
        
        return x
    
# Calculate accuracy (a classification metric)

def accuracy_fn(y_true, y_pred):
    
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc    
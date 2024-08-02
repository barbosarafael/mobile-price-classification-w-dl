# Libs

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

# Pandas option to dysplay all columns in dataframe

pd.set_option('display.max_columns', None)

# Make device agnostic code

def cuda_is_available():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return device

class ModelV0(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.layer_1 = nn.Linear(in_features = 20, out_features = 30, bias = True) # 20 features (dataset) to a next layer with 30 neurons
        self.layer_2 = nn.Linear(in_features = 30, out_features = 20, bias = True) # layer with 30 neurons and define a next layer with 20 neurons
        
    def forward(self, x): # Define a forward method containing the forward pass computation
        
        return self.layer_2(self.layer_1(x))
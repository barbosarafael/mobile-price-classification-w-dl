# 1) Read all functions and options from function script

from functions_mlp import *

# 2) Read train and test data

train = pd.read_csv('../data/train.csv')

# 3) MLP 

# a) Split train and test

X = train.drop(columns = 'price_range').to_numpy()
y = train[['price_range']].to_numpy()

# Transform to torch format

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.3,
                                                    random_state = 19, 
                                                    stratify = y)

device = cuda_is_available()

model_0 = ModelV0().to(device = device)    

print(model_0)

# Make predictions with the model

y_pred = model_0(X_test.to(device))
print(y_pred)

# Next topics: https://www.learnpytorch.io/02_pytorch_classification/
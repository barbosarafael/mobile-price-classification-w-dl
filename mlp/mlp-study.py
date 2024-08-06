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

# Device

device = cuda_is_available()

# Define the model

model_0 = ModelV0(n_features = 20, neurons_hidden = 10, n_class_pred = 4).to(device = device)    

print(model_0)

# Loss function

loss_function = nn.CrossEntropyLoss()

# Optimizer

learning_rate = 0.0001
optimizer = torch.optim.SGD(params = model_0.parameters(), 
                            lr = learning_rate)

# Traing

n_epochs = 1000

loss = []

torch.manual_seed(seed = 19)

# Put data to target device

y_train = y_train.view(-1).long()
y_test = y_test.view(-1).long()


X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(n_epochs):       
    
    # Clear gradient
    
    optimizer.zero_grad()
    
    # Training
    
    z = model_0.train()
    
    y_logits = model_0(X_train)
    p1 = torch.softmax(y_logits, dim=1)
    # print(p1)
    y_pred = torch.argmax(p1, dim = 1)
    
    # print(y_logits)
    # print(y_pred)
    
    loss_tmp = loss_function(y_logits, y_train) # -> RuntimeError: 0D or 1D target tensor expected, multi-target not supported
    
    acc = accuracy_fn(y_true = y_train, 
                      y_pred = y_pred) 
    
    
    # print(loss_tmp)
    
    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        
        print(f"Epoch: {epoch} | Loss: {loss_tmp:.5f}, Accuracy: {acc:.2f}%")
# 1) Read all functions and options from function script

from functions_mlp import *

# 2) Read train and test data

train = pd.read_csv('../data/train.csv')

# 3) MLP 

# Device

device = cuda_is_available()

# a) Split train and test

X = train.drop(columns = 'price_range').to_numpy()
y = np.array(train['price_range'].to_list())

# Split 

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.3,
                                                    random_state = 19, 
                                                    stratify = y)

# Scaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to torch tensor

X_train_tensor = torch.tensor(X_train, dtype = torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype = torch.int64).to(device)
y_train_tensor = torch.tensor(y_train, dtype = torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype = torch.int64).to(device)

# Instantiate the model

in_features = X_train.shape[1]
num_class = len(set(y))

model_0 = ModelV0(n_features = in_features, 
                  neurons_hidden = 10,
                  n_class_pred = num_class).to(device = device)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.SGD(params = model_0.parameters(), 
                            lr = learning_rate)

num_epochs = 1000

for epoch in range(num_epochs):
    
    model_0.train()
    
    # Forward pass
    
    outputs = model_0(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    _, predicted_labels = torch.max(outputs, 1)
    correct_predictions = (predicted_labels == y_train_tensor).sum().item()
    total_samples = len(y_train_tensor)
    acc = correct_predictions/total_samples
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print out what's happening every 10 epochs
    if epoch % 100 == 0:
        
        print(f"Epoch: {epoch} | Loss: {loss.item()} | Acc: {acc}")
        
# TODO: see the next steps https://www.youtube.com/watch?v=iWdVXAwurXs
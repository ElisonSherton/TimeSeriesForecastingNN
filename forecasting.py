from visualizeResults import visualize_results
import torch
from math import sqrt
from torch import nn
from curateData import curateData, standardizeData, getDL, get_preds
from train_val_split import train_val_split
from extractData import extractHistory
from datetime import date
from RNN_forecaster import forecasterModel
import pandas as pd


SYMBOL = "SBIN"
pth = f"./data/{SYMBOL}.csv"
start_date = date(2020, 1, 1)
end_date = date(2020, 10, 18)
hidden_dim = 100
rnn_layers = 2
dropout = 0.1
train_pct = 0.7

params = {"batch_size": 16,
         "shuffle": False,
         "num_workers": 4}

n_epochs = 100
n_lags = 3
learning_rate = 1e-2
device = "cpu"

# Extract data for the ticker mentioned above
extractHistory(SYMBOL, start_date, end_date, pth)

# Get the inputs and outputs from the extracted ticker data
inputs, labels, dates = curateData(pth, "Close", "Date", n_lags)
N = len(inputs)

# Perform the train test validation split
trainX, trainY, valX, valY = train_val_split(inputs, labels, train_pct)

# Standardize the data to bring the inputs on a uniform scale
trnX, SS_ = standardizeData(trainX, train = True)
valX, _ = standardizeData(valX, SS_)

# Create dataloaders for both training and validation datasets
training_generator = getDL(trnX, trainY, params)
validation_generator = getDL(valX, valY, params)

# Create the model
model = forecasterModel(n_lags, hidden_dim, rnn_layers, dropout).to(device)

# Define the loss function and the optimizer
loss_func = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Track the losses across epochs
train_losses = []
valid_losses = []

# Training loop 
for epoch in range(1, n_epochs + 1):
    ls = 0
    valid_ls = 0
    # Train for one epoch
    for xb, yb in training_generator:
        # Perform the forward pass operation
        ips = xb.unsqueeze(0)
        targs = yb
        op = model(ips)
        
        # Backpropagate the errors through the network
        optim.zero_grad()
        loss = loss_func(op, targs)
        loss.backward()
        optim.step()
        ls += (loss.item() / ips.shape[1])
    
    # Check the performance on valiation data
    for xb, yb in validation_generator:
        ips = xb.unsqueeze(0)
        ops = model.predict(ips)
        vls = loss_func(ops, yb)
        valid_ls += (vls.item() / xb.shape[1])

    rmse = lambda x: round(sqrt(x * 1.000), 3)
    train_losses.append(str(rmse(ls)))
    valid_losses.append(str(rmse(valid_ls)))
    
    # Print the total loss for every tenth epoch
    if (epoch % 10 == 0) or (epoch == 1):
        print(f"Epoch {str(epoch):<4}/{str(n_epochs):<4} | Train Loss: {train_losses[-1]:<8}| Validation Loss: {valid_losses[-1]:<8}")

# Make predictions on train, validation and test data and plot 
# the predictions along with the true values 

to_numpy = lambda x, y: (x.squeeze(0).numpy(), y.squeeze(0).numpy())
train_preds, train_labels = get_preds(training_generator, model)
train_preds, train_labels = to_numpy(train_preds, train_labels)

val_preds, val_labels = get_preds(validation_generator, model)
val_preds, val_labels = to_numpy(val_preds, val_labels)

visualize_results((train_preds, val_preds), (train_labels, val_labels), SYMBOL, 
                   f"./img/{SYMBOL}_predictions.png", f"./predictions/{SYMBOL}_predictions.csv", dates)
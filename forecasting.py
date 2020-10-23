from visualizeResults import visualize_results
import torch
from math import sqrt
from torch import nn
from curateData import curateData, standardizeData, getDL, get_preds
from train_val_split import train_val_split
from extractData import extractHistory
from datetime import date
from RNN_forecaster import forecasterModel
import streamlit as st
import pandas as pd

# Display info about the app
st.title("Analysing Indian Stocks using nsepy & PyTorch LSTMs")
st.image(image = "./img/stock_market.jpg", width = 650)

with open("./instructions.md", "r") as f:
    info = "".join(f.readlines())
st.markdown(info)

# Read info about all stocks in NSE and provide a dropdown to select one
stock_ticker_info = pd.read_csv("./data/EQUITY_L.csv")
append_ = lambda x: f"{x['NAME OF COMPANY']} -> {x['SYMBOL']}"
all_tickers = list(stock_ticker_info.apply(append_, axis = 1))

SYMBOL = st.selectbox(label = "Select equity to analyze",
                      options = all_tickers)
SYMBOL = SYMBOL.split(" -> ")[-1]
pth = f"./data/{SYMBOL}.csv"

# Provide a date field to select a start date and an end date
col1, col2 = st.beta_columns(2)
start_date = col1.date_input(label = "Select start date from which to get historical data", 
                           value = date(2020, 1, 1),
                           min_value = date(2019, 1, 1),
                           max_value = date.today())
end_date = col2.date_input(label = "Select end date upto which to fetch historical data", 
                           min_value = date(2019, 1, 1),
                           max_value = date.today())

# Provide sliders for configuring LSTM hyperparameters
col3, col4, col5 = st.beta_columns(3)

hidden_dim = col3.slider(label = "Neurons in hidden layer of the LSTM",
                        value = 80,
                        min_value = 20,
                        max_value = 150,
                        step = 5)

rnn_layers = col4.slider(label = "Number of RNN hidden layers",
                        value = 2,
                        min_value = 1,
                        max_value = 5,
                        step = 1)

dropout = col5.slider(label = "Dropout percentage",
                        value = 0.1,
                        min_value = 0.0,
                        max_value = 0.5,
                        step = 0.01)

# Provide sliders for configuring training hyperparameters
col6, col7 = st.beta_columns(2)
n_epochs = col6.slider(label = "Number of epochs to train",
                        value = 100,
                        min_value = 10,
                        max_value = 300,
                        step = 10)

batch_sz = col7.slider(label = "Minibatch size",
                        value = 16,
                        min_value = 8,
                        max_value = 64,
                        step = 4)

col8, col9 = st.beta_columns(2)
n_lags = col8.slider(label = "Number of historical timesteps to consider",
                        value = 8,
                        min_value = 1,
                        max_value = 10,
                        step = 1)

learning_rate = col9.slider(label = "Learning rate for the model",
                        value = 5e-2,
                        min_value = 1e-2,
                        max_value = 1e-1,
                        step = 1e-2)


params = {"batch_size": batch_sz,
         "shuffle": False,
         "num_workers": 4}

train_pct = 0.7
device = "cpu"

if st.button("Submit"):
    st.write(f"Extracting data for {SYMBOL} with nsepy")
    # Extract data for the ticker mentioned above
    extractHistory(SYMBOL, start_date, end_date, pth)

    # Get the inputs and outputs from the extracted ticker data
    inputs, labels, dates = curateData(pth, "Close", "Date", n_lags)
    N = len(inputs)

    # Perform the train validation split
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
    st.write("Extracted data, now training the model...")
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
            st.write(f"Epoch {str(epoch):<4}/{str(n_epochs):<4} | Train Loss: {train_losses[-1]:<8}| Validation Loss: {valid_losses[-1]:<8}")

    # Make predictions on train, validation and test data and plot 
    # the predictions along with the true values 
    to_numpy = lambda x, y: (x.squeeze(0).numpy(), y.squeeze(0).numpy())
    train_preds, train_labels = get_preds(training_generator, model)
    train_preds, train_labels = to_numpy(train_preds, train_labels)

    val_preds, val_labels = get_preds(validation_generator, model)
    val_preds, val_labels = to_numpy(val_preds, val_labels)

    visualize_results((train_preds, val_preds), (train_labels, val_labels), SYMBOL, 
                    f"./img/{SYMBOL}_predictions.png", f"./predictions/{SYMBOL}_predictions.csv", dates)
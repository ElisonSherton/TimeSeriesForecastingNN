import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from load_data import stockTickerDataset

def curateData(pth, price_col, date_col, n_steps):
    """Reads the dataset and based on n_steps/lags to consider in the time series, creates input output pairs

    Args:
        pth ([str]): [Path to the csv file]
        price_col ([str]): [The name of column in the dataframe that holds the closing price for the stock]
        date_col ([str]): [The nameo oc column in the dataframe which holds dates values]
        n_steps ([int]): [Number of steps/ lags based on which prediction is made]
    """
    df = pd.read_csv(pth)

    # Create lags for the price column
    for idx in range(n_steps):
        df[f"lag_{idx + 1}"] = df[price_col].shift(periods = (idx + 1))
    
    # Create a dataframe which has only the lags and the date
    new_df = df[[date_col, price_col] + [f"lag_{x + 1}" for x in range(n_steps)]]
    new_df = new_df.iloc[n_steps:-1, :]

    # Get a list of dates for which these inputs and outputs are
    dates = list(new_df[date_col])

    # Create input and output pairs out of this new_df
    ips = []
    ops = []
    for entry in new_df.itertuples():
        ip = entry[-n_steps:][::-1]
        op = entry[-(n_steps + 1)]
        ips.append(ip)
        ops.append(op)

    return (ips, ops, dates)

def standardizeData(X, SS = None, train = False):
    """Given a list of input features, standardizes them to bring them onto a homogenous scale

    Args:
        X ([dataframe]): [A dataframe of all the input values]
        SS ([object], optional): [A StandardScaler object that holds mean and std of a standardized dataset]. Defaults to None.
        train (bool, optional): [If False, means validation set to be loaded and SS needs to be passed to scale it]. Defaults to False.
    """
    if train:
        SS = StandardScaler()   
        new_X = SS.fit_transform(X)
        return (new_X, SS)
    else:
        new_X = SS.transform(X)
        return (new_X, None)

def getDL(x, y, params):
    """Given the inputs, labels and dataloader parameters, returns a pytorch dataloader

    Args:
        x ([list]): [inputs list]
        y ([list]): [target variable list]
        params ([dict]): [Parameters pertaining to dataloader eg. batch size]
    """
    training_set = stockTickerDataset(x, y)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    return training_generator

def get_preds(generator, model):
    """Given a pytorch neural network model and a generator object, extracts predictions and returns the same

    Args:
        generator ([object]): [A pytorch dataloader which holds inputs on which we wanna predict]
        model ([object]): [A pytorch model with which we will predict stock prices on input data]

    """
    all_preds = []
    all_labels = []
    all_ips = []
    for xb, yb in generator:
        ips = xb.unsqueeze(0)
        ops = model.predict(ips)
        all_preds.append(ops)
        all_ips.append(ips)
        all_labels.append(yb)
    return (torch.cat(all_preds), torch.cat(all_labels))
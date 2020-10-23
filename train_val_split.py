def train_val_split(x, y, train_pct):
    """Given the input x and output labels y, splits the dataset into train, validation and test datasets

    Args:
        x ([list]): [A list of all the input sequences]
        y ([list]): [A list of all the outputs (floats)]
        train_pct ([float]): [% of data in the test set]
    """
    # Perform a train test split (It will be sequential here since we're working with time series data)
    N = len(x)
    
    trainX = x[:int(train_pct * N)]
    trainY = y[:int(train_pct * N)]

    valX = x[int(train_pct * N):]
    valY = y[int(train_pct * N):]

    return (trainX, trainY, valX, valY)
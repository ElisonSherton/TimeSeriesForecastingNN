import torch

class stockTickerDataset(torch.utils.data.Dataset):
    """This class is the dataset class which is used to load data for training the LSTM 
    to forecast timeseries data
    """

    def __init__(self, inputs, outputs):
        """Initialize the class with instance variables

        Args:
            inputs ([list]): [A list of tuples representing input parameters]
            outputs ([list]): [A list of floats for the stock price]
        """
        self.inputs = inputs
        self.outputs = outputs
    
    def __len__(self):
        """Returns the total number of samples in the dataset
        """
        return len(self.outputs)
    
    def __getitem__(self, idx):
        """Given an index, it retrieves the input and output corresponding to that index and returns the same

        Args:
            idx ([int]): [An integer representing a position in the samples]
        """
        x = torch.FloatTensor(self.inputs[idx])
        y = torch.FloatTensor([self.outputs[idx]])
        
        return (x, y)
    

    


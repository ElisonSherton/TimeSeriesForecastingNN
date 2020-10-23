from nsepy import get_history
from datetime import date

def extractHistory(SYMBOL, start_date, end_date, pth):
    """Given the symbol of the stock for which to fetch data, the start and end dates, extracts historical data for this specified time
    period in a dataframe and saves the same to the specified path

    Args:
        SYMBOL ([str]): [The ticker symbol for the stock whose data is to be fetched]
        start_date ([date]): [date from which to extract historical prices]
        end_date ([date]): [date upto which to extract historical prices]
        pth ([str]): [The path where to save the dataframe]
    """
    data = get_history(symbol = SYMBOL, start = start_date, end = end_date)
    data.to_csv(pth)
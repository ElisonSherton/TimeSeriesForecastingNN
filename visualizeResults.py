import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
plt.style.use("fivethirtyeight")

def visualize_results(preds, labels, sym, img_pth, pred_pth, dates):
    """[Given predictions and labels for training and validation datasets, visualizes them in a plot]

    Args:
        preds ([list]): [Predicted values of the stock prices]
        labels ([list]): [True values of the stock prices]
        img_pth ([str]): [A string representing the path where to save the visualization]
        pred_pth ([str]): [A string representing the path where to save the predictions file]
        sym ([str]): [A string representing the ticker symbol for this stock]
        dates ([list]): [a list of dates as strings]
    """
    train_preds, val_preds = preds[0], preds[1]
    train_labels, val_labels = labels[0], labels[1]

    # Format the predictions into a dataframe and save them to a file in the predictions folder
    all_preds = np.concatenate((train_preds,val_preds))
    all_labels = np.concatenate((train_labels,val_labels))
    flags = ["train"] * len(train_labels) + ["valid"] * len(val_labels)

    df = pd.DataFrame([(x[0], y[0]) for x, y in zip(all_preds, all_labels)], columns = ["Predictions", "Ground Truth"])
    df["Type"] = flags
    df.index = dates
    df.to_csv(pred_pth)
    st.write("Predictions for the last five timestamps...")
    st.dataframe(df.tail(5), width = 600, height = 800)

    # Find out the first element which belongs to validation dataset to depict the same manually
    dt = None
    for idx, item in enumerate(df.Type):
        if item == "valid":
            dt = df.index[idx]
            break
    
    # Create the plot and save it to the path provided as an argument above
    fs = 32
    plt.figure(figsize = (40,10))
    plt.plot(df.index, df["Predictions"], color = "blue")
    plt.plot(df.index, df["Ground Truth"], color = "green")
    plt.legend(["Predictions", "Ground Truth"], fontsize = fs)
    plt.axvline(x = dt, linestyle = "--")
    plt.xticks(rotation = 90)
    plt.xlabel("Dates", fontsize = fs)
    plt.ylabel("Stock prices", fontsize = fs)
    plt.title(f"{sym} stock prices data - LSTM predictions", fontsize = fs)
    plt.tight_layout()
    plt.savefig(img_pth)
    plt.close()
    st.image(image = img_pth, caption = f"{sym} forecast analysis", width = 800)
import pandas as pd 
import os 

def load_data(data_dir):

    df = pd.read_csv(data_dir)

    return df


if __name__ == "__main__":

    DATA_DIR = os.path.join('data', 'UCI_Credit_Card.csv')

    data = load_data(DATA_DIR)

    print(data.head())
    print(data.info())
    print(data.shape)
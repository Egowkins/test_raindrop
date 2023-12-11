import pandas as pd

def optimization(train):

    for column in train.columns:
        train[column] = train[column].astype(str)
        train[column] = train[column].str.replace(',', '.')
        train[column] = pd.to_numeric(train[column], errors='coerce')
        print("!")

    return train

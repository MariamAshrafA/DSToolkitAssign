import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')

def fill_nulls(train_df):
    train = train_df.fillna(0)
    return train

"""
	Encode String
"""

"""
	Modelling
"""

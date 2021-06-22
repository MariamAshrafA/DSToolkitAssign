import pandas as pd
import numpy as np


from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('train.csv')

def fill_nulls(train_df):
    train = train_df.fillna(0)
    return train

"""
	Encode String
"""
train.set_index('PassengerId', inplace = True)
train.drop(columns = ['Name', 'Ticket', 'Cabin'], inplace = True)
train['Sex'] = train['Sex'].apply(lambda x : 1 if x == 'female' else 0)
train['Embarked'] = train['Embarked'].apply(lambda x : 1 if x == 'S' else 2 if x == 'C' else 3)

"""
	Modelling
"""
X = train.loc[:, train.columns != 'Survived']
y = train['Survived']

clf = DecisionTreeClassifier(random_state = 42)

clf.fit(X, y)

import pandas as pd
import numpy as np


from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('train.csv')

"""
	Fill nulls
"""

"""
	Encode String
"""

"""
	Modelling
"""
X = train.loc[:, train.columns != 'Survived']
y = train['Survived']

clf = DecisionTreeClassifier(random_state = 42)

clf.fit(X, y)

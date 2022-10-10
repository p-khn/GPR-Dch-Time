# Data Wrangling
import pandas as pd

# Feature Selection
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# Get data
def get_data(path='battery_data.csv'):
    data = pd.read_csv(path,
                       index_col=None,
                       dtype={'Name': str, 'Value': float})

    return data

# Separate data
def separate_it(data, target):
    X = data.drop(target, axis=1)
    y = data.loc[:, target]

    return X, y

# Feature selection using Recursive Feature Elimination
def feature_selection_RFE(data, target, n_features, step=1):
    X, y = separate_it(data, target)
    rfe_selector = RFE(estimator=RandomForestRegressor(),
                       n_features_to_select=n_features, step=step)
    rfe_selector.fit(X, y)
    data = data.loc[:, X.columns[rfe_selector.get_support()]]
    data['D_T'] = y

    return data

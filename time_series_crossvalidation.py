from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np

import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

data = sns.load_dataset('taxis')


# Data sorted by time
data = data.sort_values('pickup').reset_index()
# Split dependent and predictors
X = data.loc[:, ['pickup', 'distance']]
y = data.loc[:, 'fare']


tscv = TimeSeriesSplit(n_splits=5, test_size=100, gap=0)

# This is just a walkthrough the cv to explore and be sure what it will do.
# No need of this to actually use the Split in the CV.
for train_index, test_index in tscv.split(X):
    print("TRAIN:", len(train_index), "from: ", min(train_index),
          "to: ", max(train_index), "TEST:", len(test_index))
    X_train, X_test = pd.DataFrame(X.loc[train_index, 'distance']), pd.DataFrame(
        X.loc[test_index, 'distance'])
    y_train, y_test = y.loc[train_index], y.loc[test_index]


# Pipeline in case we add more transformers.
standard_transformer = Pipeline(steps=[
    ('standard', StandardScaler())])

# We apply the transformers just to distance variable using ColumnTransformer.
# Not keeping pickupdate because not useful as it is for quick regression
preprocessor = ColumnTransformer(
    # remainder='passthrough', #passthough features not listed
    transformers=[
        ('std', standard_transformer, ['distance']),

    ])


# Simplet function to  put up a pipeline.
clf = make_pipeline(preprocessor, LinearRegression())

# Crossvalidation using the time series split
cross_val_score(clf, X, y, cv=tscv)

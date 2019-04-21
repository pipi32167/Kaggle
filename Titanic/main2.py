# Regression Template

# Importing the libraries
import statsmodels.formula.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer
import pandas as pd


def dummy(X, i):
    labelencoder = LabelEncoder()
    X[:, i] = labelencoder.fit_transform(X[:, i])
    onehotencoder = OneHotEncoder(categorical_features=[i])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]
    # print(X)
    return X


def backwardElimination(X, y, SL):
    numVars = len(X[0])
    for i in range(numVars):
        regressor_OLS = sm.OLS(y, X).fit()
        maxVal = max(regressor_OLS.pvalues).astype(float)
        if maxVal <= SL:
            break

        for j in range(numVars):
            if regressor_OLS.pvalues[j].astype(float) == maxVal:
                X = np.delete(X, j, 1)
                break
    print(regressor_OLS.summary())
    return X


# Importing the dataset
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

pickIdx = [2, 4, 5, 6, 7, 9, 11]
X_train = train.iloc[:, pickIdx].values
y_train = train.iloc[:, 1].values
# X_test = test.iloc[:, pickIdx].values
# y_test = test.iloc[:, 1].values

print(X_train[:, 6])

labelencoder = LabelEncoder()
X_train[:, 0] = labelencoder.fit_transform(X_train[:, 0])
X_train[:, 1] = labelencoder.fit_transform(X_train[:, 1])
X_train[:, 6] = labelencoder.fit_transform(X_train[:, 6])

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
X_train[:, :] = imputer.fit(X_train[:, :]).transform(X_train[:, :])

onehotencoder = OneHotEncoder(categorical_features=[0])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]

SL = 0.05
X_opt = X_train[:, :]
X_Modeled = backwardElimination(X_opt, y_train, SL)
print(X_Modeled)


# X_train = dummy(X_train, 0)
# print(X_train)
# X_train = dummy(X_train, 1)
# print(X_train)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
# Create your regressor here

# # Predicting a new result
# y_pred = regressor.predict(6.5)

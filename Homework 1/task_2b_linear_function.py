import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the excel file
data = pd.read_excel('data.xlsx', sheet_name='data')

# Settings
X = data.iloc[:, 2:10].values   # We choose Col 3-10 as our features
Y = data.iloc[:, 11:14].values  # We choose Col 12-14 as our labels

# The session number is used as the weights for the weighted linear regression
weights = data['Session number'].values

for i in range(Y.shape[1]):
    # Fit weighted linear regression for each column using all data
    regressor = LinearRegression()
    regressor.fit(X, Y[:, i], sample_weight=weights)

    # Print the linear equation for the regression model
    print(f"\nLinear regression model for Column {12 + i}:")
    print(f"Formula: y = {' + '.join([f'{coef:.3f}*X{i+3}' for i, coef in enumerate(regressor.coef_)])} + {regressor.intercept_:.3f}")

    # Predict on the same data (since we're using all the data for training and error calculation)
    y_pred = regressor.predict(X)

    # Calculate MSE on all data
    mse = mean_squared_error(Y[:, i], y_pred)
    print(f"Mean Squared Error for Column {12 + i}: {mse:.3f}")
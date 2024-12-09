import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the excel file
data = pd.read_excel('data.xlsx', sheet_name='data')

# Settings
X = data.iloc[:, 2:10].values   # We choose Col 3-10 as our features
Y = data.iloc[:, 11:14].values  # We choose Col 12-14 as our labels

# The session number is used as the weights for the weighted linear regression
weights = data['Session number'].values

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test, weights_train, weights_test = train_test_split(
    X, Y, weights, test_size=0.2, random_state=0
)

# Create subplots for each of the three columns to predict (Col 12, 13, 14)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns for the subplots

# Train models for each dependent variable (Col 12, 13, 14)
for i in range(Y_train.shape[1]):
    # Fit weighted linear regression for each column
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train[:, i], sample_weight=weights_train)

    # Print the linear equation for the regression model
    print(f"\nLinear regression model for Column {12 + i}:")
    print(f"Formula: y = {' + '.join([f'{coef:.3f}*X{i+3}' for i, coef in enumerate(regressor.coef_)])} + {regressor.intercept_:.3f}")

    # Predict on the test set
    y_pred = regressor.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_pred - Y_test[:, i])**2))
    print(f"Root Mean Squared Error for Column {12 + i}: {rmse:.3f}")

    # Plot the results
    axs[i].scatter(Y_test[:, i], y_pred, color='blue', alpha=0.6)
    axs[i].plot([Y_test[:, i].min(), Y_test[:, i].max()], [Y_test[:, i].min(), Y_test[:, i].max()], 'k--', lw=2)
    axs[i].set_xlabel(f'Actual values (Col {12 + i})')
    axs[i].set_ylabel(f'Predicted values (Col {12 + i})')
    axs[i].set_title(f'Regression for Col {12 + i}\nRMSE = {rmse:.3f}')

# Adjust spacing between the plots
plt.tight_layout()

# Show the plots
plt.show()

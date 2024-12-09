import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Settings
show_plot = True
decimals = 5
independent_value = 'Average year'
dependent_value ='Night-chat ratio'
regressors = []
errors = []

# Load the excel file
data = pd.read_excel('data.xlsx', sheet_name='data')

# I assume we will still filter out data where there is is less than 20 messages
filtered_data = data[data['Session number'] >= 20]

# We do a scatterplot of the filtered data
plt.scatter(filtered_data[independent_value], filtered_data[dependent_value], color='blue')

# Split the data into 10 equal parts
data_chunks = np.array_split(filtered_data, 10)

# Perform linear regression against our dependent values
for index, chunk in enumerate(data_chunks):
    x = chunk[independent_value].values.reshape(-1, 1)
    y = chunk[dependent_value].values

    # Split data into training and testing data (80% training, 20% testing)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    # Create a linear regression model
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Predict the values
    y_pred_test = regressor.predict(x_test)
    y_pred_train = regressor.predict(x_train)

    # Calclate error
    mse = mean_squared_error(y_test, y_pred_test)

    # Print information about our regression model
    print("\n------------------------------------------------------------")
    print(f"Linear regression no. {index}")
    print(f"Formula:            y = {round(regressor.coef_[0], decimals)}x + {round(regressor.intercept_, decimals)}")
    print(f"Mean squared error: {round(mse, decimals)}")

    # We add the regressor and error to our lists
    regressors.append(regressor)
    errors.append(mse)

    # Plot the data
    plt.plot(x_train, y_pred_train, color='gray')

# Finally a we average the regressors and plot the average regression
average_regressor = LinearRegression()
average_regressor.coef_ = np.mean([regressor.coef_ for regressor in regressors], axis=0)
average_regressor.intercept_ = np.mean([regressor.intercept_ for regressor in regressors], axis=0)
print("\n------------------------------------------------------------")
print(f"Average linear regression")
print(f"Formula:            y = {round(average_regressor.coef_[0], decimals)}x + {round(average_regressor.intercept_, decimals)}")
print(f"Mean squared error: {round(np.mean(errors), decimals)}")
x = filtered_data[independent_value].values.reshape(-1, 1)
plt.plot(x, average_regressor.predict(x), color='black')
plt.show()
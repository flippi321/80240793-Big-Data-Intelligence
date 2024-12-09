import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Settings
show_plot = True
decimals = 5
independent_value = 'Average year'
dependent_values = ['No-response ratio', 'Night-chat ratio', 'Picture ratio']
plot_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

# Load the excel file
data = pd.read_excel('data.xlsx', sheet_name='data')

# We don't want any data where column 11 isn't greater than or equal to 20
filtered_data = data[data['Session number'] >= 20]

# We create subplots for each linear regrssion
fig, axs = plt.subplots(-(-len(dependent_values) // 2), 2)

# Perform linear regression against our dependent values
for column in dependent_values:
    index = dependent_values.index(column)
    this_axs = axs[index // 2][index % 2]

    x = filtered_data[independent_value].values.reshape(-1, 1)
    y = filtered_data[column].values

    # Train a linear Regression
    regressor = LinearRegression()
    regressor.fit(x, y)

    # Predict the values for the model
    y_pred = regressor.predict(x)

    # Calculate error for the model
    mse = mean_squared_error(y, y_pred)

    # We calculate the correlation between the dependent and independent vales
    correlation, _ = pearsonr(filtered_data[independent_value], filtered_data[column])

    # Print information about our regression model
    print("\n------------------------------------------------------------")
    print(f"Linear regression model for {column}")
    print(f"Formula:            y = {round(regressor.coef_[0], decimals)}x + {round(regressor.intercept_, decimals)}")
    print(f"Mean squared error: {round(mse, decimals)}")
    print(f"Correlation:        {round(correlation, decimals)}")

    # Plot the data
    this_axs.scatter(x, y, color=plot_colors[index])
    this_axs.plot(x, y_pred, color='black')
    this_axs.set_xlabel(independent_value)
    this_axs.set_ylabel(column)

plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.suptitle(f'{independent_value} compared to dependent values')
plt.show()
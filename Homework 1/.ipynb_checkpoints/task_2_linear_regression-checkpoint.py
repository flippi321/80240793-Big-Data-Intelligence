import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

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
    print(f"Linear regression model for {column}")
    print(f"Formula:            y = {round(regressor.coef_[0], decimals)}x + {round(regressor.intercept_, decimals)}")
    print(f"Mean squared error: {round(mse, decimals)}")

    # Plot the data
    this_axs.scatter(x_train, y_train, color=plot_colors[index])
    this_axs.plot(x_train, y_pred_train, color='black')
    this_axs.set_xlabel(independent_value)
    this_axs.set_ylabel(column)

plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.suptitle(f'{independent_value} compared to dependent values')
plt.show()
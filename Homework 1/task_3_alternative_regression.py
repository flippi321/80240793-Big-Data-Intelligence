import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Settings
decimals = 5
independent_value = 'Average year'
dependent_value = 'Night-chat ratio'

# --- Sampling strategies --- 
def sample_random(data, index):
    return np.array_split(data, index)

def sample_stratified(data, index):
    chosen_strata = 'Message number'
    fraction = 1 / index  # Fraction for each strata
    filtered_data = data.copy()
    
    # Create strata based on quantiles
    filtered_data['strata'] = pd.qcut(filtered_data[chosen_strata], 4, labels=False)
    
    # Sample each strata with fraction and reset index
    sample = filtered_data.groupby('strata').apply(lambda x: x.sample(frac=fraction, random_state=42)).reset_index(drop=True)
    return np.array_split(sample, index)

def sample_systematic(data, index):
    step = int(len(data) / index)
    sample = data.iloc[::step] 
    return np.array_split(sample, index) 

# --- Linear Regression algorithm --- 
def linreg(data, sampling, samples=10):
    regressors = []
    means = []
    std_devs = []
    
    # Split the data into equal parts
    data_chunks = sampling(data, samples)
    
    for index, chunk in enumerate(data_chunks):
        x = chunk[independent_value].values.reshape(-1, 1)
        y = chunk[dependent_value].values

        # Calculate mean and standard deviation for each chunk and add to list
        means.append(np.mean(y))
        std_devs.append(np.std(y))

        # Create a linear regression model
        regressor = LinearRegression()
        regressor.fit(x, y)

        # Add the regressor to the list
        regressors.append(regressor)
    
    # Create our average linear regression
    average_regressor = LinearRegression()
    average_regressor.coef_ = np.mean([regressor.coef_ for regressor in regressors], axis=0)
    average_regressor.intercept_ = np.mean([regressor.intercept_ for regressor in regressors], axis=0)
    
    # Calculate the MSE value for the regression
    x_full = data[independent_value].values.reshape(-1, 1)  # Full dataset's independent variable
    y_full = data[dependent_value].values                   # Full dataset's dependent variable
    y_pred_full = average_regressor.predict(x_full)
    mse = mean_squared_error(y_full, y_pred_full)
    
    # Create a DataFrame for means and standard deviations
    stats_df = pd.DataFrame({
        'Chunk': range(1, samples + 1),
        'Mean': np.round(means, decimals),
        'S. Deviation': np.round(std_devs, decimals)
    })
    
    return average_regressor, mse, stats_df

# --- Perform linear regression with different sampling strategies --- 

# Load the excel file
data = pd.read_excel('data.xlsx', sheet_name='data')

filtered_data = data[data['Session number'] >= 20]

sampling_strats = [sample_random, sample_stratified, sample_systematic]
sampling_strats_names = ['Random', 'Stratified', 'Systematic']
regressor_colors = ['black', 'gray', 'red']

# We do a scatterplot of the filtered data
plt.scatter(filtered_data[independent_value], filtered_data[dependent_value], color='blue')

# Test out the different sampling strategies
for index, strategy in enumerate(sampling_strats):

    # Estimate the regressor using our strategy and calculate mean/std
    average_regressor, regressor_error, stats_df = linreg(filtered_data, strategy, 10)
    
    # Plot the information about our regression
    print("\n------------------------------------------------------------")
    print(f"{sampling_strats_names[index]} linear regression")
    print(f"Formula:            y = {round(average_regressor.coef_[0], decimals)}x + {round(average_regressor.intercept_, decimals)}")
    print(f"Mean squared error: {round(regressor_error, decimals)}")
    print(f"Plot color:         {regressor_colors[index]}")
    
    # Display the table of mean and standard deviation
    print(f"\nStatistics for {sampling_strats_names[index]} sampling:")
    print(stats_df)

    # Plot the regression line
    x = filtered_data[independent_value].values.reshape(-1, 1)
    plt.plot(x, average_regressor.predict(x), color=regressor_colors[index])

plt.show()

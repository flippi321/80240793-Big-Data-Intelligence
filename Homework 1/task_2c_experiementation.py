import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# Load the excel file
dataframe = pd.read_excel('data.xlsx', sheet_name='data')

# Filter so we only have category 1 and 4, and categorize data
data = dataframe[dataframe['Group category'].isin([1, 4])]
grouped_data = data.groupby('Group category')

# Settings
calculate_all_options = True # WARNING, will run a long calulation instead of the original one

def binary(X, Y):
    # Splitting the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0
    )

    # Train the model
    regressor = LogisticRegression(max_iter=1000)
    regressor.fit(X_train, Y_train)

    # Print the accuracy of the model
    return regressor.score(X_test, Y_test)

# Filter so we only have category 1 and 4
filtered_data = data[data['Group category'].isin([1, 4])]

# Perform a standard logistic binar classification with all columns as variables
X = filtered_data.iloc[:, 2:13].values
Y = filtered_data['Group category'].values
print(f"\n------------------------------------------------------------")
print(f"Standard Logistic Regression Model")
print(f"Accuracy: {binary(X, Y)}")

# ALTERNATIVE SOLUTION
# We take every combination of every column to find the optimal multivariable binary operator
if(calculate_all_options):
    highest_accuracy = 0
    highest_accuracy_columns = []

    # Generate all combinations of 2 or more columns
    # The first column is name and the second is the category. We don't need neither
    for r in range(2, 13):
        for combo in itertools.combinations(range(2, 13), r):
            X = filtered_data.iloc[:, list(combo)].values  # Select the columns corresponding to the current combination
            new_accuracy = binary(X, Y)  # Call your binary classification function

            # Update if a new higher accuracy is found
            if new_accuracy > highest_accuracy:
                highest_accuracy = new_accuracy
                highest_accuracy_columns = combo

    print(f"\n------------------------------------------------------------")
    print(f"Best Logistic Regression Model")
    print(f"Accuracy: {highest_accuracy}")
    print(f"Columns: {highest_accuracy_columns}")
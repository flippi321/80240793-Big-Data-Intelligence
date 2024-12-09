import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Settings
show_boxplot = True     # Whether we should show a bar-plot for the average age for each group category
p_value_treshold = 0.05 # We reject the null hypothesis if the p-value is less than 5%

# Load the excel file
# By specifying sheet_name, we filter out the sheet 'category_data'
data = pd.read_excel('data.xlsx', sheet_name='data')

# Group the data by the 'Group category' column
grouped_data = data.groupby('Group category')['Average year']
grouped_means = grouped_data.mean()

# We visualize the data
if(show_boxplot):
    data.boxplot(column='Average year', by='Group category')
    plt.xlabel('Group category')
    plt.ylabel('Average year')
    plt.show()

# We do a one-way ANOVA test
result = f_oneway(*[group_df for group_name, group_df in grouped_data])
print("After doing a one-way ANOVA test, we get the following results:")
print("The F-value of the one-way ANOVA test is: ", result.statistic)
print("The p-value of the one-way ANOVA test is: ", result.pvalue)
print("\n")
if(result.pvalue < p_value_treshold):
    print("The p-value is less than the treshold, we reject the null hypothesis")
else:
    print("The p-value is greater than the treshold, we fail to reject the null hypothesis")
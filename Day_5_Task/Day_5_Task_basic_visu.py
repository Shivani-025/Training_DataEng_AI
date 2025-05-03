'''
Exercise 5: Create Basic Visualizations to Explore Data

Load a time series dataset.
Create a line plot to visualize the trend over time.
Display the plot.
'''


import matplotlib.pyplot as plt
import pandas as pd

# Load time series dataset
time_series_data = pd.read_csv('daily-min-temperatures.csv', parse_dates=['Date'], index_col='Date')

# Create a line plot to visualize the trend over time
time_series_data.plot()

plt.title('Time Series Trend')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

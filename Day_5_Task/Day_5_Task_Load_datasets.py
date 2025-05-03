'''
Exercise 1: Load Datasets from Various File Formats
Load a CSV file named data.csv into a Pandas DataFrame.
Load a JSON file named data.json into a Pandas DataFrame.
Load an Excel file named data.xlsx into a Pandas DataFrame.
Ask how many rows to display from the DataFrame and display the results

'''

import pandas as pd

#Load a CSV file named data.csv into a Pandas DataFrame.
csv_data = pd.read_csv('sample_data.csv')
print("Data from CSV:\n", csv_data)

#Load a JSON file named data.json into a Pandas DataFrame.
json_data = pd.read_json('sample_data.json')
print("Data from JSON:\n", json_data)

#Load an Excel file named data.xlsx into a Pandas DataFrame.
excel_data = pd.read_excel('sample_data.xlsx')
print("Data from Excel:\n", excel_data)


# Ask for the number of rows to display
num_rows = int(input("Enter the number of rows to display: "))

# Display the first 'num_rows' rows of each DataFrame
print("\nCSV file:")
print(csv_data.head(num_rows))

print("\nJSON file:")
print(json_data.head(num_rows))

print("\nExcel file:")
print(excel_data.head(num_rows))
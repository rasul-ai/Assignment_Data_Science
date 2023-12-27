# Business_Automation_Data_Science_Assignment - Data Exploration and Preprocessing

This Jupyter notebook explores and preprocesses a dataset related to student performance. The dataset is loaded from a CSV file, and various data exploration and cleaning operations are performed.

**Author:** [Your Name]
**Date:** [Date]

## Section 1: Data Loading
In this section, we mount Google Drive (if applicable) and load the dataset from a CSV file using pandas.

```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Load the data
file_path = '/content/drive/MyDrive/Business_Automation_Task/Students_Performance_data_set.csv'
df = pd.read_csv(file_path)

Section 2: Data Exploration
This section provides a basic exploration of the dataset, including displaying the first few rows, providing basic information, and generating summary statistics.

python
Copy code
# Displaying the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display basic information about the dataset
print("\nBasic information about the dataset:")
print(df.info())

# Displaying summary statistics
print("\nSummary statistics of the dataset:")
print(df.describe())

# Counting missing values in each column
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check for duplicate values
print("Duplicate rows in the dataset:")
print(df.duplicated().sum())
Section 3: Data Cleaning and Imputation
This section handles missing values in the dataset. It imputes missing values in categorical columns with the mode and performs custom parsing and averaging for specific numerical columns. Additionally, it drops rows with missing values in specific columns.

python
Copy code
# Impute missing values with the mode of each categorical column
categorical_columns = ['Do you use University transportation?', 'What is your preferable learning mode?']
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Custom parsing and averaging for selected numerical columns
column = ['How many hour do you study daily?', 'How many hour do you spent daily in social media?', \
          'Average attendance on class', 'How many hour do you spent daily on your skill development?']

def parse_and_average(value):
    # Custom function to parse and average numeric values in a string
    # ...

# Continue with the remaining code for data cleaning and imputation

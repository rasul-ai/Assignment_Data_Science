# Business_Automation_Data_Science_Assignment - Data Exploration and Preprocessing

This repository represents an assignment for a part of job interview from Business Automation. BA has given me a dataset and asked to explore the data and do necessary preprocessing so that using this dataset anyone can classify students performance.In this file I am writing a brief description of my experiments and outcomes.

**Author:** [Md Rasul Islam Bapary]
**Date:** [27.12.2023]

## This is my notebook structure.
```
Notebook Structure:
1. Data Loading
2. Data Exploration
3. Data Cleaning and Imputation
4. Feature Engineering
5. Data Visualization
6. Data Export
```

## Section 1: Data Loading
In this section, I hve mounted Google Drive (if applicable) and load the dataset from a CSV file using pandas.

```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Load the data
file_path = '/content/drive/MyDrive/Business_Automation_Task/Students_Performance_data_set.csv'
df = pd.read_csv(file_path)
```

## Section 2: Data Exploration
This section provides a basic exploration of the dataset, including displaying the first few rows, providing basic information, shape of the dataset, generating summary statistics, counting missing values in each colum, checking the duplicate rows(if there is any). This information will help to understand the basic characteristics of a dataset.

```python
# Displaying the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display basic information about the dataset
print("\nBasic information about the dataset:")
print(df.info())

# Displaying summary statistics
print("\nSummary statistics of the dataset:")
print(df.describe())

# Dataset shape
print("\nShape of the dataset:")
print(df.shape)

# Counting missing values in each column
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check for duplicate values
print("Duplicate rows in the dataset:")
print(df.duplicated().sum())
```

## Section 3: Data Cleaning and Imputation
This section handles missing values in the dataset. It imputes missing values in categorical columns with the mode and performs custom parsing and averaging for specific numerical columns. Additionally, it drops rows with missing values in specific columns.

```python
# Impute missing values with the mode of each categorical column
categorical_columns = ['Do you use University transportation?', 'What is your preferable learning mode?']
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Custom parsing and averaging for selected numerical columns
column = ['How many hour do you study daily?', 'How many hour do you spent daily in social media?', \
          'Average attendance on class', 'How many hour do you spent daily on your skill development?']

def parse_and_average(value):
    # Custom function to parse and average numeric values in a string
    # ...

for column_name in column:
    df[column_name] = df[column_name].apply(parse_and_average)
    # Replace remaining non-numeric or null values or 0 with the mean
    mean_value = df[column_name].mean()
    df[column_name].fillna(mean_value, inplace=True)
    df[column_name].replace(0, mean_value, inplace=True)

# Drop rows with missing values in specific columns
df = df.dropna(subset=['Are you engaged with any co-curriculum activities?', 'What was your previous SGPA?', 'Age',\
                       'What is your current CGPA?', 'How many Credit did you have completed?',\
                       'What is your monthly family income?', 'Are you engaged with any co-curriculum activities?',\
                       'How many hour do you spent daily on your skill development?', 'What is you interested area?',\
                       'What are the skills do you have ?', 'How many hour do you spent daily in social media?'])
```

## Section 4: Feature Engineering
In this section, new columns are created based on specific conditions. The 'target' column is created to categorize students into different performance levels (Excellent, Good, Average, Poor).
```python
import numpy as np

excellent_condition = (df['How many hour do you study daily?'] >= 3.0) & (df['What is your current CGPA?'] >= 3.75) & \
                      (df['What was your previous SGPA?'] >= 3.20) & (df['Average attendance on class'] >= 90.0) & \
                      (df['How many hour do you spent daily in social media?'] <= 3.0)

good_condition = (df['How many hour do you study daily?'] >= 2.0) & (df['What is your current CGPA?'] >= 3.20) & \
                 (df['What was your previous SGPA?'] >= 2.90) & (df['Average attendance on class'] >= 85.0) & \
                 (df['How many hour do you spent daily in social media?'] <= 4.0)

average_condition = (df['How many hour do you study daily?'] >= 1.0) & (df['What is your current CGPA?'] >= 2.80) & \
                    (df['What was your previous SGPA?'] >= 2.50) & (df['Average attendance on class'] >= 75.0) & \
                    (df['How many hour do you spent daily in social media?'] <= 5.0)

# Create the 'target' column based on conditions
df['target'] = np.select([excellent_condition, good_condition, average_condition], ['Excellent', 'Good', 'Average'], default='Poor')
```

## Section 5: Data Visualization
This section includes various data visualizations using seaborn and matplotlib to gain insights into  the distribution of different features and their relationship with the target variable.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Distribution plots for selected numerical columns
column = ['How many hour do you study daily?', 'How many hour do you spent daily in social media?', 'Average attendance on class',\
          'What was your previous SGPA?', 'What is your current CGPA?']
for column_name in column:
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Create a distribution plot
    sns.distplot(df[column_name].dropna(), kde=False, bins=30, color='blue')

    plt.xlabel(f'{column_name}')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column_name}')
    plt.show()
```

## Section 6: Data Export
The preprocessed data is saved to a new CSV file for further analysis and modeling.
```python
# Save the DataFrame to a new CSV file
df.to_csv('/content/drive/MyDrive/Business_Automation_Task/new_preprocessed_data.csv', index=False)
```

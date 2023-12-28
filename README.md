# Business_Automation_Data_Science_Assignment - Data Exploration, Preprocessing and Model Training

This repository represents an assignment for a part of job interview from Business Automation. BA has given me a dataset and asked to explore the data and do necessary preprocessing so that using this dataset anyone can classify students performance.In this file I am writing a brief description of my experiments and outcomes.

**Author:** [Md Rasul Islam Bapary]  
**Date:** [27.12.2023]

# The Preprocessed dataset after EDA is looking like this.
![Sample_dataset](https://github.com/rasul-ai/Assignment_Data_Science/blob/main/Images/Screenshot%20from%202023-12-27%2023-47-07.png)

## This is my notebook structure.
```
Notebook Structure:
1. Data Loading
2. Data Exploration
3. Data Cleaning and Imputation
4. Feature Engineering
5. Data Visualization
6. Outlier Detection
7. Data Export
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
This section handles missing values in the dataset. It imputes missing values in categorical columns with the mode and performs custom parsing and averaging for specific numerical columns because some columns should have only numerical values. To construct the target column these numerical values are important as these are being conditioned. After imputing numerical values, if there is any Nan or Null values then the code drops those rows with missing values in specific columns.

```python
# Impute missing values with the mode of each categorical column
categorical_columns = ['Do you use University transportation?', 'What is your preferable learning mode?']
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Custom parsing and averaging for selected numerical columns
column = ['How many hour do you study daily?', 'How many hour do you spent daily in social media?', \
          'Average attendance on class', 'How many hour do you spent daily on your skill development?']

def parse_and_average(value):
    # Custom function to extract and average numeric values from a string
    # Please see details on data_exploration.ipynb file

for column_name in column:
    df[column_name] = df[column_name].apply(parse_and_average)
    # Please see details on data_exploration.ipynb file


# Drop rows with missing values in specific columns
df = df.dropna(subset=['Are you engaged with any co-curriculum activities?', 'What was your previous SGPA?', 'Age',\
                       'What is your current CGPA?', 'How many Credit did you have completed?',\
                       'What is your monthly family income?', 'Are you engaged with any co-curriculum activities?',\
                       'How many hour do you spent daily on your skill development?', 'What is you interested area?',\
                       'What are the skills do you have ?', 'How many hour do you spent daily in social media?'])

# Columns to delete
columns_to_delete = ['University Admission year','H.S.C passing year',
       'Current Semester',
       'What are the skills do you have ?',
       'What is you interested area?',
       'How many Credit did you have completed?',
       'What is your monthly family income?']

# Use the drop method to remove specified columns
df.drop(columns=columns_to_delete, inplace=True)

```
### Please see the details on data_exploration.ipynb file


## Section 4: Feature Engineering
In this section, new columns are created based on specific conditions. The 'target' column is created to categorize students into different performance levels (Excellent, Good, Average, Poor). This feature represents the label for training a classification network such as Neural network, decision trees, random forest or logistic regression.

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
This section includes various data visualizations using seaborn and matplotlib to gain insights into  the distribution of different features and their relationship with the target variable. Firstly I check the distribution of all the columns in the dataset. As the target column is constructed, I checked for the correlation of target column with all the columns. Based on the correlation I select 10 columns as a part of the preprocessed dataset for classification model training.
![Correlation_matrix](https://github.com/rasul-ai/Assignment_Data_Science/blob/main/Images/corr_matrix.jpg)


```python
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix[['target']], annot=True, cmap='coolwarm', linewidths=.5)
plt.savefig("/content/drive/MyDrive/Business_Automation_Task/images/corr_matrix.jpg")
plt.show()
```
Then I have deleted rest of the columns. This time the dataset shape was (968,11).  I have checked several others visualization techniques to ensure these 10 columns actually affects the target columns. 
```python
sns.set(style="whitegrid")
# Visualize each column separately
for column in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column].dropna(), kde=False, color='skyblue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.savefig(f'/content/drive/MyDrive/Business_Automation_Task/images/Distribution_{column}.jpg')
    plt.show()
```
## Section 6: Outlier Detection
During doing the experiment of visualization I found that there are some columns which contains outlier values. For example CGPA or SGPA must be in between [2.0,4.0], but some values found above 5.0. That is why I checked all the columns if there is outlier present in them. Based on specific condition I removed outlier values. This reduces the dataset shape into (486,11).

```python
columns_of_interest = ['What was your previous SGPA?', 'What is your current CGPA?']

# Removing Outliers 
for column in columns_of_interest:
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of to detect outliers in {column}')
    plt.show()

    lower_bound = 2.0
    upper_bound = 4.0
    print(f'Lower Bound: {lower_bound}, Upper Bound: {upper_bound}')

    # Identify and remove outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    df = df[~df.index.isin(outliers.index)]

    # Display the updated boxplot
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column} (Outliers Removed)')
    plt.show()

    # Create a scatter plot without outliers
    sns.scatterplot(x=df[column], y=df['target'], hue=df['target'])
    plt.title('Scatter Plot without Outliers')
    plt.show()

print(df.shape)
```
![CGPA_Outliers](https://github.com/rasul-ai/Assignment_Data_Science/blob/main/Images/Boxplot%20of%20What%20is%20your%20current%20CGPA_%20(With%20Outliers).jpg)
![CGPA_without_outliers](https://github.com/rasul-ai/Assignment_Data_Science/blob/main/Images/Boxplot%20of%20What%20is%20your%20current%20CGPA_%20(Outliers%20Removed).jpg)
![SM_Outliers](https://github.com/rasul-ai/Assignment_Data_Science/blob/main/Images/Boxplot%20of%20How%20many%20hour%20do%20you%20spent%20daily%20in%20social%20media_%20(With%20Outliers).jpg)
![SM_without_outliers](https://github.com/rasul-ai/Assignment_Data_Science/blob/main/Images/Boxplot%20of%20How%20many%20hour%20do%20you%20spent%20daily%20in%20social%20media_%20(Outliers%20Removed).jpg)
### Please see the data_exploration.ipynb file for details.

## Section 7: Data Export
Finally the preprocessed data is saved to a new CSV file for further analysis and modeling.
```python
# Save the DataFrame to a new CSV file
df.to_csv('/content/drive/MyDrive/Business_Automation_Task/new_preprocessed_data.csv', index=False)
```


odel Training Documentation
In this section, I will provide a detailed documentation of the model training process for the student performance classification task. The goal is to build a classification model that predicts students' performance levels based on the preprocessed dataset. We will utilize a three-layer neural network as the primary model, and also compare its performance with other traditional classifiers such as Decision Tree (DT), Random Forest (RF), and Logistic Regression (LR).
Section 1: Data Loading and Preprocessing
The model training process begins with loading the preprocessed dataset and preparing it for training.
pythonCopy code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load preprocessed dataset
file_path = '/content/drive/MyDrive/Business_Automation_Task/new_preprocessed_data.csv'
df = pd.read_csv(file_path)

# Separate features and target variable
X = df.drop(columns=['target'])
y = df['target']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
Section 2: Model Training - Neural Network
Neural Network Architecture
The neural network used for this classification task consists of three layers: an input layer with the number of neurons equal to the number of features, a hidden layer with 64 neurons and a ReLU activation function, and an output layer with four neurons (one for each performance level) and a softmax activation function.
pythonCopy code
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)
Section 3: Model Evaluation
After training the neural network, we evaluate its performance on the test set. Additionally, we compare its results with other traditional classifiers: Decision Tree, Random Forest, and Logistic Regression.
Neural Network Evaluation
pythonCopy code
# Evaluate the neural network on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Neural Network - Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
Traditional Classifiers Comparison
pythonCopy code
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Evaluate and compare the models
def evaluate_model(model_name, predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print(f'{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Evaluate neural network
evaluate_model('Neural Network', model.predict_classes(X_test))

# Evaluate traditional classifiers
evaluate_model('Decision Tree', dt_predictions)
evaluate_model('Random Forest', rf_predictions)
evaluate_model('Logistic Regression', lr_predictions)
Section 4: Model Visualization
Visualize the training and validation accuracy and loss over epochs for the neural network.
pythonCopy code
import matplotlib.pyplot as plt

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

# Plot training history of the neural network
plot_training_history(history)
Section 5: Conclusion
In conclusion, the model training process involved loading and preprocessing the dataset, building and training a three-layer neural network, and evaluating its performance on the test set. Additionally, we compared the results with traditional classifiers and visualized the training history of the neural network.
The neural network showed promising results in terms of accuracy, precision, recall, and F1 score. The comparison with other classifiers provides insights into the effectiveness of different models for the given task. Further optimization and fine-tuning of hyperparameters could potentially enhance the performance of the models.



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


# Model Training Documentation
In this section, I am giving a detailed documentation of the model training process for the student performance classification task. The goal is to build a classification model that predicts students' performance levels based on the preprocessed dataset. I have created a three-layer neural network as the primary model, and also compare its performance with a two-layer neural network and other traditional classifiers such as Decision Tree (DT), Random Forest (RF), and Logistic Regression (LR). Here is the notebook structure,

## Section 1: Data Loading and Preprocessing
The model training process begins with loading the preprocessed dataset and preparing it for training.
```python
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

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=15)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=15)

# Create DataLoader for training, validation, and test sets
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
```

## Section 2: Neural Network Architecture
The neural network that I used for this classification task consists of three layers: an input layer with the number of neurons equal to the number of features, a hidden layer with 64 neurons and a ReLU activation function, and an output layer with four neurons (one for each performance level) and a softmax activation function.
```python
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
```
## Section 3: Model Training
The hyperparameters that I used for model training consist of,
```python
Learning rate of 1e-3
Epoch = 500
Batch_size = 32
```

Here the training code,
```python
for epoch in range(epochs):
    model.train()
    t_loss = 0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        t_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_accuracy = correct_train / total_train

    train_losses.append(t_loss / len(train_loader))

    # Validation loop
    model.eval()
    v_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            val_outputs = model(inputs)
            val_loss = criterion(val_outputs, labels)
            v_loss += val_loss.item()

            _, predicted_val = torch.max(val_outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()

    val_accuracy = correct_val / total_val

    val_losses.append(v_loss / len(val_loader))

    print(f'Epoch {epoch + 1}/{epochs}, T_Accuracy: {train_accuracy * 100:.2f}%, V_Accuracy: {val_accuracy * 100:.2f}%,\
     T_loss: {t_loss / len(train_loader):.4f}, V_loss: {v_loss / len(val_loader):.4f}')
```

## Section 4: Model Evaluation
After training the neural network, I evaluated its performance on the test set. During evaluation I found there are some overfitting in the model training. I have tried to reduce them using Batch Normalization, Regularization technic like Dropout, Weight_Decay and Learning rate schedulers.


## Comparison with Builtin Logistic Regression, Decision Tree, Random Forest
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

# Logistic Regression
logreg_model = LogisticRegression(random_state=42, max_iter=500)
logreg_model.fit(X_train, y_train)
logreg_preds = logreg_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
```

## Section 5: Model Visualization
Here are some visualization image of the training and validation accuracy and loss over epochs for the neural network for some of the experiment.

## Section 6: Conclusion
In conclusion, the model training process involved loading and preprocessing the dataset, building and training a three-layer neural network, and evaluating its performance on the test set. Additionally, I compared the results with traditional classifiers and visualized the training history of the neural network.
The neural network showed promising results in terms of accuracy, precision, recall, and F1 score. The comparison with other classifiers provides insights into the effectiveness of different models for the given task and I found that Decission tree and Random Forest performs greatly over my custom neural network. Further optimization and fine-tuning of hyperparameters could potentially enhance the performance of the models.



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


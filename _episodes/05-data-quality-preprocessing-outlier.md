---
title: Outlier Detection and Handling Outliers
teaching: 130
exercises: 130
questions:
- "How much of the data is missing? Is it a small fraction or a significant portion?"


objectives:
- "Learn the difference between deleting incomplete observations and imputing missing values."


keypoints:
- "Deletion: Simple but risks losing large amounts of data and introducing bias."

---

## Graphical Outlier Detection
- Box Plot

## Statistical Outlier Detection:
- Interquartile Range (IQR) Method
- Z-Score Method

## Machine learning Models
- Isolation Forest
- Local Outlier Factor (LOF)

## Handling Outliers
- Removal
- Capping
- Transformation

### Importing Libraries
```python
# Import pandas and numpy
import pandas as pd
import numpy as np

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Zscore
from scipy import stats

# Outlier Detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Import the datetime class from the datetime module
from pandas.tseries.offsets import MonthEnd

# Interactive Widgets
import ipywidgets as widgets

# Display libraries
from IPython.display import display

from PIL import Image
```

### Import the data

[Data](../data/gdpreal.csv)

```python
gdpreal = pd.read_csv('gdpreal.csv')
gdpreal.head()
```

### Convert the 'Quarter' column to datetime format and adjust for quarter-end
```python
gdpreal['Quarter'] = pd.PeriodIndex(gdpreal['Quarter'], freq='Q').to_timestamp()
gdpreal['Quarter'] = gdpreal['Quarter'] + MonthEnd(3)

# Convert 'RealGDP' to float (ensure clean data)
gdpreal['RealGDP'] = gdpreal['RealGDP'].str.replace(',', '').astype(float)

gdpreal.head()
```


### Plot the Real GDP data
```python
plt.figure(figsize=(12, 6))
plt.plot(
    gdpreal['Quarter'],
    gdpreal['RealGDP'],
    marker='o',
    linestyle='-',
    color='blue',
    label='Real GDP (Billion Naira)'
)

# Add titles and labels
plt.title('Nigeria Real GDP (2010 Q1 - 2024 Q3)', fontsize=16)
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Real GDP (Billion Naira)', fontsize=12)

# Improve y-axis ticks
plt.yticks(fontsize=10)  # Reduce frequency and size of y-ticks
plt.ticklabel_format(axis='y', style='plain')  # Prevent scientific notation on the y-axis

# Grid, legend, and layout
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.legend(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()
```

### Graphical Outlier Detection: Box Plot

```python
# Drop the Date column
gdpreal_d = gdpreal.drop(columns=['Quarter'])
gdpreal_d.head()

def plot_boxplot(column):
    plt.figure(figsize=(10, 6))
    gdpreal_d.boxplot(column=column, color='b', patch_artist=True, figsize=(8, 6),
                      grid= 1, fontsize=12)
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.show()

# Create a dropdown widget for selecting the column
column_selector = widgets.Dropdown(
    options=gdpreal_d.columns,
    description='Column:',
    disabled=False,
)

# Link the dropdown widget to the plot_boxplot function
interactive_plot = widgets.interactive_output(plot_boxplot, {'column': column_selector})

# Display the widget and the interactive plot
display(column_selector, interactive_plot)
```
### Statistical Outlier Detection: Interquartile Range (IQR) Method

```python
def detect_outliers_IQR(data, column_names):
    # Initialize a dictionary to store the count of outliers for each column
    outlier_counts = {}
    
    # Iterate over each column name provided in the column_names list
    for column_name in column_names:
        # Calculate the first quartile (Q1) for the column
        Q1 = data[column_name].quantile(0.25)
        # Calculate the third quartile (Q3) for the column
        Q3 = data[column_name].quantile(0.75)
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1

        # Calculate the lower bound for outliers
        lower_bound = Q1 - 1.5 * IQR
        # Calculate the upper bound for outliers
        upper_bound = Q3 + 1.5 * IQR

        # Create a new column 'IQR_Outlier' to mark outliers (1 if outlier, 0 otherwise)
        data['IQR_Outlier'] = ((data[column_name] < lower_bound) | (data[column_name] > upper_bound)).astype(int)

        # Filter the data to get only the rows that are outliers
        iqr_outliers = data[data['IQR_Outlier'] == 1]
        
        # Count the number of outliers in the current column
        outlier_count = iqr_outliers.shape[0]
        
        # Store the count of outliers in the dictionary
        outlier_counts[column_name] = outlier_count
        
        # Print the outliers detected in the current column
        print(f"Outliers detected by Interquartile Range (IQR) in {column_name}:")
        print(iqr_outliers)
        print("-----------------------------")  

    # Report the total counts of outliers detected for each column
    print("Total outlier counts detected by Interquartile Range (IQR) Method:")
    for column_name, count in outlier_counts.items():
        print(f"{column_name}: {count}")
```

### Detect outliers in data using IQR Method
```python
df_IQR = gdpreal_d.copy()

data_col_bilate = df_IQR.columns

# Convert the data column to a list
data_col_list = data_col_bilate.tolist()

detect_outliers_IQR(df_IQR, data_col_list)
```
### Statistical Outlier Detection: Z-Score Method
```python
def detect_outliers_zscore(data, column_names):
    # Initialize a dictionary to store the count of outliers for each column
    outlier_counts = {}
    
    # Z-Score Method
    for column_name in column_names:
        # Calculate the z-scores for the column
        z_scores = np.abs(stats.zscore(data[column_name]))
        # Create a new column 'ZScore_Outlier' to mark outliers (1 if outlier, 0 otherwise)
        data['ZScore_Outlier'] = (z_scores > 3).astype(int)
        # Filter the data to get only the rows that are outliers
        z_outliers = data[data['ZScore_Outlier'] == 1]
        
        # Count the number of outliers in the current column
        outlier_count = z_outliers.shape[0]
        # Store the count of outliers in the dictionary
        outlier_counts[column_name] = outlier_count
        
        # Print the outliers detected in the current column
        print(f"Outliers detected by Z-Score in {column_name}:")
        print(z_outliers)
        print("-----------------------------")
    
    # Report the total counts of outliers detected for each column
    print("Total outlier counts detected by Z-score Method:")
    for column_name, count in outlier_counts.items():
        print(f"{column_name}: {count}")
```

### Detect outliers in data using Z-Score Method

```python
df_zscore = gdpreal_d.copy()
detect_outliers_zscore(df_zscore, data_col_list)
```
### Isolation Forest

- The Isolation Forest method is a **tree-based algorithm** that identifies outliers in data by randomly partitioning features to create shorter paths for anomalies.

- is an **unsupervised** learning algorithm used primarily for anomaly and outliers detection. 
 
- It operates on the principle that anomalies are "few and different" from the majority of the data.
 
#### How Isolation Forest Works:
 - **Random Feature Selection**: A feature is selected at random.
 - **Random Split Value**: A random split value is chosen between the selected feature's minimum and maximum values.
 - **Recursive Partitioning**: The data is partitioned recursively until all data points are isolated
- **Scoring Data Points**
  - Short Path Length: Indicates potential anomalies.
  - Long Path Length: Indicates normal data points.

- Fast, generalize well, and robust to noise

# Initialize the Isolation Forest model

```python
iso_forest = IsolationForest(contamination=0.01, random_state=42)

def detect_outliers_IF(data, column_names):
    # Initialize a dictionary to store the count of outliers for each column
    outlier_counts = {}
    
    # Iterate over each column name provided in the column_names list
    for column_name in column_names:

        # Fit the Isolation Forest model and predict outliers for the current column
        data['IFOutlier'] = iso_forest.fit_predict(data[[column_name]])

        # Filter the data to get only the rows that are outliers (predicted as -1)
        outliers = data[data['IFOutlier'] == -1]
        
        # Count the number of outliers in the current column
        outlier_count = outliers.shape[0]
        
        # Store the count of outliers in the dictionary
        outlier_counts[column_name] = outlier_count
        
        # Print the outliers detected in the current column
        print(f"Outliers detected in {column_name}:")
        print(outliers)
        print("-----------------------------")
    
    # Report the total counts of outliers detected for each column
    print("Total outlier counts detected by Isolation Forest:")
    for column_name, count in outlier_counts.items():
        print(f"{column_name}: {count}")
```

### Detect outliers in data using Isolation Forest Method

```python
df_IF= gdpreal_d.copy()

detect_outliers_IF(df_IF, data_col_list)
```
### Local Outlier Factor
- is an **unsupervised** machine learning algorithm that identifies outliers by comparing the **density** of a data point to the density of its neighbors.
  1. Estimating local density using the distance to its k nearest neighbors.
  2. Comparing local densities of its neighbors.
  3. Identifying outlier
     
- LOF can be used to identify outliers in a variety of situations, including:
   - Cyber-attacks: LOF can be used to identify cyber-attacks in computer networks.
  - Fraudulent transactions: LOF can be used to identify fraudulent transactions

[lof1]('../assets/img/lof1.png')

[lof2]('../assets/img/lof1.png')

### Initialize the Local Outlier Factor (LOF) model

```python
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)

def detect_outliers_LOF(data, column_names):

    # Initialize a dictionary to store the count of outliers for each column
    outlier_counts = {}
    
    # Iterate over each column name provided in the column_names list
    for column_name in column_names:

        # Fit the LOF model and predict outliers for the current column
        lof_labels = lof.fit_predict(data[[column_name]])

        # Create a new column 'LOF_Outlier' to mark outliers (-1 if outlier, 1 otherwise)
        data['LOF_Outlier'] = lof_labels

        # Filter the data to get only the rows that are outliers (predicted as -1)
        lof_outliers = data[data['LOF_Outlier'] == -1]
        
        # Count the number of outliers in the current column
        outlier_count = lof_outliers.shape[0]
        
        # Store the count of outliers in the dictionary
        outlier_counts[column_name] = outlier_count
        
        # Print the outliers detected in the current column
        print(f"Outliers detected by Local Outlier Factor in {column_name}:")
        print(lof_outliers)
        print("-----------------------------")
    
    # Report the total counts of outliers detected for each column
    print("Total outlier counts detected by Local Outlier Factor:")
    for column_name, count in outlier_counts.items():
        print(f"{column_name}: {count}")
```

### Detect outliers in data using Local Outlier Factor Method

```python
df_LOF= gdpreal_d.copy()
detect_outliers_LOF(df_LOF, data_col_list)
```
###  Handling Outliers
- Removal: Remove rows that contain outlier values beyond a certain threshold.  
- Capping: Cap the outlier values at a specified upper or lower percentile to reduce their impact.
- - Transformation: Apply transformations (e.g., log, square root) to reduce the effect of extreme values.

### Capping

```python
df_out = gdpreal_d.copy()
data_col_list = df_out.columns
# convert the column names to a list
# data_col_list = ['CPYC' ]  

for column in data_col_list:
    percentile_5 = df_out[column].quantile(0.05)
    percentile_95 = df_out[column].quantile(0.95)
    
    # Capping the values
    df_out[column] = df_out[column].clip(lower=percentile_5, upper=percentile_95)
```
    
### Summary statistics of the data after capping

```python
df_out.describe().T
```


### Box Plot after capping

```python
def plot_boxplot(column):
    plt.figure(figsize=(10, 6))
    df_out.boxplot(column=column, color='b', patch_artist=True, figsize=(8, 6),
                      grid= 1, fontsize=12)
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.show()

# Create a dropdown widget for selecting the column
column_selector = widgets.Dropdown(
    options=df_out.columns,
    description='Column:',
    disabled=False,
)

# Link the dropdown widget to the plot_boxplot function
interactive_plot = widgets.interactive_output(plot_boxplot, {'column': column_selector})

# Display the widget and the interactive plot
display(column_selector, interactive_plot)
```





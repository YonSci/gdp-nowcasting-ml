---
title: Exploratory Data Analysis (EDA)
teaching: 130
exercises: 130
questions:
- "What are the basics of Numpy?"
objectives:
- "Understand the purpose and importance of EDA in the data analysis pipeline."
- "Learn how to use Python libraries for EDA."
- "To gain insights into the data by summarizing its main characteristics."
keypoints:
- "To understand the structure and contents of the dataset."

---

# Exploratory Data Analysis (EDA)

**Exploratory Data Analysis (EDA)** is an **analytical** approach aimed at uncovering the inherent characteristics of datasets, utilizing **statistical** (non-graphical) and **visualization** (graphical) techniques.

## Agenda

- to gain insights into the data by summarizing its main characteristics:

  - **Find patterns**,
  - **Identify outliers**,
  - **Explore the relationship between variables**,
  - **Helps to indentify features** (aka feature selection)

## EDA techniques:

  - Basic Statistical Summary
  - Histogram and Density Plot
  - Box Plots
  - Violin Plot
  - Time Series Plot and Bar Chart
  - Correlation Analysis
  - Bivariate Relationships (Bivariate Scatter and Pair Plot )
  - Automatic EDA Tools

## Importing Libraries

```python
# Import pandas and numpy
import pandas as pd
import numpy as np

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Interactive Widgets
import ipywidgets as widgets

# Correlation Matrix
import klib

# Automatic Visualization Tools
# Summary Statistics
import summarytools as st
```

## Load Example Data

```python
# Load the data from CSV files
gdp_df = pd.read_csv('gdp_gt_data.csv')
```

## Data Understanding/Explore the DataFrame

```python
# Display the shape of the data frame
gdp_df.shape
```

```python
# Display the first few rows of the data frame
gdp_df.head()
```

```python
# Display the last few rows of the data frame
gdp_df.tail()
```

```python
# Display the data types of the columns
gdp_df.dtypes
```

```python
# Display the information of the data frame
gdp_df.info()
```

```python
# Display the column name of the data frame
gdp_df.columns
```
## Basic Statistical Summary

```python
# Drop the Date column
gdp_gt_d = gdp_df.drop(columns=['Date'])

# Calculate the summary statistics
summary_stats = gdp_gt_d.describe().T

# Skewness 
summary_stats['Skewness'] = gdp_gt_d.skew()

# kurtosis
summary_stats['Kurtosis'] = gdp_gt_d.kurtosis()

# Add coefficient of variation (CV)
summary_stats['CV'] = (gdp_gt_d.std() / gdp_gt_d.mean()) * 100

summary_stats.T
```

## Histogram and Density Plot

```python
# Extract the GDP column from the DataFrame
gdp = gdp_gt_d['GDP']

# Create a new figure for the plot with a specified size
plt.figure(figsize=(10, 6))

# Create a histogram plot with a kernel density estimate (KDE) overlay
# - gdp_gt_d['GDP']: The data to plot
# - kde=True: Add a KDE plot
# - bins=10: Number of bins for the histogram
# - alpha=0.7: Transparency level of the histogram bars

sns.histplot(gdp_gt_d['GDP'], kde=True, bins=10, alpha=0.7)

# Set the title of the plot
plt.title('Histogram of GDP')

# Set the label for the x-axis
plt.xlabel('GDP')

# Set the label for the y-axis
plt.ylabel('Frequency')

# Add a grid to the plot for better readability
plt.grid(True)

# Adjust the layout to make sure everything fits without overlapping
plt.tight_layout()

# Display the plot
plt.show()
```

## Histogram and Density plot with dropdown widget
```python
def plot_distribution(column):
    plt.figure(figsize=(10, 6))
    sns.histplot(gdp_gt_d[column], kde=True, color='r', bins=15)
    plt.title(f'{column} Distribution', size=18)
    plt.xlabel(column, size=14)
    plt.ylabel('Density/Frequency', size=14)
    plt.show()
    

# Create a dropdown widget for selecting the column
column_selector = widgets.Dropdown(
    options=gdp_gt_d.columns,
    description='Column:',
    disabled=False,
)

# Link the dropdown widget to the plot_distribution function
interactive_plot = widgets.interactive_output(plot_distribution, {'column': column_selector})

# Display the widget and the interactive plot
display(column_selector, interactive_plot)

```

## Box Plots

```python
# Extract the GDP column from the DataFrame
gdp = gdp_gt_d['GDP']

# Create a new figure for the plot with a specified size
plt.figure(figsize=(10, 5))

# Create a boxplot for the GDP data
# - x=gdp: The data to plot
# - color='r': The color of the boxplot
sns.boxplot(x=gdp, color='r')

# Set the title of the plot with a specified font size
plt.title('GDP Boxplot', size=18)

# Set the label for the x-axis with a specified font size
plt.xlabel("GDP", size=14)

# Add a grid to the plot for better readability
plt.grid(True)

# Adjust the layout to make sure everything fits without overlapping
plt.tight_layout()

# Display the plot
plt.show()
```


## Box Plot with dropdown widget

```python
# Create a function to plot the boxplot
def boxplot_plot(column):
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=gdp_gt_d[column], color='r')
    plt.title(f'{column} Boxplot', size=18)
    plt.xlabel(column, size=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Create a dropdown widget for selecting the column
column_selector = widgets.Dropdown(
    options=gdp_gt_d.columns,
    description='Column:',
    disabled=False
)

# Link the dropdown widget to the boxplot_plot function
interactive_box_plot = widgets.interactive_output(boxplot_plot, {'column': column_selector})

# Display the widget and the interactive plot
display(column_selector, interactive_box_plot)
```

## Violin Plot

```python
# Extract the GDP column from the DataFrame
gdp = gdp_gt_d['GDP']

# Create a new figure for the plot with a specified size
plt.figure(figsize=(10, 6))

# Create a violin plot for the GDP data
# - y=gdp: The data to plot on the y-axis
sns.violinplot(y=gdp)

# Set the title of the plot
plt.title('Violin Plot of GDP')

# Set the label for the y-axis
plt.ylabel("GDP")

# Display the plot
plt.show()
```

## Violin Plot with dropdown widget

```python
def plot_violin(column):
    plt.figure(figsize=(10, 6))
    sns.violinplot(y=gdp_gt_d[column])
    plt.title(f'Violin Plot of {column}')
    plt.ylabel(column)
    plt.show()

# Create a dropdown widget for selecting the column
column_selector = widgets.Dropdown(
    options=gdp_gt_d.columns,
    description='Column:',
    disabled=False,
)

# Link the dropdown widget to the plot_violin function
interactive_plot = widgets.interactive_output(plot_violin, {'column': column_selector})

# Display the widget and the interactive plot
display(column_selector, interactive_plot)
```

## Time Series Plot 

```python

# Copy the original dataset
gdp_gt_dt = gdp_df.copy()

# Convert the 'Date' column to datetime
gdp_gt_dt['Date'] = pd.to_datetime(gdp_gt_dt['Date'])

# Set 'Year-Month' as the index
gdp_gt_dt.set_index('Date', inplace=True)

# Extract the GDP column from the DataFrame
gdp = gdp_gt_dt['GDP']

# Create a new figure for the plot with a specified size
plt.figure(figsize=(10, 6))

# Plot the GDP time series
# - gdp.plot(): Plots the GDP data as a time series
gdp.plot()

# Set the title of the plot
plt.title('GDP Time Series')

# Set the label for the x-axis
plt.xlabel('Date')

# Set the label for the y-axis
plt.ylabel("GDP")

# Add a grid to the plot for better readability
plt.grid(True)

# Adjust the layout to make sure everything fits without overlapping
plt.tight_layout()

# Display the plot
plt.show()
```

## Time Series with dropdown widget

```python
# Create a function to plot the time series
def plot_time_series(column):
    plt.figure(figsize=(10, 6))
    gdp_gt_dt[column].plot()
    plt.title(f'{column}')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Create a dropdown widget for selecting the column
column_selector = widgets.Dropdown(
    options=gdp_gt_dt.columns,
    description='Column:',
    disabled=False,
)

# Link the dropdown widget to the plot_time_series function
interactive_plot = widgets.interactive_output(plot_time_series, {'column': column_selector})

# Display the widget and the interactive plot
display(column_selector, interactive_plot)
```

## Correlation Analysis

### Correlation Table 

```python
klib.corr_mat(gdp_gt_dt)  
```

### Correlation Heatmap

```python
plt.figure(figsize=(12, 8))
sns.heatmap(gdp_gt_dt.corr(), 
            annot=True, 
            cmap='coolwarm',  # 'viridis', 'plasma', 'inferno', 'cividis'
            linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
```
### Correlation Heatmap (Remove the second half)

```python
klib.corr_plot(gdp_gt_dt, 
               figsize=(15,12),
               annot=True,
               cmap='coolwarm', # 'viridis', 'plasma', 'inferno', 'cividis'
               linewidths=0.9,
               method='pearson', # 'pearson', 'kendall', 'spearman'  
               )
```

### Correlation with target variables (1)

```python
klib.corr_mat(gdp_gt_dt, 
              method='pearson',
              target='GDP')
```

### Correlation with target variables (2)

```python
# Correlation of all features with the target
klib.corr_plot(gdp_gt_dt,
                target='GDP',
                figsize=(15,12),
                annot=True,
                cmap='coolwarm',
                linewidths=0.6,
                method='pearson', 
                annot_kws={'size': 12},
                ) 
```

### Positive Correlations

```python
# Positive correlations
klib.corr_plot(gdp_gt_dt, 
               split='pos', 
               figsize=(15,12),
               annot=True,
               cmap='coolwarm',
               linewidths=0.5,
               method='pearson') 
```

### Negative Correlations

```python
# Negative correlations
klib.corr_plot(gdp_gt_dt, 
               split='neg', 
               figsize=(15,12),
               annot=True,
               cmap='coolwarm',
               linewidths=-0.5,
               method='pearson'
               )
```

### Positive Correlation greater than certain threshold

```python
klib.corr_plot(gdp_gt_dt, 
               method='pearson',
               threshold=0.6,
               split='pos', # 'neg', 'low', 'high'
               figsize=(15,12),
               cmap='coolwarm',
               linewidths=0.5)
```

### Negative Correlation less than certain threshold

```python
klib.corr_plot(gdp_gt_dt, 
               method='pearson',
               threshold=-0.8,
               split='neg', # 'neg', 'low', 'high', 'pos'
               figsize=(15,12),
               cmap='coolwarm',
               linewidths=0.5)
```

## Bivariate Relationships (Bivariate Scatter and Pair Plot)

### Bivariate Scatter Plot

```python
# Create a new figure for the plot with a specified size
plt.figure(figsize=(10, 6))

# Create a scatter plot for GDP vs Economic Crisis
# - x=gdp_gt_dt['GDP']: The data for the x-axis
# - y=gdp_gt_dt['Economic Crisis']: The data for the y-axis
# - color='r': The color of the scatter points
# - data=gdp_gt_dt: The DataFrame containing the data
sns.scatterplot(x=gdp_gt_dt['GDP'], y=gdp_gt_dt['Economic Crisis'], color='r', data=gdp_gt_dt)

# Set the title of the plot with a specified font size
plt.title('GDP vs Economic Crisis', size=18)

# Set the label for the x-axis with a specified font size
plt.xlabel("GDP", size=14)

# Set the label for the y-axis with a specified font size
plt.ylabel("Economic Crisis", size=14)

# Display the plot
plt.show()
```

## Bivariate Scatter Plot with dropdown widget

```python
def plot_scatter(x_column, y_column):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_column, y=y_column, color='r', data=gdp_gt_dt)
    plt.title(f'{x_column} vs {y_column}', size=18)
    plt.xlabel(x_column, size=14)
    plt.ylabel(y_column, size=14)
    plt.show()

# Create dropdown widgets for selecting the columns
x_column_selector = widgets.Dropdown(
    options=gdp_gt_dt.columns,
    description='X Column:',
    disabled=False,
)

y_column_selector = widgets.Dropdown(
    options=gdp_gt_dt.columns,
    description='Y Column:',
    disabled=False,
)

# Link the dropdown widgets to the plot_scatter function
interactive_plot = widgets.interactive_output(plot_scatter, {'x_column': x_column_selector, 'y_column': y_column_selector})

# Display the widgets and the interactive plot
display(x_column_selector, y_column_selector, interactive_plot)
```

###  Bivariate Pair plot

#### Bivariate Pair plot (1)
```python
# Create a pair plot for the DataFrame gdp_gt_dt
# - gdp_gt_dt: The DataFrame containing the data
# - corner=True: Only plot the lower triangle of the pair plot to avoid redundancy
sns.pairplot(gdp_gt_dt, corner=True)

# Display the plot
plt.show()
```

#### Bivariate Pair plot (2)
```python
# Create a pair plot for the DataFrame gdp_gt_dt with various customizations
sns.pairplot(gdp_gt_dt, 
             # Use '+' markers for the scatter plots
             markers="+",
             # Use kernel density estimate (KDE) plots for the diagonal
             diag_kind="kde",
             # Use regression plots for the off-diagonal
             kind='reg',
             # Additional keyword arguments for the plots
             plot_kws={
                 # Customize the line in the regression plots
                 'line_kws': {'color': 'b'}, 
                 # Customize the scatter points
                 'scatter_kws': {'alpha': 0.8, 'color': 'red'}
             },
             # Only plot the lower triangle of the pair plot to avoid redundancy
             corner=True)

# Display the plot
plt.show()
```


## Automatic EDA Tools

```python
stdf = st.dfSummary(gdp_gt_dt)
stdf
```
## Exercise

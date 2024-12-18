---
title: Identifying and Handling Missing Data
teaching: 130
exercises: 130
questions:
- "How much of the data is missing? Is it a small fraction or a significant portion?"
- Are the missing values spread randomly across the dataset, or are they concentrated in certain variables or cases?
- How will removing or imputing data affect statistical assumptions, model performance, and interpretability?
- Is a simple approach (like mean or median imputation) sufficient, or do we need more sophisticated methods (like regression, KNN, or MICE)?
- How does the choice of imputation method preserve data distribution, relationships between variables, and seasonal/trend components?

objectives:
- "Learn the difference between deleting incomplete observations and imputing missing values."
- Gain insights into when to prefer simple imputation techniques (mean, median) versus more advanced methods (KNN, regression, MICE, STL).
- Recognize how to implement these methods using common Python libraries and functions.
- Develop the ability to choose the most appropriate missing data handling technique for a given dataset and analysis goal.  

keypoints:
- "Deletion: Simple but risks losing large amounts of data and introducing bias."
- "Imputation: Replaces missing values with informed guesses, preserving sample size but adding complexity."
- "Basic pandas Functions: Quickly fill missing values with constants or forward/backward fill."
- "Statistical Measures (Mean, Median, Mode): Straightforward but may not maintain underlying data relationships."
- "Interpolation (Linear, Polynomial, Spline): Uses trends in existing data to estimate missing points."
- "Regression and KNN Imputation: Leverages relationships between variables to predict missing values."
- "MICE and STL Imputation: Handles complex, multivariate, and seasonal/trend-based patterns, offering more robust data reconstruction."
- "Simpler methods are easier and faster to implement but may distort analyses."
- "Advanced methods better preserve data structure but can be more complex and resource-intensive." 
---

# Identifying and Handling Missing Data

- Missing data occurs when values are not stored for certain observations in a dataset.
- Unhandled missing data can lead to biased analyses, reduced statistical power, and potentially incorrect conclusions. Proper handling ensures more reliable insights.

## Approaches to Handling Missing Data
  
  - **Deletion**: Remove any row or observation that contains at least one missing value.
      - Pros:
         - Easy to implement.
      - Cons:
        - May result in significant data loss, reducing sample size.
        - Can lead to biased results.
        - Provides no insight into why data are missing.
      
   - **Imputation**: Replace missing values with estimated values derived from the available data.
  - Why Impute?: Preserves dataset size, potentially reduces bias, and retains the integrity of relationships among variables (if done correctly).  

## Common Imputation Techniques

    - Basic Methods:
       - Pandas Functions (fillna)
       - Mean, Median, & Mode Imputation
       
    - Time-Series Oriented Approaches
      - Forward or Backward Fill
      
    - Interpolation Methods
         - Linear Interpolation
         - Polynomial Interpolation
         - Spline Interpolation
    
    - Model-Based Methods
         - Regression Imputation
         - K-Nearest Neighbors (KNN) Imputation
         - Multiple Imputation by Chained Equations (MICE)
         - Seasonal Trend Decomposition using Loess (STL) Imputation
         
---
## Agenda

- Assessing Missing Data
  
  - Using Python (pandas) to identify missing values (isnull(), info())    
  - Quantifying the extent of missingness in a dataset.    
  - Visualizing missing patterns (e.g., using missingno library)  
 
- Simple Imputation Techniques
  
    - Mean, median, mode imputation    
    - Forward and backward fill     
    - Linear and polynomial interpolation  
  
- Advanced Imputation Techniques
    
  - Regression-based imputation    
  - K-Nearest Neighbors (KNN) imputation   
  - Multiple Imputation by Chained Equations (MICE)  
  - Seasonal and trend decomposition (STL)  
   
- Wrap-Up and Q&A
  - Recap of key points and best practices  

    
---


### Importing Libraries

```python
# Import pandas and numpy
import pandas as pd
import numpy as np

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import the datetime class from the datetime module
from pandas.tseries.offsets import MonthEnd

# Miissing data visualization
import missingno

# Interactive plotting libraries
import plotly.graph_objs as go
import plotly.express as px

# Linear Regressor Methods
from sklearn.linear_model import LinearRegression

# Import the KNNImputer class
from sklearn.impute import KNNImputer

# import mice imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# estimator models 
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor

# Import Seasonal Trend decomposition using Loess (STL)
from statsmodels.tsa.seasonal import STL

from PIL import Image
```

### Load data

```python
# Load GDP data without missing data 
gdpreal_govexp = pd.read_csv('gdpreal_govexp.csv')

gdpreal_govexp.head()
```

### Prepare the data for analysis/plotting 
  - Conver the Quarter column to datetime format  
  -  Fix the index of the GDP Data  
  - Convert 'RealGDP'  column to float
    
```python
# Convert the 'Quarter' column to datetime format and adjust for quarter-end
gdpreal_govexp['Quarter'] = pd.PeriodIndex(gdpreal_govexp['Quarter'], freq='Q').to_timestamp()

# Fix the index of the GDP Data to the [3, 6, 9, 12] or [1, 4, 7, 10] intervals
gdpreal_govexp['Quarter'] = gdpreal_govexp['Quarter'] + MonthEnd(3)

# Convert 'RealGDP' to float (ensure clean data)
gdpreal_govexp['RealGDP'] = gdpreal_govexp['RealGDP'].str.replace(',', '').astype(float)

gdpreal_govexp.head()
```


### Plot the Real GDP data using Matplotlib

```python
plt.figure(figsize=(12, 6))
plt.plot(
    gdpreal_govexp['Quarter'],
    gdpreal_govexp['RealGDP'],
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
### Plot the Real GDP data using Plotly

```python
# Create the plot
fig = go.Figure()

# Add the Real GDP line plot
fig.add_trace(go.Scatter(
    x=gdpreal_govexp['Quarter'],
    y=gdpreal_govexp['RealGDP'],
    mode='lines+markers',
    marker=dict(symbol='circle'),
    line=dict(dash='solid', color='blue'),
    name='Real GDP (Billion Naira)'
))

# Add titles and labels
fig.update_layout(
    title='Nigeria Real GDP (2010 Q1 - 2024 Q3)',
    xaxis_title='Quarter',
    yaxis_title='Real GDP (Billion Naira)',
    title_font_size=16,
    xaxis=dict(tickangle=45),
    yaxis=dict(tickformat=',', tickfont=dict(size=10)),
    legend=dict(font=dict(size=12)),
    margin=dict(l=20, r=20, t=40, b=20),
    template='plotly_white',
    width=1200,  # Set the width of the figure
    height=600   # Set the height of the figure
)

# Add grid lines
fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')
fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')

# Show the plot
fig.show()
```

### Identifing Missing Values

```python
# Missing value in each coloumns
mis_val = gdpreal_govexp.isnull().sum()

# Percentage of missing values
mis_val_percent = 100 * gdpreal_govexp.isnull().sum() / len(gdpreal_govexp)

# Make a table with the results
mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

# Rename the columns
mis_val_table_df = mis_val_table.rename(
columns = {0 : 'Missing Values', 1 : '% of Missing Values'})

mis_val_table_df
```

### Load GDP data with missing values 
```python
# Load GDP data with missing data 
gdpng_realdf_miss = pd.read_csv('gdpng_realdf_miss.csv')

gdpng_realdf_miss.head()
```

### Plot the GDP data with missing values

```python
plt.figure(figsize=(12, 6))
plt.plot(
    gdpng_realdf_miss['Quarter'],
    gdpng_realdf_miss['RealGDP'],
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

### Identifing Missing Values

```python
# Missing value in each coloumns
mis_val = gdpng_realdf_miss.isnull().sum()

# Percentage of missing values
mis_val_percent = 100 * gdpng_realdf_miss.isnull().sum() / len(gdpng_realdf_miss)

# Make a table with the results
mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

# Rename the columns
mis_val_table_df = mis_val_table.rename(
columns = {0 : 'Missing Values', 1 : '% of Missing Values'})

mis_val_table_df
```


## Showing the distribution of missing values

```python
missingno.matrix(gdpng_realdf_miss,
                 figsize=(12,6), 
                 fontsize=12, 
                 color= (0.27, 0.52, 1.0),
                 sparkline=False,
                 label_rotation=90)

plt.title('Missing Values Matrix', fontsize=14)
```

### Basic methods using pandas functions

1) Imputation with a single value (Constant Imputation)  
   - This is the most straightforward approach. We impute a single value for all missing values.

```python
# Replace all NaN elements with 15000
df_fill15k = gdpng_realdf_miss.fillna(15000)
df_fill15k.head()
```

  - Pandas Functions (fillna): Quickly fill with a fixed value or simple strategy.
       - Mean, Median, & Mode Imputation: Simple statistical measures; however, may distort variability and distributions.
         
    - Time-Series Oriented Approaches
      
      - Forward or Backward Fill: Propagates the last known value forward or backward, suitable for time-ordered data.  
    
    - Interpolation Methods
        
         - Linear Interpolation: Estimates missing values as points on a straight line between known values.
         - Polynomial Interpolation: Uses higher-order curves for more complex trends.
         - Spline Interpolation: Employs piecewise functions for smoother fits, often better for continuous data.  
    
    - Model-Based Methods
      
         - Regression Imputation: Uses regression models to predict missing values based on other features.
         - K-Nearest Neighbors (KNN) Imputation: Replaces missing values with averages or other statistics from the most similar observations.
         - Multiple Imputation by Chained Equations (MICE): Iteratively imputes multiple sets of plausible values, accounting for uncertainty and providing robust estimates.
         - Seasonal Trend Decomposition using Loess (STL) Imputation: Decomposes a time series into seasonal, trend, and residual components, then estimates missing values more accurately by leveraging these patterns.  












## Exercise

1) Identifying Missing Data:
   - Load a provided dataset and use df.info() and df.isnull().sum()` to identify how many missing values are present in each column.
   - Create a missingness heatmap using the missingno library.
   - Quantifying Impact of Missing Values:

2) Calculate what percentage of rows have missing values above a certain threshold (e.g., rows with more than 30% missing)?     
3) Observe how deleting rows/columns with too many missing values changes the dataset’s size and variable distributions?  
4) Use .fillna() to replace missing values in a numeric column with its mean and compare the distribution of the column before and after mean imputation (via histograms or descriptive statistics)?     
5) Apply forward fill imputation on a time-series column and discuss the potential implications?    
6) Use linear interpolation and compare results with mean/median imputation?       
7) Apply polynomial interpolation and compare differences in the resulting values?       
8) Implement KNN imputation using fancyimpute or sklearn-based methods?       
9) Perform multiple imputation using MICE (if a suitable library is available, such as statsmodels or impyute)?    
10) Compare model performance (e.g., a simple regression model trained on the dataset) before and after MICE imputation to observe potential improvement?   
11) Evaluating the quality of imputation using known values to measure error (e.g., mean absolute error, RMSE) and discuss how to interpret these errors and decide if the chosen imputation method is adequate?      
  
   


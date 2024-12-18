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

---

## Common Imputation Techniques

- Basic Methods
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

### Install important packages 
```python
pip install -U scikit-learn
pip install statsmodels
pip install pillow
pip install plotly
pip install missingno
```



### Load data

[GDP data csv](gdpreal_govexp.csv)

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

### Convert to the datetime and Set the index
```python
gdpreal_govexp['Quarter'] = pd.to_datetime(gdpreal_govexp['Quarter'])

# Set 'Quarter' as the index
gdpreal_govexp.set_index('Quarter', inplace=True)

gdpreal_govexp.head()
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

###  Convert to the datetime and Set the index
```python
# Set 'Quarter' as the index
gdpng_realdf_miss.set_index('Quarter', inplace=True)

gdpng_realdf_miss.index = pd.to_datetime(gdpng_realdf_miss.index)

gdpng_realdf_miss.head()
```

### Showing the distribution of missing values

```python
missingno.matrix(gdpng_realdf_miss,
                 figsize=(12,6), 
                 fontsize=12, 
                 color= (0.27, 0.52, 1.0),
                 sparkline=False,
                 label_rotation=90)

plt.title('Missing Values Matrix', fontsize=14)
```

### Get indices of missing values
```python
# Copy the dataframe that have missing value
df_with_missing = gdpng_realdf_miss.copy()

# Get indices of missing values
missing_indices = df_with_missing[df_with_missing['RealGDP'].isnull()].index
missing_indices
```

### Convert the 'Quarter' column to datetime

```python
gdpreal_govexp['Quarter'] = pd.to_datetime(gdpreal_govexp['Quarter'])

# Set 'Quarter' as the index
gdpreal_govexp.set_index('Quarter', inplace=True)

gdpreal_govexp.head()
```


### Basic methods using pandas functions

#### Imputation with a single value (Constant Imputation)  
   - This is the most straightforward approach. We impute a single value for all missing values.

```python
# Replace all NaN elements with 15000
df_fill15k = gdpng_realdf_miss.fillna(15000)
df_fill15k.head()
```

#### Plot the original GDP with the imputed data using constant imputation method

```
trace_imputed = go.Scatter(
    x=df_fill15k.index,
    y=df_fill15k['RealGDP'],
    mode='lines',
    name='Imputed Data',
    line=dict(color='red'),
    opacity=0.6  # Set the transparency level (0.0 to 1.0)
)

trace_non_missing = go.Scatter(
    x=gdpreal_govexp.index,
    y=gdpreal_govexp['RealGDP'],
    mode='lines',
    name='Non-Missing Data',
    line=dict(color='blue'),
    opacity=0.6  # Set the transparency level (0.0 to 1.0)
)

trace_missing_indice = go.Scatter(
    x=missing_indices,
    y=df_fill15k['RealGDP'],
    mode='markers',
    name='Missing Indice',
    marker=dict(color='red', size=6),
    opacity=0.6  # Set the transparency level (0.0 to 1.0)
)

# Create the layout for the plot
layout = go.Layout(
    title='Constant Imputation of Missing Values',
    xaxis=dict(title='Year'),
    yaxis=dict(title='RealGDP'),
    legend=dict(x=0, y=1)
)

# Create the figure and add the traces
fig = go.Figure(data=[trace_imputed, trace_non_missing, trace_missing_indice], layout=layout)

# Display the plot
fig.show()
```

#### Fill each individual columns with a specific value


```python
df_fill_spec15k = gdpng_realdf_miss.fillna(value={"RealGDP": 15000})
df_fill_spec15k.head()
```

#### Pandas Functions (fillna)
 - Quickly (imple statistical measures) fill with a fixed value or simple strategy, however, may distort variability and distributions.

```python
# Mean imputation
df_fill_mean = gdpng_realdf_miss.fillna(gdpng_realdf_miss['RealGDP'].mean())
df_fill_mean.head() 

# Median imputation
# df_fill_median = gdpng_realdf_miss.fillna(gdpng_realdf_miss['RealGDP'].median())
# df_fill_median

# Mode imputation
# df_fill_mode = gdpng_realdf_miss.fillna(gdpng_realdf_miss['RealGDP'].mode()[0])
# df_fill_mode

# Maximum imputation
# df_fill_max = gdpng_realdf_miss.fillna(gdpng_realdf_miss['RealGDP'].max())
# df_fill_max

# Minimum imputation
# df_fill_min = gdpng_realdf_miss.fillna(gdpng_realdf_miss['RealGDP'].min())
# df_fill_min
```

#### Plot the original GDP with the imputed data using Mean imputation method

```python
trace_imputed = go.Scatter(
    x=df_fill_mean.index,
    y=df_fill_mean['RealGDP'],
    mode='lines',
    name='Imputed Data',
    line=dict(color='red'),
    opacity=0.6  
)

trace_non_missing = go.Scatter(
    x=gdpreal_govexp.index,
    y=gdpreal_govexp['RealGDP'],
    mode='lines',
    name='Non-Missing Data',
    line=dict(color='blue'),
    opacity=0.6  
)

trace_missing_indice = go.Scatter(
    x=missing_indices,
    y=df_fill_mean['RealGDP'],
    mode='markers',
    name='Missing Indice',
    marker=dict(color='red', size=6),
    opacity=0.6  
)

# Create the layout for the plot
layout = go.Layout(
    title='Mean Imputation of Missing Values',
    xaxis=dict(title='Year-Month'),
    yaxis=dict(title='Real GDP (Billion Naira)'),
    legend=dict(x=0, y=1)
)

# Create the figure and add the traces
fig = go.Figure(data=[trace_imputed, trace_non_missing, trace_missing_indice], layout=layout)

# Display the plot
fig.show()
```

#### Time-Series Oriented Approaches 

##### Forward or Backward Fill

- Fill missing values with a `forward` or `backward` fill methods.  
- These methods replace missing values either with the **immediately preceding** observed value or the **subsequent** observed value (similar adjacent).

Pros:
   - Simple and easy to implement.  
   - Does not require any model fitting.  
   - Can be a reasonable assumption in certain time-series datasets.
       
Cons:  
   - It can introduce bias into the data if the assumption of similar adjacent values does not hold.  
   - It does not consider the possible variability around the missing value.  
   - It can lead to overestimation or underestimation of the data analysis.

```python
# Backward fill
df_fill_bfill = gdpng_realdf_miss.bfill() 
df_fill_bfill.head()

# Forward fill
# df_fill_ffill = gdpng_realdf_miss.ffill() 
# df_fill_ffill
```
#### Plot the original GDP with the imputed data using fill backward method

```python
trace_imputed = go.Scatter(
    x=df_fill_bfill.index,
    y=df_fill_bfill['RealGDP'],
    mode='lines',
    name='Imputed Data',
    line=dict(color='red'),
    opacity=0.6  
)

trace_non_missing = go.Scatter(
    x=gdpreal_govexp.index,
    y=gdpreal_govexp['RealGDP'],
    mode='lines',
    name='Non-Missing Data',
    line=dict(color='blue'),
    opacity=0.6  
)

trace_missing_indice = go.Scatter(
    x=missing_indices,
    y=df_fill_bfill['RealGDP'],
    mode='markers',
    name='Missing Indice',
    marker=dict(color='red', size=6),
    opacity=0.6  
)

# Create the layout for the plot
layout = go.Layout(
    title='Fill Forward Imputation of Missing Values',
    xaxis=dict(title='Year-Month'),
    yaxis=dict(title='RealGDP'),
    legend=dict(x=0, y=1)
)

# Create the figure and add the traces
fig = go.Figure(data=[trace_imputed, trace_non_missing, trace_missing_indice], layout=layout)

# Display the plot
fig.show()
```
    
#### Interpolation Methods

##### Linear Interpolation

- is an imputation technique that assumes a linear relationship between data points.  
- is a method of **curve fitting** used to estimate the value between two known values.  
- In the context of missing data, linear interpolation can be used to estimate the missing values by drawing a straight line between two points.
  
Pros  
 - Simple and fast.  
 - It can provide a good estimate for the missing values when the data shows a linear trend.  

Cons  
 - It can provide poor estimates when the data does not show a linear trend.  
 - It's not suitable for data with a seasonal pattern.

```python
linear_interpolation = Image.open('image/linear_interpolation.png')
display(linear_interpolation)
```
```python
df_interpolate = gdpng_realdf_miss.interpolate( method='linear')

df_interpolate.head()
```

#### Plot the original GDP with the imputed data using linear interpolation method

```python
trace_imputed = go.Scatter(
    x=df_interpolate.index,
    y=df_interpolate['RealGDP'],
    mode='lines',
    name='Imputed Data',
    line=dict(color='red'),
    opacity=0.6  
)

trace_non_missing = go.Scatter(
    x=gdpreal_govexp.index,
    y=gdpreal_govexp['RealGDP'],
    mode='lines',
    name='Non-Missing Data',
    line=dict(color='blue'),
    opacity=0.6  
)

trace_missing_indice = go.Scatter(
    x=missing_indices,
    y=df_interpolate['RealGDP'],
    mode='markers',
    name='Missing Indice',
    marker=dict(color='red', size=6),
    opacity=0.6  
)

# Create the layout for the plot
layout = go.Layout(
    title='Linear Interpolation Imputation of Missing Values',
    xaxis=dict(title='Year-Month'),
    yaxis=dict(title='RealGDP'),
    legend=dict(x=0, y=1)
)

# Create the figure and add the traces
fig = go.Figure(data=[trace_imputed, trace_non_missing, trace_missing_indice], layout=layout)

# Display the plot
fig.show()
```

##### Polynomial Interpolation

 - is a method of estimating values between known data points using polynomials.  
 - The idea is to find a polynomial of a certain degree that passes through all the given data points.   
 - This polynomial can then be used to estimate values at intermediate points.


```python
df_interpolate_poly = gdpng_realdf_miss.interpolate(method='polynomial', order=5)
df_interpolate_poly.head()
```

#### Plot the original GDP with the imputed data using polynomial interpolation method


```python
trace_imputed = go.Scatter(
    x=df_interpolate_poly.index,
    y=df_interpolate_poly['RealGDP'],
    mode='lines',
    name='Imputed Data',
    line=dict(color='red'),
    opacity=0.6  
)

trace_non_missing = go.Scatter(
    x= gdpreal_govexp.index,
    y=gdpreal_govexp['RealGDP'],
    mode='lines',
    name='Non-Missing Data',
    line=dict(color='blue'),
    opacity=0.6  
)

trace_missing_indice = go.Scatter(
    x=missing_indices,
    y=df_interpolate_poly['RealGDP'],
    mode='markers',
    name='Missing Indice',
    marker=dict(color='red', size=6),
    opacity=0.6  
)

# Create the layout for the plot
layout = go.Layout(
    title='Polynomial Interpolation Imputation of Missing Values',
    xaxis=dict(title='Year-Month'),
    yaxis=dict(title='RealGDP'),
    legend=dict(x=0, y=1)
)

# Create the figure and add the traces
fig = go.Figure(data=[trace_imputed, trace_non_missing, trace_missing_indice], layout=layout)

# Display the plot
fig.show()
```

##### Spline Interpolation

- is a method of interpolating data points using **piecewise polynomials** called **splines**.       
- Instead of fitting a single polynomial to all data points, spline interpolation fits multiple polynomials to subsets of the data points, ensuring smoothness at the boundaries where the polynomials meet.    

Pros:  

  - Provides a smoother and more flexible fit than linear interpolation.    
  - Does not suffer from the problem of overfitting that can occur with polynomial interpolation.  
    
Cons:  
  
  - More computationally intensive than linear interpolation.    
  - Can create unrealistic estimates if the data is not smooth.    

```python
df_interpolate_spline = gdpng_realdf_miss.interpolate(method='spline', order=2)
df_interpolate_spline.head()
```

#### Plot the original GDP with the imputed data using spline interpolation method


```python
trace_imputed = go.Scatter(
    x=df_interpolate_spline.index,
    y=df_interpolate_spline['RealGDP'],
    mode='lines',
    name='Imputed Data',
    line=dict(color='red'),
    opacity=0.6  
)

trace_non_missing = go.Scatter(
    x=gdpreal_govexp.index,
    y=gdpreal_govexp['RealGDP'],
    mode='lines',
    name='Non-Missing Data',
    line=dict(color='blue'),
    opacity=0.6  
)

trace_missing_indice = go.Scatter(
    x=missing_indices,
    y=df_interpolate_spline['RealGDP'],
    mode='markers',
    name='Missing Indice',
    marker=dict(color='red', size=6),
    opacity=0.6  
)

# Create the layout for the plot
layout = go.Layout(
    title='Spline Interpolation Imputation of Missing Values',
    xaxis=dict(title='Year-Month'),
    yaxis=dict(title='RealGDP'),
    legend=dict(x=0, y=1)
)

# Create the figure and add the traces
fig = go.Figure(data=[trace_imputed, trace_non_missing, trace_missing_indice], layout=layout)

# Display the plot
fig.show()
```
    
#### Model-Based Methods
      
##### Regression Imputation
 
 - Uses regression models to predict missing values based on other features.
 - It can be used when the data is numeric and there is a **strong correlation** between the variable with missing values and other variables.


```python
# Copy the original data 
df_with_missing = gdpng_realdf_miss.copy()
df_with_missing.head()
```

```python
# Drop the missing values 
df_non_missing = df_with_missing.dropna()
df_non_missing.head()
```

```python
# Instantiate linear model
model = LinearRegression()
```

```python
# Reshape data for model fitting (sklearn requires 2D array for predictors)
X = df_non_missing['Govexp'].values.reshape(-1, 1)
Y = df_non_missing['RealGDP'].values

X.shape, Y.shape
```

```python
# Fit the model
model.fit(X, Y)
```

```python
# Predict missing RealGDP values using rainfall
predicted_gdp = model.predict(df_with_missing.loc[missing_indices, "Govexp"].values.reshape(-1, 1))
predicted_gdp
```

```python
# Fill missing RealGDP values with predicted values
df_with_missing.loc[missing_indices, "RealGDP"] = predicted_gdp
df_with_missing.head()
```


#### Plot the original GDP with the imputed data using regression interpolation method


```python
trace_imputed = go.Scatter(
    x=df_with_missing.index,
    y=df_with_missing['RealGDP'],
    mode='lines',
    name='Imputed Data',
    line=dict(color='red'),
    opacity=0.6  # Set the transparency level (0.0 to 1.0)
)

trace_non_missing = go.Scatter(
    x=gdpreal_govexp.index,
    y=gdpreal_govexp['RealGDP'],
    mode='lines',
    name='Non-Missing Data',
    line=dict(color='blue'),
    opacity=0.6  # Set the transparency level (0.0 to 1.0)
)

trace_missing_indice = go.Scatter(
    x=missing_indices,
    y=df_with_missing['RealGDP'],
    mode='markers',
    name='Missing Imputation',
    marker=dict(color='red', size=10),
    opacity=0.6  # Set the transparency level (0.0 to 1.0)
)

# Create the layout for the plot
layout = go.Layout(
    title='Regression based Imputation',
    xaxis=dict(title='Year-Month'),
    yaxis=dict(title='RealGDP'),
    legend=dict(x=0, y=1)
)

# Create the figure and add the traces
fig = go.Figure(data=[trace_imputed, trace_non_missing, trace_missing_indice], layout=layout)

# Display the plot
fig.show()
```
  
##### K-Nearest Neighbors (KNN) Imputation

- is a machine learning-based imputation technique that uses the **k-nearest neighbors** algorithm to estimate missing values in a dataset.   
- Each sample's missing values are imputed using the mean value from n_neighbors nearest neighbors found in the training set.    
- It calculates the distance between data points and uses the values of the k-nearest neighbors to impute missing values. 

```python
df_with_missing = gdpng_realdf_miss.copy()

# Create an instance of the KNNImputer class
knn_imputer = KNNImputer(n_neighbors=4,
                         weights='distance') # uniform, distance 

# Impute missing values using the KNN imputer
df_knn_imputed = knn_imputer.fit_transform(df_with_missing)

# Convert the imputed data to a DataFrame
df_knn_imputed = pd.DataFrame(df_knn_imputed, 
                              columns=df_with_missing.columns, 
                              index=df_with_missing.index)

df_knn_imputed.head()
```

#### Plot the original GDP with the imputed data using KNN interpolation method


```python
trace_imputed = go.Scatter(
    x=df_knn_imputed.index,
    y=df_knn_imputed['RealGDP'],
    mode='lines',
    name='Imputed Data',
    line=dict(color='red'),
    opacity=0.6 )

trace_non_missing = go.Scatter(
    x=gdpreal_govexp.index,
    y=gdpreal_govexp['RealGDP'],
    mode='lines',
    name='Non-Missing Data',
    line=dict(color='blue'),
    opacity=0.6  )

trace_missing_indice = go.Scatter(
    x=missing_indices,
    y=df_knn_imputed['RealGDP'],
    mode='markers',
    name='Missing Imputation',
    marker=dict(color='red', size=10),
    opacity=0.6  # Set the transparency level (0.0 to 1.0)
)

# Create the layout for the plot
layout = go.Layout(
    title='K-Nearest Neighbors (KNN) based Imputation ',
    xaxis=dict(title='Year-Month'),
    yaxis=dict(title='RealGDP'),
    legend=dict(x=0, y=1)
)

# Create the figure and add the traces
fig = go.Figure(data=[trace_imputed, trace_non_missing, trace_missing_indice], layout=layout)

# Display the plot
fig.show()
```
  
##### Multiple Imputation by Chained Equations (MICE)

 - an **iterative imputation** technique that imputes missing values in a dataset by modeling each **feature with missing values** as a function of the **other features** in the dataset.  
- It iteratively imputes missing values in each feature using different models based on the other features.    
- It best suited for machine learning applications.      
- It also considers uncertainty when imputing the missing data.    

```python
df_with_missing = gdpng_realdf_miss.copy()


# Create an instance of the IterativeImputer class
mice_imputer = IterativeImputer(
estimator=RandomForestRegressor(), # BayesianRidge, LinearRegression, DecisionTreeRegressor, RandomForestRegressor, KNeighborsRegressor, ExtraTreesRegressor
random_state=0, 
initial_strategy='mean', # median, most_frequent, constant
max_iter=30,
verbose=0)


# Impute missing values using the MICE imputer
df_mice_imputed = mice_imputer.fit_transform(df_with_missing)

df_mice_imputed = pd.DataFrame(df_mice_imputed,
                               columns=df_with_missing.columns, 
                               index=df_with_missing.index)

df_mice_imputed.head()
```

#### Plot the original GDP with the imputed data using MICE interpolation method

```python
df_with_missing = gdpng_realdf_miss.copy()
trace_imputed = go.Scatter(
    x=df_mice_imputed.index,
    y=df_mice_imputed['RealGDP'],
    mode='lines',
    name='Imputed Data',
    line=dict(color='red'),
    opacity=0.6 )

trace_non_missing = go.Scatter(
    x=gdpreal_govexp.index,
    y=gdpreal_govexp['RealGDP'],
    mode='lines',
    name='Non-Missing Data',
    line=dict(color='blue'),
    opacity=0.6  )

trace_missing_indice = go.Scatter(
    x=missing_indices,
    y=df_mice_imputed['RealGDP'],
    mode='markers',
    name='Missing Imputation',
    marker=dict(color='red', size=10),
    opacity=0.6  # Set the transparency level (0.0 to 1.0)
)

# Create the layout for the plot
layout = go.Layout(
    title='MICE based Imputation',
    xaxis=dict(title='Year-Month'),
    yaxis=dict(title='RealGDP'),
    legend=dict(x=0, y=1)
)

# Create the figure and add the traces
fig = go.Figure(data=[trace_imputed, trace_non_missing, trace_missing_indice], layout=layout)

# Display the plot
fig.show()
```



  
##### Seasonal Trend Decomposition using Loess (STL) Imputation

- Seasonal Trend decomposition using Loess (STL) is a statistical method for decomposing a time series into three components: **trend**, **seasonal**, and **remainder (random)**.
   
- STL imputation can be used when dealing with time series data that exhibits a seasonal pattern.    
- It can be used for imputing missing data in a time series.     
    - the missing values are initially estimated via interpolation to allow for STL decomposition.       
    - afterward, the seasonal and trend components of the decomposed time series are extracted.       
    - The missing values are then re-estimated by interpolating the trend component and re-adding the seasonal component.  
  

```python
df_with_missing = df_with_missing.copy()

stl = STL(df_with_missing["RealGDP"].interpolate(method="linear",
                                                    limit_direction="both"), 
                                                    seasonal=13)
res = stl.fit()

# Extract the seasonal and trend components
seasonal_component = res.seasonal

# Create the deseasonalised series
df_deseasonalised = df_with_missing["RealGDP"] - seasonal_component

# Interpolate missing values in the deseasonalised series
df_deseasonalised_imputed = df_deseasonalised.interpolate(method="linear", limit_direction="both")

# Add the seasonal component back to create the final imputed series
df_imputed = df_deseasonalised_imputed + seasonal_component

# Update the original dataframe with the imputed values
df_with_missing.loc[missing_indices, "RealGDP"] = df_imputed[missing_indices]

df_with_missing.head()  
```

#### Plot the original GDP with the imputed data using STL interpolation method


```python
trace_imputed = go.Scatter(
    x=df_with_missing.index,
    y=df_with_missing['RealGDP'],
    mode='lines',
    name='Imputed Data',
    line=dict(color='red'),
    opacity=0.6 )

trace_non_missing = go.Scatter(
    x=gdpreal_govexp.index,
    y=gdpreal_govexp['RealGDP'],
    mode='lines',
    name='Non-Missing Data',
    line=dict(color='blue'),
    opacity=0.6  )

trace_missing_indice = go.Scatter(
    x=missing_indices,
    y=df_with_missing['RealGDP'],
    mode='markers',
    name='Missing Imputation',
    marker=dict(color='red', size=10),
    opacity=0.6  # Set the transparency level (0.0 to 1.0)
)

# Create the layout for the plot
layout = go.Layout(
    title='Seasonal Trend Decomposition using Loess (STL) based Imputation',
    xaxis=dict(title='Year-Month'),
    yaxis=dict(title='RealGDP'),
    legend=dict(x=0, y=1)
)

# Create the figure and add the traces
fig = go.Figure(data=[trace_imputed, trace_non_missing, trace_missing_indice], layout=layout)

# Display the plot
fig.show()
```

---

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
  
   


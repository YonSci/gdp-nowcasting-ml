
# Stationarity Tests in Time Series

- Time series is said to be stationary if its **statistical properties** do not change over time (it has constant mean and variance, and covariance is independent of time).
- If the target variable is non-stationary, models might struggle to capture patterns effectively, leading to poor forecasting performance.
- Most machine learning models like Random Forest, XGBoost, LightGBM, LSTMs, and other modern ML algorithms do not require the target variable to be stationary.

## How to Check Stationarity

1. **Visual**
2. **Global vs Local Test**
3. **Statistical Test**
   - Augmented Dickey-Fuller (ADF)
   - KPSS (Kwiatkowski, Phillips, Schmidt, and Shin) test

---

## Augmented Dickey-Fuller (ADF)

- The ADF test is used to determine whether a given time series is stationary or not.
- The test uses an autoregressive model across multiple lag values.

  - **Null Hypothesis (H0):** If accepted, it suggests the time series has some time-dependent structure (non-stationary).
  - **Alternate Hypothesis (H1):** If the null hypothesis is rejected, the time series is stationary.

- **Interpretation of Results:**
  - If **p-value < 0.05**, reject the null hypothesis and infer that the time series is stationary.
  - If **p-value > 0.05**, fail to reject the null hypothesis (accept that the series is non-stationary).

- **ADF Statistic:** Value of the test statistic. More negative values indicate stronger evidence against the null hypothesis.

- By setting `autolag='AIC'`, the ADF test chooses the number of lags that yields the lowest AIC.

---

## Importing Libraries

```python
# Import pandas and numpy
import pandas as pd
import numpy as np

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import the datetime class from the datetime module
from pandas.tseries.offsets import MonthEnd

# Stationarity and Autocorrelation Tests
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

# Statsmodels API
import statsmodels.api as sm

# For loading images
from PIL import Image
```

---

## Import the Data

```python
gdpreal = pd.read_csv('gdpreal.csv')

gdpreal.head()

gdpreal.tail()

# Convert the 'Quarter' column to datetime format and adjust for quarter-end
gdpreal['Quarter'] = pd.PeriodIndex(gdpreal['Quarter'], freq='Q').to_timestamp()
gdpreal['Quarter'] = gdpreal['Quarter'] + MonthEnd(3)

# Convert 'RealGDP' to float (ensure clean data)
gdpreal['RealGDP'] = gdpreal['RealGDP'].str.replace(',', '').astype(float)

gdpreal.head()
```

---

## Plotting Real GDP Data

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
plt.yticks(fontsize=10)
plt.ticklabel_format(axis='y', style='plain')

# Grid, legend, and layout
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.legend(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()
```

---

## Augmented Dickey-Fuller (ADF) Test

```python
def adf_test(series, column_name):
    result = adfuller(series)
    print(f'Augmented Dickey-Fuller Test for {column_name}:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is non-stationary.")
    print("-----------------------------")

# Check stationarity of the RealGDP column
adf_test(gdpreal['RealGDP'], 'RealGDP')
```

---

## KPSS (Kwiatkowski, Phillips, Schmidt, and Shin) Test

- The KPSS test has the null hypothesis opposite to that of the ADF test.

  - **Null Hypothesis (H0):** The process is trend stationary.
  - **Alternate Hypothesis (H1):** The time series has some time-dependent structure (non-stationary).

- **Regression Parameters:**
  - `'c'`: Includes a constant (or an intercept) in the regression.
  - `'nc'`: No constant or trend.
  - `'ct'`: Constant and trend.
  - `'ctt'`: Constant, linear, and quadratic trend.

```python
def kpss_test(series, column_name):
    result = kpss(series, regression='c')
    print(f'KPSS Test for {column_name}:')
    print(f'KPSS Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print("The series is stationary.")
    else:
        print("The series is non-stationary.")
    print("-----------------------------")

# Check stationarity
kpss_test(gdpreal['RealGDP'], 'RealGDP')
```

---

## Autocorrelation and Partial Autocorrelation Check

### Autocorrelation (ACF)

- Measures the linear relationship between a time series and its lagged values.
- Helps identify MA (Moving Average) order and patterns like seasonality.

### Partial Autocorrelation (PACF)

- Measures the direct relationship between a time series and its lagged values, excluding intermediate lags.
- Useful for identifying the AR (Autoregressive) order.

```python
def plot_acf(series, column_name, lags=None):
    plt.figure(figsize=(18, 6))
    sm.graphics.tsa.plot_acf(series, lags=lags, alpha=0.05)
    plt.title(f'Autocorrelation Function (ACF) for "{column_name}"', fontsize=14)
    plt.xlabel("Lags", fontsize=12)
    plt.ylabel("Autocorrelation", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_acf(gdpreal['RealGDP'], 'RealGDP', lags=40)
```

### Partial Autocorrelation Plot

```python
def plot_pacf(series, column_name, lags=None):
    plt.figure(figsize=(12, 6))
    sm.graphics.tsa.plot_pacf(series, lags=lags, alpha=0.05, method='ywm')
    plt.title(f'Partial Autocorrelation Function (PACF) for "{column_name}"', fontsize=14)
    plt.xlabel("Lags", fontsize=12)
    plt.ylabel("Partial Autocorrelation", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_pacf(gdpreal['RealGDP'], 'RealGDP')
```

---

### Interpretation of ACF and PACF

- **Significant ACF lags:** Indicates strong short-term correlation.
- **Significant PACF lags:** Indicates direct influence of recent observations.
- After lag 10, ACF becomes insignificant, suggesting mean-reverting behavior.
- PACF suggests an AR(3) structure with significant lags up to 3 and a correction at lag 4.

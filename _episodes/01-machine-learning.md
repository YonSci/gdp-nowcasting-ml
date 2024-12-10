---
title: Machine Learning Fundamentals
teaching: 1
exercises: 0
questions:
- "What are the fundamental concepts in ML I can use in sklearn framewrok ?"
- "How do I write documentation for my ML code?"
- "How do I train and test ML models for Physical Sciences Problems?"
objectives:
- "Gain an understanding of fundamental machine learning concepts relevant to physical sciences."
- "Develop proficiency in optimizing data preprocessing techniques for machine learning tasks in Python."
- "Learn and apply best practices for training, evaluating, and interpreting machine learning models in the domain of physical sciences."
keypoints:
- "Data representations are crucial for ML in science, including spatial data (vector, raster), point clouds, time series, graphs, and more"
- "ML algorithms like linear regression, k-nearest neighbors,support vector Machine, xgboost and random forests are vital algorithms"
- "Supervised learning is a popular ML approach, with decision trees, random forests, and neural networks being widely used"
- "Fundamentals of data engineering are crucial for building robust ML pipelines, including data storage, processing, and serving"
---

<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>


# Machine Learning Concepts

Machine learning is a field of study that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves


## Types of ML Algorithms

Types of Machine Learning
There are three main categories of machine learning algorithms:

- Supervised Learning: Algorithms learn from labeled training data to predict outcomes. It includes classification (predicting categories) and regression (predicting numerical values).
- Unsupervised Learning: Algorithms find hidden patterns in unlabeled data. A common task is clustering, which groups similar examples together.
- Reinforcement Learning: Algorithms learn by interacting with an environment and receiving rewards or penalties for actions to maximize performance

![](../fig/ML_type.png)


## Supervised Learning

### Regression

Regression is a statistical method used to examine the relationship between one or more independent variables (predictors) and a dependent variable (response). It aims to predict continuous numerical outcomes. In regression analysis, the goal is to find the best-fitting line or curve that describes the relationship between the independent and dependent variables. Examples of regression techniques include Linear Regression, Polynomial Regression, Ridge Regression, Lasso Regression, Decision Tree Regression and etc.

**Linear regression** is a fundamental statistical method used to model the relationship between a dependent variable  Y and one or more independent variables 𝑋. The goal is to find the **best-fitting linear equation** that can predict the dependent variable based on the values of the independent variables.

**The Linear Regression Model**
The linear regression model can be represented as:

* \\((y = mx + b)\\), 
where \(y\) is the predicted output, \\(m\\) is the slope (coefficients associated with features), \\(x\\) is the input feature, and \\(b\\) is the intercept.


* \\[Y = \Theta + \Theta_1\cdot X_1 + \Theta_2\cdot X_2 + ... \Theta_p\cdot X_p + \epsilon\\],  where: \\(\Theta_0\\) is the intercept, (\\(\Theta_1, \Theta_2, …,\Theta_𝑝\\)) are the coefficients (weights), (\\(X_1, X_2, …,X_𝑝\\)) are the independent variables, \\(ϵ\\) is the error term.

**Cost Function**
The cost function, also known as the loss function, measures the performance of the linear regression model. The most commonly used cost function for linear regression is the Mean Squared Error (MSE), which is defined as:

\\[J(\Theta) = \frac{1}{2m} \sum_{i=1}^m (h_\Theta(x^{(i)}) - y^{(i)} )^2\\]


*  where \\(h_\Theta(x^{(i)})\\) represents the predicted output value for the ith training example, \\(y^{(i)}\\) represents the actual output value for the ith training example, and m is the total number of training examples.
* The goal of training a linear regression model is to minimize the cost function J(Θ) by finding the optimal values of the model parameters Θ.
* The process of minimizing the cost function J(Θ) is usually done using gradient descent, a popular optimization algorithm that iteratively adjusts the model parameters in the direction of steepest descent of the cost function until convergence is reached.
* The value of the cost function J(Θ) can be used to evaluate the performance of the trained model on a test set or on new, unseen data. A lower value of J(Θ) indicates a better fit of the model to the data.

  #### Gradient Descent
* Gradient descent is an optimization algorithm used to minimize the cost function by iteratively adjusting the model parameters (coefficients). The update rule for gradient descent in the context of linear regression is:

Minimize the cost function \\(J(\Theta)\\)

By updating Equation and repeat unitil convergence
        
\\[\Theta_j := \Theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_{\Theta}(x^{(i)}) - y^{(i)})x_j^{(i)}\\] (simultaneously update \\(\Theta_j\\) for all \\(j\\))


- **Regularization:** Linear Regression does not inherently include regularization, making it susceptible to overfitting if the number of features is large compared to the number of samples.

**Lasso Regression (L1 Regularization):**
- **Objective:** Lasso Regression extends linear regression by introducing **L1 regularization**, which adds a penalty term based on the **absolute values of the coefficients**.
- **Equation:** The Lasso regression objective function includes the sum of squared differences and a penalty term proportional to the absolute values of the coefficients.
- **Benefits:** Lasso tends to produce sparse models, i.e., it can force some coefficients to be exactly zero, effectively performing feature selection.
- **Use Case:** It is particularly useful when dealing with high-dimensional datasets with many features.

**Ridge Regression (L2 Regularization):**
- **Objective:** Ridge Regression is another extension of linear regression, but it introduces **L2 regularization**, adding a penalty term based on the **squared values of the coefficients**.
- **Equation:** The Ridge regression objective function includes the sum of squared differences and a penalty term proportional to the squared values of the coefficients.
- **Benefits:** Ridge helps prevent multicollinearity by shrinking the coefficients, and it is effective when there are many correlated features.
- **Use Case:** It is commonly used in situations where multicollinearity is a concern, as it can stabilize the model and produce more reliable coefficients.

**Comparison:**
- **Regularization Types:** Linear Regression has no regularization, Lasso uses L1 regularization, and Ridge uses L2 regularization.
- **Effect on Coefficients:** Lasso can lead to sparse models with some coefficients exactly zero, effectively performing feature selection. Ridge tends to shrink coefficients toward zero without making them exactly zero.
- **Use Cases:** Lasso is beneficial when feature selection is crucial, while Ridge is useful for preventing multicollinearity and stabilizing the model.

#### Linear Regression
~~~
# Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict
y_pred_linear = linear_model.predict(X_test)

# Calculate Mean Squared Error
mse_linear = mean_squared_error(y_test, y_pred_linear)

# Print coefficients and MSE
print("Linear Regression Coefficients:", linear_model.coef_)
print("Linear Regression Intercept:", linear_model.intercept_)
print("Linear Regression MSE:", mse_linear)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Data')
plt.plot(X_test, y_pred_linear, label='Linear Regression', color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression')
plt.show()
~~~
{: .python}

~~~
Linear Regression Coefficients: [[2.79932366]]
Linear Regression Intercept: [4.14291332]
Linear Regression MSE: 0.6536995137170021
~~~
{: .output}

![](../fig/LiR1.png)


#### Lasso

~~~
#Lasso
from sklearn.linear_model import Lasso

# Initialize and train the lasso regression model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# Predict
y_pred_lasso = lasso_model.predict(X_test)

# Calculate Mean Squared Error
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Print coefficients and MSE
print("Lasso Regression Coefficients:", lasso_model.coef_)
print("Lasso Regression Intercept:", lasso_model.intercept_)
print("Lasso Regression MSE:", mse_lasso)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Data')
plt.plot(X_test, y_pred_lasso, label='Lasso Regression', color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Lasso Regression')
plt.show()

~~~
{: .python}

~~~
Lasso Regression Coefficients: [2.50457141]
Lasso Regression Intercept: [4.41885953]
Lasso Regression MSE: 0.6584189249611411
~~~
{: .output}

![](../fig/Lasso1.png)

~~~
from sklearn.linear_model import Ridge

# Initialize and train the ridge regression model
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)

# Predict
y_pred_ridge = ridge_model.predict(X_test)

# Calculate Mean Squared Error
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

# Print coefficients and MSE
print("Ridge Regression Coefficients:", ridge_model.coef_)
print("Ridge Regression Intercept:", ridge_model.intercept_)
print("Ridge Regression MSE:", mse_ridge)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred_ridge, label='Ridge Regression', color='green')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Ridge Regression')
plt.show()
~~~
{: .python}

~~~
Ridge Regression Coefficients: [[2.69985029]]
Ridge Regression Intercept: [4.23604]
Ridge Regression MSE: 0.647613237305426
~~~
{: .output}

![](../fig/Rigde1.png)


### Example- Air Quality Prediction

An air quality index (AQI) is used by government agencies to communicate to the public how polluted the air currently is or how polluted it is forecast to become. Public health risks increase as the AQI rises. Different countries have their own air quality indices, corresponding to different national air quality standards.

For Air quality prediction we will use 4 algorithms:

1.Linear Regression

2.Lasso Regression

3.Ridge Regression

4.Decision Tree Regressor

By using the above algorithms, we will train our model by providing training data and once the model will be trained, we will perform prediction. After prediction, we will evaluate the performance of these algorithmns by error check and accuracy check.

~~~
# Data Exploration
data=pd.read_csv('city_day.csv')
data
~~~
{: .python}

~~~

City	Date	PM2.5	PM10	NO	NO2	NOx	NH3	CO	SO2	O3	Benzene	Toluene	Xylene	AQI	AQI_Bucket
0	Ahmedabad	2015-01-01	NaN	NaN	0.92	18.22	17.15	NaN	0.92	27.64	133.36	0.00	0.02	0.00	NaN	NaN
1	Ahmedabad	2015-01-02	NaN	NaN	0.97	15.69	16.46	NaN	0.97	24.55	34.06	3.68	5.50	3.77	NaN	NaN
2	Ahmedabad	2015-01-03	NaN	NaN	17.40	19.30	29.70	NaN	17.40	29.07	30.70	6.80	16.40	2.25	NaN	NaN
3	Ahmedabad	2015-01-04	NaN	NaN	1.70	18.48	17.97	NaN	1.70	18.59	36.08	4.43	10.14	1.00	NaN	NaN
4	Ahmedabad	2015-01-05	NaN	NaN	22.10	21.42	37.76	NaN	22.10	39.33	39.31	7.01	18.89	2.78	NaN	NaN
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
29526	Visakhapatnam	2020-06-27	15.02	50.94	7.68	25.06	19.54	12.47	0.47	8.55	23.30	2.24	12.07	0.73	41.0	Good
29527	Visakhapatnam	2020-06-28	24.38	74.09	3.42	26.06	16.53	11.99	0.52	12.72	30.14	0.74	2.21	0.38	70.0	Satisfactory
29528	Visakhapatnam	2020-06-29	22.91	65.73	3.45	29.53	18.33	10.71	0.48	8.42	30.96	0.01	0.01	0.00	68.0	Satisfactory
29529	Visakhapatnam	2020-06-30	16.64	49.97	4.05	29.26	18.80	10.03	0.52	9.84	28.30	0.00	0.00	0.00	54.0	Satisfactory
29530	Visakhapatnam	2020-07-01	15.00	66.00	0.40	26.85	14.05	5.20	0.59	2.10	17.05	NaN	NaN	NaN	50.0	Good
29531 rows × 16 columns
~~~
{: .output}


~~~
data.columns
~~~
{: .python}

~~~
Index(['City', 'Date', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
       'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI', 'AQI_Bucket'],
      dtype='object')
~~~
{: .output}


~~~
# Data cleaning
data.isnull().sum()
~~~
{: .python}

~~~
City              0
Date              0
PM2.5          4598
PM10          11140
NO             3582
NO2            3585
NOx            4185
NH3           10328
CO             2059
SO2            3854
O3             4022
Benzene        5623
Toluene        8041
Xylene        18109
AQI            4681
AQI_Bucket     4681
dtype: int64
~~~
{: .output}


~~~
# Get the data types of all columns
column_data_types = data.dtypes

# Separate columns into numerical and categorical
numerical_columns = column_data_types[column_data_types != 'object'].index.tolist()
categorical_columns = column_data_types[column_data_types == 'object'].index.tolist()

# Print the lists of numerical and categorical columns
print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)
~~~
{: .python}

~~~
Numerical columns: ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
Categorical columns: ['City', 'Date', 'AQI_Bucket']
~~~
{: .output}


~~~
# Create a SimpleImputer object to impute missing values with the mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent') # mean

# Impute missing values in the 'age' and 'blood_pressure' columns
data[['PM2.5', 'PM10','NO','NO2', 'NOx', 'NH3','CO','SO2', 'O3', 'Benzene','Toluene','Xylene', 'AQI']] = imputer.fit_transform(data[['PM2.5', 'PM10','NO','NO2', 'NOx', 'NH3','CO','SO2', 'O3', 'Benzene','Toluene','Xylene', 'AQI']])
~~~
{: .python}

~~~
data.isnull().sum()
~~~
{: .python}

~~~
City             0
Date             0
PM2.5            0
PM10             0
NO               0
NO2              0
NOx              0
NH3              0
CO               0
SO2              0
O3               0
Benzene          0
Toluene          0
Xylene           0
AQI              0
AQI_Bucket    4681
dtype: int64
~~~
{: .output}


~~~

#Drop unwanted columns.
newdata=data.drop(['City', 'Date','NOx', 'NH3','Benzene', 'Toluene', 'Xylene', 'AQI_Bucket'],axis=1)
newdata.columns
~~~
{: .python}

~~~
Index(['PM2.5', 'PM10', 'NO', 'NO2', 'CO', 'SO2', 'O3', 'AQI'], dtype='object')
~~~
{: .output}


~~~
#converting data into int datatype to avoid errors below.
prepareddata=newdata.astype(int)
prepareddata.head()
~~~
{: .python}

~~~
	PM2.5	PM10	NO	NO2	CO	SO2	O3	AQI
0	11	94	0	18	0	27	133	102
1	11	94	0	15	0	24	34	102
2	11	94	17	19	17	29	30	102
3	11	94	1	18	1	18	36	102
4	11	94	22	21	22	39	39	102
~~~
{: .output}


~~~
#Data visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
x=prepareddata['AQI']
y1=prepareddata['PM2.5']
y2=prepareddata['PM10']
y3=prepareddata['NO']
y4=prepareddata['NO2']
y5=prepareddata['CO']
y6=prepareddata['SO2']
y7=prepareddata['O3']
plt.figure(figsize=(15,8))
plt.scatter(x,y1,label='PM 2.5',color='salmon')
plt.scatter(x,y2,label='PM 10',color='palegreen')
plt.scatter(x,y3,label='NO',color='yellow')
plt.scatter(x,y4,label='NO2',color='steelblue')
plt.scatter(x,y5,label='CO',color='lime')
plt.scatter(x,y6,label='SO2',color='violet')
plt.scatter(x,y7,label='O3',color='springgreen')
plt.title('AQI and its Pollutents',fontsize=18)
plt.xlabel('AQI',fontsize=14)
plt.ylabel('Value',fontsize=14)
plt.legend()
plt.show()
~~~
{: .python}

![](../fig/visualization.png)

~~~
#to find correlation between different columns.
corr = prepareddata.corr() 
sns.heatmap(corr, annot=True)
~~~
{: .python}

![](../fig/corr_airpollution.png)

**Insights** :

1.When the value of pollutents is less, Air Quality Index (AQI) is less.

2.AQI highly depends on 

* ground-level ozone

* particle pollution (also known as particulate matter, including PM2.5 and PM10)

* carbon monoxide

* sulfur dioxide

* nitrogen dioxide


 **Data training** 


~~~
# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split
# Here, X is the data which will have features and y will have our target i.e. Air Quality Index(AQI).
x=prepareddata[['PM2.5', 'PM10', 'NO', 'NO2','CO', 'SO2','O3']]  
y=prepareddata['AQI']
~~~
{: .python}

~~~
# Split data into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) 
#Ratio used for splitting training and testing data is 8:2 respectively
~~~
{: .python}

### **Model Creation**

#### 1. Linear Regression

~~~
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Importing linear regression model
reg1 = LinearRegression()
# Fitting data into the model.
reg1.fit(x_train, y_train)
# Making predictions 
pred1 = reg1.predict(x_test)

# Evaluation metrics
mse = mean_squared_error(y_test, pred1)
mae = mean_absolute_error(y_test, pred1)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred1)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")
~~~
{: .python}

~~~
Mean Squared Error (MSE): 3895.522779525787
Mean Absolute Error (MAE): 34.1526735890227
Root Mean Squared Error (RMSE): 62.414123237659815
R-squared (R²): 0.787273226848741
~~~
{: .output}

#### Normalizing Data
To minimize errors, let's normalize the target variable and see how it affects the results.

* Data normalization is a preprocessing technique used in data analysis and machine learning to scale and transform the features of a dataset. 
* The goal is to bring the values of different features to a similar scale, preventing certain features from dominating the others due to their inherent magnitude. 
* This process aids in improving the performance of algorithms, especially those sensitive to the scale of input features.


~~~
from sklearn.preprocessing import StandardScaler

# Normalize only the target variable y
scaler = StandardScaler()
y_normalized = scaler.fit_transform(y.values.reshape(-1, 1))
~~~
{: .python}

~~~
x_train, x_test, y_train, y_test = train_test_split(x, y_normalized, test_size=0.2) 
~~~
{: .python}


~~~
reg11 = LinearRegression()
# Fitting data into the model.
reg11.fit(x_train, y_train)
# Making predictions 
pred11 = reg11.predict(x_test)

# Evaluation metrics
mse = mean_squared_error(y_test, pred11)
mae = mean_absolute_error(y_test, pred11)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred11)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")
~~~
{: .python}

~~~
Mean Squared Error (MSE): 0.19359671932407133
Mean Absolute Error (MAE): 0.25973107295720904
Root Mean Squared Error (RMSE): 0.4399962719433783
R-squared (R²): 0.8056477921764784
~~~
{: .output}

#### 2.  Lasso

~~~
from sklearn.linear_model import Lasso

# Importing Lasso regression model
reg2 = Lasso()
# Fitting data into the model.
reg2.fit(x_train, y_train)
# Making predictions 
pred2 = reg2.predict(x_test)

# Evaluation metrics
mse = mean_squared_error(y_test, pred2)
mae = mean_absolute_error(y_test, pred2)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred2)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")
~~~
{: .python}

~~~
Mean Squared Error (MSE): 0.2259483677459704
Mean Absolute Error (MAE): 0.27506376480375005
Root Mean Squared Error (RMSE): 0.47534026522689027
R-squared (R²): 0.7731698952395925
~~~
{: .output}

#### 3.  Ridge
~~~
from sklearn.linear_model import Ridge

# Importing Ridge regression model
reg3 = Ridge()
# Fitting data into the model.
reg3.fit(x_train, y_train)
# Making predictions 
pred3 = reg3.predict(x_test)

# Evaluation metrics
mse = mean_squared_error(y_test, pred3)
mae = mean_absolute_error(y_test, pred3)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred3)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

~~~
{: .python}

~~~
Mean Squared Error (MSE): 0.19359676998518713
Mean Absolute Error (MAE): 0.25973112535973847
Root Mean Squared Error (RMSE): 0.43999632951331213
R-squared (R²): 0.8056477413176636
~~~
{: .output}

#### 4.  Decision Tree

~~~
from sklearn.tree import DecisionTreeRegressor

# Importing Decision Tree Regressor model
reg4 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10)
# Fitting data into the model.
reg4.fit(x_train, y_train)
# Making predictions 
pred4 = reg4.predict(x_test)

# Evaluation metrics
mse = mean_squared_error(y_test, pred4)
mae = mean_absolute_error(y_test, pred4)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred4)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")
~~~
{: .python}

~~~
Mean Squared Error (MSE): 0.13564028035880735
Mean Absolute Error (MAE): 0.17782375739941364
Root Mean Squared Error (RMSE): 0.3682937419490146
R-squared (R²): 0.8638303993498619
~~~
{: .output}


> ## Exercise: RandomForestRegressor
> - Apply the same procedure for RandomForestRegressor.
> 
> > ## Solution
> > ~~~
> > from sklearn.ensemble import RandomForestRegressor
>> from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
> > from sklearn.preprocessing import StandardScaler
> > scaler = StandardScaler()
> > y_train_normalized = scaler.fit_transform(y_train.reshape(-1, 1))
> > y_test_normalized = scaler.transform(y_test.reshape(-1, 1))
> > reg5 = RandomForestRegressor(n_estimators=100, random_state=42)
> > reg5.fit(x_train, y_train_normalized)
> > pred5 = reg5.predict(x_test)
> > pred5_unnormalized = scaler.inverse_transform(pred5.reshape(-1, 1))
> > mse = mean_squared_error(y_test, pred5_unnormalized)
> > mae = mean_absolute_error(y_test, pred5_unnormalized)
> > rmse = np.sqrt(mse)
> > r2 = r2_score(y_test, pred5_unnormalized)
> > print(f"Mean Squared Error (MSE): {mse}")
> > print(f"Mean Absolute Error (MAE): {mae}")
> > print(f"Root Mean Squared Error (RMSE): {rmse}")
> > print(f"R-squared (R²): {r2}")
> > ~~~
> > {: .python}
> > ~~~
> > Mean Squared Error (MSE): 0.10985359506420236
>> Mean Absolute Error (MAE): 0.15927167807287307
>> Root Mean Squared Error (RMSE): 0.33144169180144245
>> R-squared (R²): 0.8897177141605406
> > ~~~
> > {: .output}
> {: .solution}
{: .challenge}


### Classification

Classification, a cornerstone of supervised learning, is invaluable in physical sciences for predicting categorical labels from historical data. Examples of such problem in physical sciences are:
1. Binary: Distinguishing healthy from diseased cells in medical images.
2. Multiclass: Classifying galaxy types based on their features in astronomy.
3. Multilabel: Identifying multiple geological formations in satellite images in geology.

Example: Weather Pattern Classification in Climate Science
Objective: Predict tomorrow's weather type (e.g., sunny, cloudy, rainy, snowy) from historical weather data.

Dataset:
- Features: Temperature, humidity, wind speed, atmospheric pressure, past weather.
- Labels: Weather types.

Our training explores various classification methods, focusing on predicting weather patterns to enhance understanding in physical sciences. Let's start with importing libraries and 

~~~
import numpy as np
import pandas as pd
data = pd.read_csv('weatherAUS.csv')
data.head()
~~~
{: .python}

~~~
Date	Location	MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustDir	WindGustSpeed	WindDir9am	...	Humidity9am	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	RainToday	RainTomorrow
0	01-12-2008	Albury	13.4	22.9	0.6	NaN	NaN	W	44.0	W	...	71.0	22.0	1007.7	1007.1	8.0	NaN	16.9	21.8	No	No
1	02-12-2008	Albury	7.4	25.1	0.0	NaN	NaN	WNW	44.0	NNW	...	44.0	25.0	1010.6	1007.8	NaN	NaN	17.2	24.3	No	No
2	03-12-2008	Albury	12.9	25.7	0.0	NaN	NaN	WSW	46.0	W	...	38.0	30.0	1007.6	1008.7	NaN	2.0	21.0	23.2	No	No
3	04-12-2008	Albury	9.2	28.0	0.0	NaN	NaN	NE	24.0	SE	...	45.0	16.0	1017.6	1012.8	NaN	NaN	18.1	26.5	No	No
4	05-12-2008	Albury	17.5	32.3	1.0	NaN	NaN	W	41.0	ENE	...	82.0	33.0	1010.8	1006.0	7.0	8.0	17.8	29.7	No	No
5 rows × 23 columns
~~~
{: .output}


> ## Exercise: Data Type Information and Non-Null Counts
> - Write a code in Pandas to print the DataFrame information.
> 
> > ## Solution
> > ~~~
> > df.info()
> > ~~~
> > {: .python}
> > ~~~
> > <class 'pandas.core.frame.DataFrame'>
> > RangeIndex: 145460 entries, 0 to 145459
> > Data columns (total 23 columns):
> >  #   Column         Non-Null Count   Dtype  
> > ---  ------         --------------   -----  
> > 0   Date           145460 non-null  object 
> > 1   Location       145460 non-null  object 
> > 2   MinTemp        143975 non-null  float64
> > 3   MaxTemp        144199 non-null  float64
> > 4   Rainfall       142199 non-null  float64
> > 5   Evaporation    82670 non-null   float64
> > 6   Sunshine       75625 non-null   float64
> > 7   WindGustDir    135134 non-null  object 
> > 8   WindGustSpeed  135197 non-null  float64
> > 9   WindDir9am     134894 non-null  object 
> > 10  WindDir3pm     141232 non-null  object 
> > 11  WindSpeed9am   143693 non-null  float64
> > 12  WindSpeed3pm   142398 non-null  float64
> > 13  Humidity9am    142806 non-null  float64
> > 14  Humidity3pm    140953 non-null  float64
> > 15  Pressure9am    130395 non-null  float64
> > 16  Pressure3pm    130432 non-null  float64
> > 17  Cloud9am       89572 non-null   float64
> > 18  Cloud3pm       86102 non-null   float64
> > 19  Temp9am        143693 non-null  float64
> > 20  Temp3pm        141851 non-null  float64
> > 21  RainToday      142199 non-null  object 
> > 22  RainTomorrow   142193 non-null  object 
> > dtypes: float64(16), object(7)
> > memory usage: 25.5+ MB
> > ~~~
> > {: .output}
> {: .solution}
{: .challenge}


### Data Preprocessing

Data preprocessing is a critical step in preparing raw data for machine learning algorithms. It ensures that the data is consistent, free of missing values, and ready for analysis.

Before applying any predictive algorithm, it's essential to preprocess the data. This step is crucial and cannot be overlooked. For our dataset, we follow these preprocessing steps:

1. **Feature Selection**: We drop features that don't contribute significantly to predicting the target variable, RainTomorrow. To decide which columns to drop, we consider:

   a. **Missing Values**: Columns with a high proportion of missing values may be less useful unless they are highly predictive.

   b. **Correlation with Target**: Features that have little or no correlation with the target variable may be less important.

   c. **Domain Knowledge**: Certain features may be more relevant based on our knowledge of weather prediction.

**Calculate missing value percentages for all columns**

~~~
data.isnull().sum()
~~~
{: .python}

~~~
Date                 0
Location             0
MinTemp           1485
MaxTemp           1261
Rainfall          3261
Evaporation      62790
Sunshine         69835
WindGustDir      10326
WindGustSpeed    10263
WindDir9am       10566
WindDir3pm        4228
WindSpeed9am      1767
WindSpeed3pm      3062
Humidity9am       2654
Humidity3pm       4507
Pressure9am      15065
Pressure3pm      15028
Cloud9am         55888
Cloud3pm         59358
Temp9am           1767
Temp3pm           3609
RainToday         3261
RainTomorrow      3267
dtype: int64
~~~
{: .output}
~~~
# Calculate missing value percentages for all columns
missing_percentages = (data.isnull().sum() / len(data)) * 100

# Display missing value percentages for all columns
print("Missing value percentages for all columns:")
print(missing_percentages)
~~~
{: .python}

~~~
Missing value percentages for all columns:
Date              0.000000
Location          0.000000
MinTemp           1.020899
MaxTemp           0.866905
Rainfall          2.241853
Evaporation      43.166506
Sunshine         48.009762
WindGustDir       7.098859
WindGustSpeed     7.055548
WindDir9am        7.263853
WindDir3pm        2.906641
WindSpeed9am      1.214767
WindSpeed3pm      2.105046
Humidity9am       1.824557
Humidity3pm       3.098446
Pressure9am      10.356799
Pressure3pm      10.331363
Cloud9am         38.421559
Cloud3pm         40.807095
Temp9am           1.214767
Temp3pm           2.481094
RainToday         2.241853
RainTomorrow      2.245978
dtype: float64
~~~
{: .output}

Columns with a high percentage of missing values are likely to be less important unless they have a strong correlation with the target.

- Evaporation (missing 42.82%): Likely less important.
- Sunshine (missing 48.02%): Likely less important.
- Cloud9am (missing 38.42%): Possibly less important.
- Cloud3pm (missing 40.79%): Possibly less important.
~~~
#Visualize missing values
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
~~~
{: .python}


![](../fig/missing.png)


#### Replace Missing Values

To handle missing values, we employ the `SimpleImputer` class from the `sklearn` library. 

~~~
from sklearn.impute import SimpleImputer

# Define numerical and categorical columns
numerical_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
categorical_cols = ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

# Impute numerical columns with mean
num_imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

# Impute categorical columns with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

print(data.head())
~~~
{: .python}




~~~
         Date Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  \
0  01-12-2008   Albury     13.4     22.9       0.6     5.468232  7.611178   
1  02-12-2008   Albury      7.4     25.1       0.0     5.468232  7.611178   
2  03-12-2008   Albury     12.9     25.7       0.0     5.468232  7.611178   
3  04-12-2008   Albury      9.2     28.0       0.0     5.468232  7.611178   
4  05-12-2008   Albury     17.5     32.3       1.0     5.468232  7.611178   

  WindGustDir  WindGustSpeed WindDir9am  ... Humidity9am  Humidity3pm  \
0           W           44.0          W  ...        71.0         22.0   
1         WNW           44.0        NNW  ...        44.0         25.0   
2         WSW           46.0          W  ...        38.0         30.0   
3          NE           24.0         SE  ...        45.0         16.0   
4           W           41.0        ENE  ...        82.0         33.0   

   Pressure9am  Pressure3pm  Cloud9am  Cloud3pm  Temp9am  Temp3pm  RainToday  \
0       1007.7       1007.1  8.000000   4.50993     16.9     21.8         No   
1       1010.6       1007.8  4.447461   4.50993     17.2     24.3         No   
2       1007.6       1008.7  4.447461   2.00000     21.0     23.2         No   
3       1017.6       1012.8  4.447461   4.50993     18.1     26.5         No   
4       1010.8       1006.0  7.000000   8.00000     17.8     29.7         No   

   RainTomorrow  
0            No  
1            No  
2            No  
3            No  
4            No  

[5 rows x 23 columns]
~~~
{: .output}


~~~
data.isnull().sum()
~~~
{: .python}


~~~
Date             0
Location         0
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustDir      0
WindGustSpeed    0
WindDir9am       0
WindDir3pm       0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
RainToday        0
RainTomorrow     0
dtype: int64
~~~
{: .output}


#### Data Encoding
Data encoding is a process of transforming categorical data into a numerical format suitable for analysis by machine learning algorithms. Categorical data consists of discrete labels, such as colors, types, or categories, which are not inherently numerical. Two common encoding techniques are Label Encoding and One-Hot Encoding.

**Label Encoding**:

* Method: In label encoding, each unique category is assigned a unique integer label.
* Example: If we have categories like 'Red,' 'Green,' and 'Blue,' label encoding might assign them labels 0, 1, and 2, respectively.
* Use Case: Label encoding is often used when there is an ordinal relationship between categories, meaning there is a meaningful order or ranking among them.

**One-Hot Encoding**:

* Method: One-hot encoding creates binary columns for each category and indicates the presence or absence of the category with a 1 or 0, respectively.
* Example: For the categories 'Red,' 'Green,' and 'Blue,' one-hot encoding would create three binary columns, each representing one color, with values like [1, 0, 0] for 'Red,' [0, 1, 0] for 'Green,' and [0, 0, 1] for 'Blue.'
* Use Case: One-hot encoding is commonly used when there is no inherent order among categories, and each category is considered equally distinct.

**Considerations**:

* Label encoding might introduce unintended ordinal relationships in the data, which can be problematic for some algorithms.
* One-hot encoding avoids this issue by representing categories independently, but it can lead to a large number of features, especially when dealing with a high number of categories.
* The choice between label encoding and one-hot encoding depends on the nature of the data and the requirements of the machine learning algorithm being used.

**Application**:

* Data encoding is crucial when working with machine learning models that require numerical input, such as linear regression, support vector machines, or neural networks.
* Many machine learning libraries and frameworks provide convenient functions for implementing these encoding techniques.

~~~
from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Encode categoricalencode categorical data into numerical data  columns
for column in categorical_cols:
    data[column] = label_encoder.fit_transform(data[column])

# Print the first few rows to verify encoding
print(data.head())
~~~
{: .python}


~~~
 Date  Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  \
0   105         2     13.4     22.9       0.6     5.468232  7.611178   
1   218         2      7.4     25.1       0.0     5.468232  7.611178   
2   331         2     12.9     25.7       0.0     5.468232  7.611178   
3   444         2      9.2     28.0       0.0     5.468232  7.611178   
4   557         2     17.5     32.3       1.0     5.468232  7.611178   

   WindGustDir  WindGustSpeed  WindDir9am  ...  Humidity9am  Humidity3pm  \
0           13           44.0          13  ...         71.0         22.0   
1           14           44.0           6  ...         44.0         25.0   
2           15           46.0          13  ...         38.0         30.0   
3            4           24.0           9  ...         45.0         16.0   
4           13           41.0           1  ...         82.0         33.0   

   Pressure9am  Pressure3pm  Cloud9am  Cloud3pm  Temp9am  Temp3pm  RainToday  \
0       1007.7       1007.1  8.000000   4.50993     16.9     21.8          0   
1       1010.6       1007.8  4.447461   4.50993     17.2     24.3          0   
2       1007.6       1008.7  4.447461   2.00000     21.0     23.2          0   
3       1017.6       1012.8  4.447461   4.50993     18.1     26.5          0   
4       1010.8       1006.0  7.000000   8.00000     17.8     29.7          0   

   RainTomorrow  
0             0  
1             0  
2             0  
3             0  
4             0  

[5 rows x 23 columns]
~~~
{: .output}

**Correlation with Target**
* Calculate the correlation of numerical features with RainTomorrow. For categorical features, we can use other methods like Chi-square test for independence or converting them to numerical and checking correlation.


~~~
# Calculate correlation of each numerical feature with the target variable
correlation_with_target = data.corr()['RainTomorrow'].sort_values(ascending=False)

# Print correlation values
print(correlation_with_target)
~~~
{: .python}


~~~
RainTomorrow     1.000000
Humidity3pm      0.433179
RainToday        0.305744
Cloud3pm         0.298050
Humidity9am      0.251470
Cloud9am         0.249978
Rainfall         0.233900
WindGustSpeed    0.220442
WindSpeed9am     0.086661
WindSpeed3pm     0.084207
MinTemp          0.082173
WindGustDir      0.048774
WindDir9am       0.035341
WindDir3pm       0.028890
Date             0.005732
Location        -0.005498
Temp9am         -0.025555
Evaporation     -0.088288
MaxTemp         -0.156851
Temp3pm         -0.187806
Pressure3pm     -0.211977
Pressure9am     -0.230975
Sunshine        -0.321533
Name: RainTomorrow, dtype: float64
~~~
{: .output}


~~~
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation with Target Variable (RainTomorrow)')
plt.show()
~~~
{: .python}


![](../fig/heatmap_tomo.png)


~~~
data
~~~
{: .python}

~~~
	Date	Location	MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustDir	WindGustSpeed	WindDir9am	...	Humidity9am	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	RainToday	RainTomorrow
0	105	2	13.4	22.900000	0.6	5.468232	7.611178	13	44.00000	13	...	71.0	22.0	1007.7	1007.1	8.000000	4.50993	16.9	21.8	0	0
1	218	2	7.4	25.100000	0.0	5.468232	7.611178	14	44.00000	6	...	44.0	25.0	1010.6	1007.8	4.447461	4.50993	17.2	24.3	0	0
2	331	2	12.9	25.700000	0.0	5.468232	7.611178	15	46.00000	13	...	38.0	30.0	1007.6	1008.7	4.447461	2.00000	21.0	23.2	0	0
3	444	2	9.2	28.000000	0.0	5.468232	7.611178	4	24.00000	9	...	45.0	16.0	1017.6	1012.8	4.447461	4.50993	18.1	26.5	0	0
4	557	2	17.5	32.300000	1.0	5.468232	7.611178	13	41.00000	1	...	82.0	33.0	1010.8	1006.0	7.000000	8.00000	17.8	29.7	0	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
145455	2317	41	2.8	23.400000	0.0	5.468232	7.611178	0	31.00000	9	...	51.0	24.0	1024.6	1020.3	4.447461	4.50993	10.1	22.4	0	0
145456	2430	41	3.6	25.300000	0.0	5.468232	7.611178	6	22.00000	9	...	56.0	21.0	1023.5	1019.1	4.447461	4.50993	10.9	24.5	0	0
145457	2543	41	5.4	26.900000	0.0	5.468232	7.611178	3	37.00000	9	...	53.0	24.0	1021.0	1016.8	4.447461	4.50993	12.5	26.1	0	0
145458	2656	41	7.8	27.000000	0.0	5.468232	7.611178	9	28.00000	10	...	51.0	24.0	1019.4	1016.5	3.000000	2.00000	15.1	26.0	0	0
145459	2769	41	14.9	23.221348	0.0	5.468232	7.611178	13	40.03523	2	...	62.0	36.0	1020.2	1017.9	8.000000	8.00000	15.0	20.9	0	0
145460 rows × 23 columns
~~~
{: .output}
* Based on the parsentage of the missing value let us drop the following features
~~~
data.drop(['Evaporation', 'Sunshine','Cloud9am','Cloud3pm'], axis=1, inplace=True)
data
~~~
{: .python}

~~~
	Date	Location	MinTemp	MaxTemp	Rainfall	WindGustDir	WindGustSpeed	WindDir9am	WindDir3pm	WindSpeed9am	WindSpeed3pm	Humidity9am	Humidity3pm	Pressure9am	Pressure3pm	Temp9am	Temp3pm	RainToday	RainTomorrow
0	105	2	13.4	22.900000	0.6	13	44.00000	13	14	20.0	24.0	71.0	22.0	1007.7	1007.1	16.9	21.8	0	0
1	218	2	7.4	25.100000	0.0	14	44.00000	6	15	4.0	22.0	44.0	25.0	1010.6	1007.8	17.2	24.3	0	0
2	331	2	12.9	25.700000	0.0	15	46.00000	13	15	19.0	26.0	38.0	30.0	1007.6	1008.7	21.0	23.2	0	0
3	444	2	9.2	28.000000	0.0	4	24.00000	9	0	11.0	9.0	45.0	16.0	1017.6	1012.8	18.1	26.5	0	0
4	557	2	17.5	32.300000	1.0	13	41.00000	1	7	7.0	20.0	82.0	33.0	1010.8	1006.0	17.8	29.7	0	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
145455	2317	41	2.8	23.400000	0.0	0	31.00000	9	1	13.0	11.0	51.0	24.0	1024.6	1020.3	10.1	22.4	0	0
145456	2430	41	3.6	25.300000	0.0	6	22.00000	9	3	13.0	9.0	56.0	21.0	1023.5	1019.1	10.9	24.5	0	0
145457	2543	41	5.4	26.900000	0.0	3	37.00000	9	14	9.0	9.0	53.0	24.0	1021.0	1016.8	12.5	26.1	0	0
145458	2656	41	7.8	27.000000	0.0	9	28.00000	10	3	13.0	7.0	51.0	24.0	1019.4	1016.5	15.1	26.0	0	0
145459	2769	41	14.9	23.221348	0.0	13	40.03523	2	2	17.0	17.0	62.0	36.0	1020.2	1017.9	15.0	20.9	0	0
145460 rows × 19 columns
~~~
{: .output}
* Let us assign the features and target values for further analysis or modeling
~~~
y = data.RainTomorrow.copy()
X = data.drop(['RainTomorrow', 'Date'], axis=1)
~~~
{: .python}
~~~
print(X)
~~~
{: .python}
~~~
        Location  MinTemp    MaxTemp  Rainfall  WindGustDir  WindGustSpeed  \
0              2     13.4  22.900000       0.6           13       44.00000   
1              2      7.4  25.100000       0.0           14       44.00000   
2              2     12.9  25.700000       0.0           15       46.00000   
3              2      9.2  28.000000       0.0            4       24.00000   
4              2     17.5  32.300000       1.0           13       41.00000   
...          ...      ...        ...       ...          ...            ...   
145455        41      2.8  23.400000       0.0            0       31.00000   
145456        41      3.6  25.300000       0.0            6       22.00000   
145457        41      5.4  26.900000       0.0            3       37.00000   
145458        41      7.8  27.000000       0.0            9       28.00000   
145459        41     14.9  23.221348       0.0           13       40.03523   

        WindDir9am  WindDir3pm  WindSpeed9am  WindSpeed3pm  Humidity9am  \
0               13          14          20.0          24.0         71.0   
1                6          15           4.0          22.0         44.0   
2               13          15          19.0          26.0         38.0   
3                9           0          11.0           9.0         45.0   
4                1           7           7.0          20.0         82.0   
...            ...         ...           ...           ...          ...   
145455           9           1          13.0          11.0         51.0   
145456           9           3          13.0           9.0         56.0   
145457           9          14           9.0           9.0         53.0   
145458          10           3          13.0           7.0         51.0   
145459           2           2          17.0          17.0         62.0   

        Humidity3pm  Pressure9am  Pressure3pm  Temp9am  Temp3pm  RainToday  
0              22.0       1007.7       1007.1     16.9     21.8          0  
1              25.0       1010.6       1007.8     17.2     24.3          0  
2              30.0       1007.6       1008.7     21.0     23.2          0  
3              16.0       1017.6       1012.8     18.1     26.5          0  
4              33.0       1010.8       1006.0     17.8     29.7          0  
...             ...          ...          ...      ...      ...        ...  
145455         24.0       1024.6       1020.3     10.1     22.4          0  
145456         21.0       1023.5       1019.1     10.9     24.5          0  
145457         24.0       1021.0       1016.8     12.5     26.1          0  
145458         24.0       1019.4       1016.5     15.1     26.0          0  
145459         36.0       1020.2       1017.9     15.0     20.9          0  

[145460 rows x 17 columns]
~~~
{: .output}


~~~
print(y)
~~~
{: .python}

~~~
0         0
1         0
2         0
3         0
4         0
         ..
145455    0
145456    0
145457    0
145458    0
145459    0
Name: RainTomorrow, Length: 145460, dtype: int32
~~~
{: .output}

### Feature Scaling

Feature scaling is a vital data preprocessing step that aims to normalize data values. Scaling features helps prevent certain data sets from overpowering others, thus avoiding biased model predictions. A common technique used in feature scaling is normalization, which scales data between 0 and 1. However, normalization may not be suitable for features with non-normal distributions. Here are some key aspects of data normalization:

1. **Scale Consistency:**
   - Features in a dataset often have different units and ranges. Normalization ensures that all features are on a consistent scale, facilitating easier comparison and analysis.

2. **Preventing Dominance:**
   - In models employing distance-based metrics (e.g., k-nearest neighbors, clustering algorithms), features with larger scales may dominate the influence, leading to biased results. Normalization helps mitigate this issue.

3. **Algorithm Sensitivity:**
   - Many machine learning algorithms, such as gradient-based optimization methods (e.g., gradient descent), are sensitive to the scale of input features. Normalizing the data aids algorithms in converging faster and performing more reliably.

4. **Standardization vs. Min-Max Scaling:**
   - Two common normalization techniques are standardization and min-max scaling.
      - **Standardization (Z-score normalization):** It transforms data to have a mean of 0 and a standard deviation of 1. The formula is \\( z = \frac{(x - \mu)}{\sigma} \\), where \\( x \\) is the original value, \\( \mu \\) is the mean, and \\( \sigma \\) is the standard deviation.
      - **Min-Max Scaling:** It scales data to a specific range, often between 0 and 1. The formula is \\( x_{\text{scaled}} = \frac{(x - \text{min})}{(\text{max} - \text{min})} \\).

5. **Applicability:**
   - The choice between standardization and min-max scaling depends on the characteristics of the data and the requirements of the algorithm. Standardization is less affected by outliers, making it suitable for robustness against extreme values.

~~~
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)
~~~
{: .python}

~~~
[[-1.53166617e+00  1.89446615e-01 -4.53363105e-02 ... -1.40531282e-02
   1.70232282e-02 -5.29795450e-01]
 [-1.53166617e+00 -7.53100728e-01  2.65043084e-01 ...  3.24642790e-02
   3.81984952e-01 -5.29795450e-01]
 [-1.53166617e+00  1.10901003e-01  3.49692009e-01 ...  6.21684769e-01
   2.21401794e-01 -5.29795450e-01]
 ...
 [ 1.20928479e+00 -1.06728318e+00  5.18989861e-01 ... -6.96308433e-01
   6.44757393e-01 -5.29795450e-01]
 [ 1.20928479e+00 -6.90264238e-01  5.33098015e-01 ... -2.93157571e-01
   6.30158924e-01 -5.29795450e-01]
 [ 1.20928479e+00  4.25083451e-01 -5.01222327e-16 ... -3.08663373e-01
  -1.14362992e-01 -5.29795450e-01]]
~~~
{: .output}


### Splitting Dataset into Training set and Test set

* Splitting a dataset into training and test sets is a fundamental step in machine learning to evaluate the performance of a model. The training set is used to train the model, while the test set is used to evaluate its performance on unseen data. Here's how you can split a dataset into training and test sets in Python:
~~~
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
~~~
{: .python}

~~~
[[-1.60194696e+00  3.62246961e-01 -7.64852179e-01 ... -1.22593745e-01
  -5.18641804e-16  1.88752093e+00]
 [-5.47735053e-01  1.94886832e+00  2.29661730e+00 ...  2.03271279e+00
   2.55715683e+00 -5.29795450e-01]
 [-3.36892671e-01  3.93665206e-01  4.76665398e-01 ...  4.79700813e-02
   4.11181890e-01 -5.29795450e-01]
 ...
 [ 2.95634473e-01 -7.21682483e-01 -2.14634162e-01 ... -5.10238804e-01
  -7.05675855e-02 -5.29795450e-01]
 [-6.18015846e-01 -5.33173014e-01 -1.18809681e+00 ... -4.48215595e-01
  -1.18005123e+00 -5.29795450e-01]
 [ 5.06476854e-01  5.66465552e-01  2.45180700e+00 ...  2.03271279e+00
   1.78343797e+00 -5.29795450e-01]]
~~~
{: .output}
## Training Model
### 1. Logistic Regression
1. **Logistic Regression** is used to model the probability that a given input belongs to a particular class. It's commonly used when the dependent variable is binary (0/1, True/False, Yes/No), but it can also be extended to handle multiclass classification tasks.

2. **How it Works**
Model: Logistic Regression models the relationship between the independent variables (features) and the probability of a binary outcome using the logistic function (sigmoid function).
Sigmoid Function: The sigmoid function maps any real-valued number to the range [0, 1], which makes it suitable for modeling probabilities.
Hypothesis: The logistic regression hypothesis can be represented as:

\\[ h_\theta(x) = g(\theta^T\cdot x) \\]

where: \\( h_\theta(x) \\) is the predicted probability that \\(y = 1 \\) given x. 
g(z) is the sigmoid function, \\( g(z) = \frac{1}{1 + e^{-z}} \\), \\( \theta^T \\)  is the transpose of the parameter vector. \\( x \\)  is the feature vector.

**3. Training Process**

* **Cost Function**: Logistic Regression uses the logistic loss (or cross-entropy loss) as the cost function to penalize incorrect predictions.
* **Optimization**: The model parameters ( \\(\theta \\) ) are learned by minimizing the cost function using optimization algorithms like gradient descent.

**4. Model Interpretation**
* **Coefficients**: The coefficients ( \\(\theta \\) ) learned by Logistic Regression represent the impact of each feature on the predicted probability of the positive class.
* **Interpretation**: Positive coefficients indicate that an increase in the corresponding feature value increases the probability of the positive class, while negative coefficients indicate the opposite.

**5. Applications**
* **Binary Classification**: Predicting whether an email is spam or not, whether a patient has a disease or not.
* **Probability Estimation**: Estimating the probability of a customer buying a product or defaulting on a loan.

~~~
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
~~~
{: .python}

~~~
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
~~~
{: .python}


~~~
y_pred = pd.Series(model.predict(X_test))

y_test = y_test.reset_index(drop=True)
z = pd.concat([y_test, y_pred], axis=1)
z.columns = ['True', 'Prediction']
z.head(50)
~~~
{: .python}


~~~
	True	Prediction
0	1	1
1	0	0
2	0	0
3	0	0
4	0	0
5	0	0
6	0	0
7	1	0
8	0	0
9	1	1
10	0	0
11	0	0
12	0	0
13	0	0
14	0	0
15	1	0
16	1	1
17	0	0
18	1	0
19	0	0
20	0	0
21	0	0
22	0	0
23	1	1
24	0	0
25	0	0
26	0	0
27	1	1
28	1	0
29	0	0
30	0	0
31	0	0
32	0	0
33	0	0
34	0	0
35	1	0
36	0	0
37	1	1
38	0	0
39	0	0
40	0	0
41	1	1
42	0	0
43	0	0
44	0	0
45	0	0
46	0	0
47	1	1
48	1	0
49	0	0
~~~
{: .output}
~~~
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:",f1_score(y_test, y_pred))
~~~
{: .python}

~~~
Accuracy: 0.8386841743434621
Precision: 0.7126323565624231
Recall: 0.45077881619937693
F1-Score: 0.5522373819292052
~~~
{: .output}

~~~
cnf_matrix = confusion_matrix(y_test, y_pred)

labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix', y=1.1)
plt.ylabel('True')
plt.xlabel('Predicted')
~~~
{: .python}


![](../fig/confusion_logistic.png)

~~~
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
#Calculate and plot the ROC AUC curve of the model
y_score = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='{} (AUC = {:.4f})'.format(model.__class__.__name__, roc_auc))

# Plot the ROC AUC curve for the logistic regression model
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve')
plt.legend(loc='lower right')
plt.show();
~~~
{: .python}


![](../fig/ruc_logistic.png)


### Decision Tree

~~~
# Define the model using the Decision Tree Classifier with the default hyperparameters
from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier()
# Try to optimize the model hyperparameters that best fit the training data
model2.fit(X_train, y_train)
~~~
{: .python}

~~~
  DecisionTreeClassifier
DecisionTreeClassifier()
~~~
{: .output}

~~~
# model prediction
y_pred2 = model2.predict(X_test)
pred_train = model2.predict(X_train)
~~~
{: .python}

~~~
Compare performance between test and train data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2))
print("Recall:", recall_score(y_test, y_pred2))
print("F1-Score:",f1_score(y_test, y_pred2))
~~~
{: .python}
~~~
Accuracy: 0.7795957651588066
Precision: 0.5006045949214026
Recall: 0.5158878504672897
F1-Score: 0.5081313286284136
~~~
{: .output}

~~~
cnf_matrix = confusion_matrix(y_test, y_pred2)

labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix', y=1.1)
plt.ylabel('True')
plt.xlabel('Predicted')
~~~
{: .python}

![](../fig/confusion_DT.png)

~~~
#Calculate and plot the ROC AUC curve of the model
y_score = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='{} (AUC = {:.4f})'.format(model.__class__.__name__, roc_auc))

# Plot the ROC AUC curve for the logistic regression model
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve')
plt.legend(loc='lower right')
plt.show();
~~~
{: .python}

![](../fig/ruc_DT.png)


### Random Forest


~~~
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)
~~~
{: .python}

~~~
# Compare performance between test and train data
y_pred_rf = classifier.predict(X_test)
pred_train = classifier.predict(X_train)
~~~
{: .python}


~~~
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-Score:",f1_score(y_test, y_pred_rf))
~~~
{: .python}

~~~
Accuracy: 0.8500274989687887
Precision: 0.7472950228420293
Recall: 0.48411214953271026
F1-Score: 0.5875791662728046
~~~
{: .output}


> ## Exercise: Support vector machine 
> - Write a code  to implement a Support Vector Machine (SVM) in Python using scikit-learn.
> 
> > ## Solution
> > ~~~
> > from sklearn.svm import SVC
>> from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
> > classifier2 = SVC(kernel="linear",probability=True) 
>> classifier2.fit(X_train, y_train) 
>> y_pred2 = classifier2.predict(X_test)
>> pred_train = classifier2.predict(X_train)
>> print("Accuracy:", accuracy_score(y_test, y_pred2))
>> print("Precision:", precision_score(y_test, y_pred2))
>> print("Recall:", recall_score(y_test, y_pred2))
>> print("F1-Score:",f1_score(y_test, y_pred2))
> > ~~~
> > {: .python}
> > ~~~
> > Accuracy: 0.7795957651588066
>> Precision: 0.5006045949214026
>> Recall: 0.5158878504672897
>> F1-Score: 0.5081313286284136
> > ~~~
> > {: .output}
> {: .solution}
{: .challenge}


### Ensemble Learning Techniques

### Ensembele Learning

* Ensemble Learning is a powerful machine learning paradigm where multiple models (often called "weak learners") are trained and combined to solve a particular problem. The idea is that combining the predictions of multiple models can often produce better results than any single model alone. Here are key points and concepts related to Ensemble Learning:

**Key Concepts**
_Weak Learners_:

- Individual models that are often simple and may not perform well on their own.
- Examples include decision trees, linear classifiers, and other base algorithms.
_Strong Learner_:

- The combined model created from the ensemble of weak learners.
- The strong learner often has better generalization and predictive performance.

**Types of Ensemble Methods**

1. Bagging (Bootstrap Aggregating):

- Involves training multiple models on different random subsets of the training data.
- Each model in the ensemble is trained independently.
- The final prediction is made by averaging (for regression) or majority voting (for classification).
- Example: Random Forests.

2. Boosting:

- Trains models sequentially, with each model trying to correct the errors of the previous one.
- Models focus more on instances that were previously misclassified.
- The final model is a weighted sum of the individual models.
- Examples: AdaBoost, Gradient Boosting Machines (GBM), XGBoost, LightGBM.

3. Stacking (Stacked Generalization):

- Combines the predictions of multiple base models using a "meta-model".
- Base models are trained on the training data, and their predictions are used as input features for the meta-model.
- The meta-model learns to predict the final output based on these predictions.

**Advantages of Ensemble Learning**

* **Improved Accuracy**: Ensemble methods often achieve higher accuracy and robustness compared to individual models.
* **Reduction of Overfitting**: By combining multiple models, ensembles can reduce the risk of overfitting to the training data.
* **Versatility**: Ensemble methods can be applied to a wide variety of algorithms and problems.

**Common Ensemble Algorithms**

-1. Random Forests:

* An ensemble of decision trees using bagging.
* Often used for both classification and regression tasks.
* Reduces variance by averaging multiple trees.

-2. AdaBoost:

* Sequentially builds an ensemble by training each new model to correct the errors of the previous models.
* Works well with weak learners like decision stumps.

-3. Gradient Boosting Machines (GBM):

* An extension of boosting that optimizes a loss function by sequentially adding models to minimize errors.
* Popular implementations include XGBoost, LightGBM, and CatBoost.

-4. XGBoost:

* XGBoost is a highly efficient and scalable implementation of gradient boosting.
* It is specifically optimized for speed and performance, making it suitable for large datasets and complex problems.
* XGBoost includes several enhancements over traditional gradient boosting, such as regularization techniques and parallelization.
* It can handle a variety of data types and is commonly used for both classification and regression tasks.
* XGBoost is known for its exceptional performance in machine learning competitions and real-world applications.

#### Bagging Classifiers

~~~
from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier(n_estimators=20)
bag_clf.fit(X_train, y_train)
~~~
{: .python}

~~~
bag_clf_pred_test = bag_clf.predict(X_test)
bag_clf_pred_train = bag_clf.predict(X_train)
~~~
{: .python}

~~~
print(accuracy_score(y_test, bag_clf_pred_test))
print(accuracy_score(y_train, bag_clf_pred_train))
~~~
{: .python}

~~~
0.8424309088409184
0.9959267152481782
~~~
{: .output}

~~~
print("Accuracy:", accuracy_score(y_test, bag_clf_pred_test))
print("Precision:", precision_score(y_test, bag_clf_pred_test))
print("Recall:", recall_score(y_test, bag_clf_pred_test))
print("F1-Score:",f1_score(y_test, bag_clf_pred_test))
~~~
{: .python}

~~~
Accuracy: 0.8424309088409184
Precision: 0.7141857209519366
Recall: 0.47679127725856696
F1-Score: 0.5718288810013077
~~~
{: .output}


> ## Exercise: Bagging with SVC base estimator
> - Write a code  to implement a Bagging with a Support Vector Classifier (SVC) base estimator in Python using scikit-learn.
> 
> > ## Solution
> > ~~~
> > from sklearn.svm import SVC
>> from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
> > from sklearn.ensemble import BaggingClassifier
> > bag_clf2 = BaggingClassifier(SVC(),random_state=42)
> > bag_clf2.fit(X_train, y_train)
> > bag_clf_pred_test2 = bag_clf2.predict(X_test)
> > bag_clf_pred_train2 = bag_clf2.predict(X_train)
> > print(accuracy_score(y_test, bag_clf_pred_test2))
> > print(accuracy_score(y_train, bag_clf_pred_train2))
> > print("Accuracy:", accuracy_score(y_test, bag_clf_pred_test2))
> > print("Precision:", precision_score(y_test, bag_clf_pred_test2))
> > print("Recall:", recall_score(y_test, bag_clf_pred_test2))
> > print("F1-Score:",f1_score(y_test, bag_clf_pred_test2))
> > ~~~
> > {: .python}
> > ~~~
> > Accuracy: 0.7795957651588066
>> Precision: 0.5006045949214026
>> Recall: 0.5158878504672897
>> F1-Score: 0.5081313286284136
> > ~~~
> > {: .output}
> {: .solution}
{: .challenge}


#### Boosting classifier

#### 1. AdaBoost

~~~
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(n_estimators=20)
ada_clf.fit(X_train, y_train)
~~~
{: .python}

~~~
ada_clf_pred_test = ada_clf.predict(X_test)
ada_clf_pred_train = ada_clf.predict(X_train)

print(accuracy_score(y_test, ada_clf_pred_test))
print(accuracy_score(y_train, ada_clf_pred_train))
~~~
{: .python}


~~~
0.8360717723085385
0.8397067922452908
~~~
{: .output}

~~~
print("Accuracy:", accuracy_score(y_test, ada_clf_pred_test))
print("Precision:", precision_score(y_test, ada_clf_pred_test))
print("Recall:", recall_score(y_test, ada_clf_pred_test))
print("F1-Score:",f1_score(y_test, ada_clf_pred_test))
~~~
{: .python}


~~~
Accuracy: 0.8360717723085385
Precision: 0.7078821455552757
Recall: 0.4378504672897196
F1-Score: 0.5410451352131652
~~~
{: .output}

#### 2. Gradient Boosting

~~~
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(n_estimators=20)
gb_clf.fit(X_train, y_train)
~~~
{: .python}

~~~
gb_clf_pred_test = gb_clf.predict(X_test)
gb_clf_pred_train = gb_clf.predict(X_train)
print(accuracy_score(y_test, gb_clf_pred_test))
print(accuracy_score(y_train, gb_clf_pred_train))
~~~
{: .python}

~~~
0.8349718135569916
0.8398700673724735
~~~
{: .output}

~~~
print("Accuracy:", accuracy_score(y_test, gb_clf_pred_test))
print("Precision:", precision_score(y_test, gb_clf_pred_test))
print("Recall:", recall_score(y_test, gb_clf_pred_test))
print("F1-Score:",f1_score(y_test, gb_clf_pred_test))
~~~
{: .python}

~~~
Accuracy: 0.8349718135569916
Precision: 0.7557661927330174
Recall: 0.37258566978193147
F1-Score: 0.499113197704747
~~~
{: .output}

#### 3. XGBoost

~~~
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(n_estimators=20)
xgb_clf.fit(X_train, y_train)
~~~
{: .python}


~~~
xgb_clf_pred_test = xgb_clf.predict(X_test)
xgb_clf_pred_train = xgb_clf.predict(X_train)
print(accuracy_score(y_test, xgb_clf_pred_test))
print(accuracy_score(y_train, xgb_clf_pred_train))
~~~
{: .python}

~~~
0.8483431871304826
0.8596865117558091
~~~
{: .output}


~~~
print("Accuracy:", accuracy_score(y_test, xgb_clf_pred_test))
print("Precision:", precision_score(y_test, xgb_clf_pred_test))
print("Recall:", recall_score(y_test, xgb_clf_pred_test))
print("F1-Score:",f1_score(y_test, xgb_clf_pred_test))
~~~
{: .python}

~~~
Accuracy: 0.8483431871304826
Precision: 0.7397325692454633
Recall: 0.48255451713395636
F1-Score: 0.5840874811463047
~~~
{: .output}

#### Stacking

~~~
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
~~~
{: .python}

~~~
# Define base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=10, random_state=42))
]
# Create stacking classifier
stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
~~~
{: .python}


~~~
# Train stacking classifier
stack_clf.fit(X_train, y_train)

# Make predictions
y_pred = stack_clf.predict(X_test)
~~~
{: .python}

~~~
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:",f1_score(y_test, y_pred))
~~~
{: .python}

~~~
Accuracy: 0.8449401897428847
Precision: 0.7227537922987165
Recall: 0.482398753894081
F1-Score: 0.5786081270434377
~~~
{: .output}


 ### Comparison of different ML Algorithms
 
~~~
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Assuming X and y are defined previously
# X = ...
# y = ...

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a list of classification models to evaluate
models = [
    LogisticRegression(max_iter=8000),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    XGBClassifier(n_estimators=10),
    GradientBoostingClassifier(n_estimators=10),
    AdaBoostClassifier(n_estimators=10),
    StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=10)),
            ('gb', GradientBoostingClassifier(n_estimators=10)),
            ('xgb', XGBClassifier(n_estimators=10))
        ],
        final_estimator=LogisticRegression()
    )
]

# Train and evaluate each model
for model in models:
    # Train the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate and print the classification report of the model
    report = classification_report(y_test, y_pred)
    print(model.__class__.__name__)
    print(report)

    # Check if the model has the predict_proba method
    if hasattr(model, "predict_proba"):
        # Calculate and plot the ROC AUC curve of the model
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='{} (AUC = {:.4f})'.format(model.__class__.__name__, roc_auc))
    
    print('-----------------------------------')

# Plot the ROC AUC curves for all models
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve')
plt.legend(loc='lower right')
plt.show()
~~~
{: .python}

~~~
LogisticRegression
              precision    recall  f1-score   support

           0       0.86      0.95      0.90     22672
           1       0.71      0.45      0.55      6420

    accuracy                           0.84     29092
   macro avg       0.79      0.70      0.73     29092
weighted avg       0.83      0.84      0.82     29092

-----------------------------------
DecisionTreeClassifier
              precision    recall  f1-score   support

           0       0.86      0.85      0.86     22672
           1       0.50      0.52      0.51      6420

    accuracy                           0.78     29092
   macro avg       0.68      0.68      0.68     29092
weighted avg       0.78      0.78      0.78     29092

-----------------------------------
RandomForestClassifier
              precision    recall  f1-score   support

           0       0.87      0.96      0.91     22672
           1       0.75      0.48      0.59      6420

    accuracy                           0.85     29092
   macro avg       0.81      0.72      0.75     29092
weighted avg       0.84      0.85      0.84     29092

-----------------------------------
XGBClassifier
              precision    recall  f1-score   support

           0       0.86      0.95      0.90     22672
           1       0.73      0.45      0.56      6420

    accuracy                           0.84     29092
   macro avg       0.80      0.70      0.73     29092
weighted avg       0.83      0.84      0.83     29092

-----------------------------------
GradientBoostingClassifier
              precision    recall  f1-score   support

           0       0.83      0.98      0.90     22672
           1       0.81      0.28      0.41      6420

    accuracy                           0.83     29092
   macro avg       0.82      0.63      0.65     29092
weighted avg       0.82      0.83      0.79     29092

-----------------------------------
C:\Users\Yadi Milki Wabi\anaconda3\Lib\site-packages\sklearn\ensemble\_weight_boosting.py:519: FutureWarning: The SAMME.R algorithm (the default) is deprecated and will be removed in 1.6. Use the SAMME algorithm to circumvent this warning.
  warnings.warn(
AdaBoostClassifier
              precision    recall  f1-score   support

           0       0.86      0.94      0.90     22672
           1       0.68      0.44      0.53      6420

    accuracy                           0.83     29092
   macro avg       0.77      0.69      0.71     29092
weighted avg       0.82      0.83      0.82     29092

-----------------------------------
StackingClassifier
              precision    recall  f1-score   support

           0       0.87      0.95      0.91     22672
           1       0.73      0.48      0.58      6420

    accuracy                           0.85     29092
   macro avg       0.80      0.72      0.74     29092
weighted avg       0.84      0.85      0.83     29092

-----------------------------------

~~~
{: .output}

![](../fig/ruc_comparison1.png)


#### Check for Data imbalance

~~~
import pandas as pd
import matplotlib.pyplot as plt

# Assuming y is a pandas Series or a numpy array
# If y is a numpy array, convert it to pandas Series
if isinstance(y, np.ndarray):
    y = pd.Series(y)

# Count the occurrences of each class in the target column
class_counts = y.value_counts()

# Plot the counts using a bar graph
class_counts.plot(kind='bar', color=['blue', 'orange'])

# Add titles and labels
plt.title('Class Distribution in Target Column')
plt.xlabel('Class')
plt.ylabel('Count')

# Show the plot
plt.show()
~~~
{: .python}

![](../fig/bar_imbalanced.png)


~~~
###### Or in pei chart
import pandas as pd
import matplotlib.pyplot as plt

# Assuming y is a pandas Series or a numpy array
# If y is a numpy array, convert it to pandas Series
if isinstance(y, np.ndarray):
    y = pd.Series(y)

# Count the occurrences of each class in the target column
class_counts = y.value_counts()

# Plot the counts using a pie chart
class_counts.plot(kind='pie', autopct='%1.1f%%', colors=['blue', 'orange'], labels=['Class 0', 'Class 1'], startangle=90)

# Add a title
plt.title('Class Distribution in Target Column')

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')

# Show the plot
plt.show()
~~~
{: .python}


![](../fig/pie_imbalaced.png)


##### Balanced distribution

~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data to balance the classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Count the occurrences of each class in the resampled target column
class_counts = pd.Series(y_train_res).value_counts()

# Plot the counts using a pie chart
class_counts.plot(kind='pie', autopct='%1.1f%%', colors=['blue', 'orange'], labels=['Class 0', 'Class 1'], startangle=90)

# Add a title
plt.title('Balanced Class Distribution in Target Column After SMOTE')

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')

# Show the plot
plt.show()
~~~
{: .python}


![](../fig/balanced.png)

~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

# Assuming X and y are defined previously
# X = ...
# y = ...

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data to balance the classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Create a list of classification models to evaluate
models = [
    LogisticRegression(max_iter=8000),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    XGBClassifier(n_estimators=10),
    GradientBoostingClassifier(n_estimators=10),
    AdaBoostClassifier(n_estimators=10),
    StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=10)),
            ('gb', GradientBoostingClassifier(n_estimators=10)),
            ('xgb', XGBClassifier(n_estimators=10))
        ],
        final_estimator=LogisticRegression()
    )
]

# Train and evaluate each model
for model in models:
    # Train the model on the resampled training set
    model.fit(X_train_res, y_train_res)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate and print the classification report of the model
    report = classification_report(y_test, y_pred)
    print(model.__class__.__name__)
    print(report)

    # Check if the model has the predict_proba method
    if hasattr(model, "predict_proba"):
        # Calculate and plot the ROC AUC curve of the model
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='{} (AUC = {:.4f})'.format(model.__class__.__name__, roc_auc))
    
    print('-----------------------------------')

# Plot the ROC AUC curves for all models
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve')
plt.legend(loc='lower right')
plt.show()

~~~
{: .pytthon}



~~~
LogisticRegression
              precision    recall  f1-score   support

           0       0.92      0.79      0.85     22672
           1       0.50      0.75      0.60      6420

    accuracy                           0.78     29092
   macro avg       0.71      0.77      0.72     29092
weighted avg       0.82      0.78      0.79     29092

-----------------------------------
DecisionTreeClassifier
              precision    recall  f1-score   support

           0       0.87      0.83      0.85     22672
           1       0.47      0.55      0.51      6420

    accuracy                           0.76     29092
   macro avg       0.67      0.69      0.68     29092
weighted avg       0.78      0.76      0.77     29092

-----------------------------------
RandomForestClassifier
              precision    recall  f1-score   support

           0       0.89      0.90      0.90     22672
           1       0.65      0.62      0.63      6420

    accuracy                           0.84     29092
   macro avg       0.77      0.76      0.77     29092
weighted avg       0.84      0.84      0.84     29092

-----------------------------------
XGBClassifier
              precision    recall  f1-score   support

           0       0.91      0.84      0.87     22672
           1       0.55      0.69      0.61      6420

    accuracy                           0.81     29092
   macro avg       0.73      0.77      0.74     29092
weighted avg       0.83      0.81      0.81     29092

-----------------------------------
GradientBoostingClassifier
              precision    recall  f1-score   support

           0       0.90      0.80      0.85     22672
           1       0.50      0.70      0.58      6420

    accuracy                           0.78     29092
   macro avg       0.70      0.75      0.72     29092
weighted avg       0.81      0.78      0.79     29092

-----------------------------------
AdaBoostClassifier
              precision    recall  f1-score   support

           0       0.90      0.82      0.86     22672
           1       0.52      0.68      0.59      6420

    accuracy                           0.79     29092
   macro avg       0.71      0.75      0.72     29092
weighted avg       0.82      0.79      0.80     29092

-----------------------------------
StackingClassifier
              precision    recall  f1-score   support

           0       0.89      0.89      0.89     22672
           1       0.60      0.61      0.61      6420

    accuracy                           0.83     29092
   macro avg       0.75      0.75      0.75     29092
weighted avg       0.83      0.83      0.83     29092

-----------------------------------

~~~
{: .output}



![](../fig/ruc_balanced.png)




## Unsupervised Learning



~~~

~~~
{: .python}

~~~

~~~
{: .output}







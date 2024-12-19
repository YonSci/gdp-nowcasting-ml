---
title: Machine Learning Algorithms
teaching: 130
exercises: 130
questions:
- "How much of the data is missing? Is it a small fraction or a significant portion?"

objectives:
- "Learn the difference between deleting incomplete observations and imputing missing values."


keypoints:
- "Deletion: Simple but risks losing large amounts of data and introducing bias."

---


# Machine Learning Algorithms

## Linear Models

### Linear Regression

#### Assumptions:
- **Linearity**: The relationship between `predictors` and the `target` is linear.
- **Independence of errors**: Residuals should be independent of each other.
- **Normality of residuals** (for inference).
- **Homoscedasticity**: variance of the residuals is constant across all levels of the predictor variable(s)

#### How It Works:
- Finds a linear combination of input features to predict the target value. Minimizes the sum of squared residuals.

**Pros:**
- Simple and interpretable.
- Fast to train and widely understood.

**Cons:**
- Sensitive to outliers.
- Might underfit if the relationship is non-linear.

#### Install Machine Learning Frameworks and Libraries

```python
pip install xgboost
pip install lightgbm
pip install sklearn
pip install -U scikit-learn
```

#### Importing Libraries

```python

# Import the NumPy library for numerical operations
import numpy as np

# Import the Pandas library for data manipulation and analysis
import pandas as pd

# Import Matplotlib for plotting and visualization
import matplotlib.pyplot as plt

# Import train_test_split from scikit-learn for splitting data into training and testing sets
from sklearn.model_selection import train_test_split

# Import various regression models from scikit-learn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet  # Linear models
from sklearn.tree import DecisionTreeRegressor  # Decision tree model
from sklearn.neighbors import KNeighborsRegressor  # K-nearest neighbors model
from sklearn.svm import SVR  # Support vector regression model

# Import ensemble models from scikit-learn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor

# Import XGBoost library for gradient boosting
import xgboost as xgb

# Import LightGBM library for gradient boosting
import lightgbm as lgb

# Import r2_score from scikit-learn for evaluating model performance
from sklearn.metrics import r2_score

# Import StandardScaler from scikit-learn for feature scaling
from sklearn.preprocessing import StandardScaler
```

#### Load the data from the csv file
```python
# Load the data from the csv file
data = pd.read_csv('gdp_data.csv', index_col='Year')
data.head()
```

#### Plot the timeseries of the data
```python
# Create a function to plot the time series
def plot_time_series(column):
    plt.figure(figsize=(10, 6))
    data[column].plot()
    plt.title(f'{column}')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Create a dropdown widget for selecting the column
column_selector = widgets.Dropdown(
    options=data.columns,
    description='Column:',
    disabled=False,
)

# Link the dropdown widget to the plot_time_series function
interactive_plot = widgets.interactive_output(plot_time_series, {'column': column_selector})

# Display the widget and the interactive plot
display(column_selector, interactive_plot)
```


#### Split data
```python
X = data.drop("gdp", axis=1)
y = data["gdp"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
```

#### Define and Train the Linear Regression Model
```python
lr = LinearRegression()
lr.fit(X_train, y_train)
```

#### Make predictions
```python
predictions_lr = lr.predict(X_test)
predictions_lr
```

#### Evaluate the model

```python
r2_lr = r2_score(y_test, predictions_lr)
print(f"  R^2: {r2_lr:.4f}")
```

#### Plot the predictions
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.gdp, label='Actual GDP', marker='o', color='blue')
plt.plot(y_test.index, predictions_lr, label='Predicted GDP', linestyle='--', color='red', marker='o')
plt.title('GDP Prediction with Linear Regression')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
### Ridge Regression

#### Assumptions:
- Similar to linear regression, with an added assumption that parameter weights should be small.

#### How It Works:
- Adds an L2 penalty (sum of squared coefficients) to the cost function to shrink coefficients and reduce variance.

**Pros:**
- Reduces model complexity and prevents overfitting.
- Handles multicollinearity well.

**Cons:**
- Still assumes a linear relationship.
- Coefficients are shrunk but not set to zero.

#### Define and train the Ridge Regression Model

```python
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
```

#### Make predictions
```python
predictions_ridge = ridge.predict(X_test)
predictions_ridge
```

#### Evaluate the model

```python
r2_ridge = r2_score(y_test, predictions_ridge)
print(f"  R^2: {r2_ridge:.4f}")
```

#### Plot the predictions
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.gdp, label='Actual GDP', marker='o', color='blue')
plt.plot(y_test.index, predictions_ridge, label='Predicted GDP', linestyle='--', color='red', marker='o')
plt.title('GDP Prediction with Ridge Regression')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


### Lasso Regression

#### Assumptions:
- Similar to linear regression, but encourages sparsity in coefficients.

#### How It Works:
- Adds an L1 penalty (absolute sum of coefficients) to the cost function, which can set some coefficients to exactly zero.

**Pros:**
- Performs feature selection by eliminating irrelevant features.

**Cons:**
- Can be unstable for some datasets (feature selection might be too aggressive).

#### Define and train Lasso Regression Model

```python
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
```

#### Make predictions
```python
predictions_lasso = lasso.predict(X_test)
predictions_lasso
```

#### Evaluate the model

```python
r2_lasso = r2_score(y_test, predictions_lasso)
print(f"  R^2: {r2_lasso =:.4f}")
```

#### Plot the predictions
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.gdp, label='Actual GDP', marker='o', color='blue')
plt.plot(y_test.index, predictions_lasso, label='Predicted GDP', linestyle='--', color='red', marker='o')
plt.title('GDP Prediction with Lasso Regression')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


### Elastic Net Regression

#### Assumptions:
- Similar to linear regression, combines L1 and L2 penalties.

#### How It Works:
- Balances between Ridge and Lasso.
- Encourages both coefficient shrinkage and sparsity.

**Pros:**
- More flexible than Ridge or Lasso alone.
- Good for high-dimensional data.

**Cons:**
- Additional hyperparameters to tune (ratio of L1 vs L2).


#### Define and train ElasticNet Regression Model

```python
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)
```

#### Make predictions
```python
predictions_elastic = elastic.predict(X_test)
predictions_elastic
```

#### Evaluate the model

```python
r2_elastic = r2_score(y_test, predictions_elastic)
print(f"  R^2: {r2_elastic:.4f}")
```

#### Plot the predictions
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.gdp, label='Actual GDP', marker='o', color='blue')
plt.plot(y_test.index, predictions_elastic, label='Predicted GDP', linestyle='--', color='red', marker='o')
plt.title('GDP Prediction with Elastic Net Regression')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


---

## Tree-Based Models

### Decision Tree Regressor

#### Assumptions:
- Non-parametric model, no strict assumptions about data distribution.

#### How It Works:
- Splits data into subsets based on feature values, creating a tree structure that ends in leaf nodes predicting average values.

**Pros:**
- Easy to interpret.
- Can capture non-linear relationships.

**Cons:**
- Prone to overfitting if not regularized.
- Unstable to small variations in data.

#### Define and Train Decision Tree Regressor
```python
dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
```

#### Make predictions
```python
predictions_dt = dt.predict(X_test)
predictions_dt
```

#### Evaluate the model

```python
r2_dt = r2_score(y_test, predictions_dt)
print(f"  R^2: {r2_dt:.4f}")
```

#### Plot the predictions
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.gdp, label='Actual GDP', marker='o', color='blue')
plt.plot(y_test.index, predictions_dt, label='Predicted GDP', linestyle='--', color='red', marker='o')
plt.title('GDP Prediction with Decision Tree Regressor')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Instance-Based Learning

### K-Nearest Neighbors (KNN) Regressor

#### Assumptions:
- Similar points exist in local neighborhoods.

#### How It Works:
- Predicts target by averaging the values of the nearest k data points.

**Pros:**
- Simple concept, no explicit training process.
- Can model complex, non-linear relationships.

**Cons:**
- Sensitive to the scale of features and choice of k.
- Computationally expensive at prediction time.


#### Define and Train K-Nearest Neighbors (KNN) Regressor
```python
# Step 1: Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 2: Train the KNN model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
```

#### Make predictions
```python
# Step 3: Standardize the test data
X_test_scaled = scaler.transform(X_test)

# Step 4: Make predictions
predictions_knn = knn.predict(X_test_scaled)
predictions_knn
```

#### Evaluate the model

```python
r2_knn = r2_score(y_test, predictions_knn)
print(f"  R^2: {r2_knn:.4f}")
```

#### Plot the predictions
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.gdp, label='Actual GDP', marker='o', color='blue')
plt.plot(y_test.index, predictions_knn, label='Predicted GDP', linestyle='--', color='red', marker='o')
plt.title('GDP Prediction with K-Nearest Neighbors (KNN) Regressor')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Support Vector Regressor (SVR)

#### Assumptions:
- Data can be separated (in a high-dimensional space) by a hyperplane with minimum error.

#### How It Works:
- Fits a regression line that tries to fit as many points within a certain epsilon margin. Uses kernel tricks for non-linear data.

**Pros:**
- Robust to outliers.
- Handles non-linearities with kernel functions.

**Cons:**
- Parameter tuning (C, epsilon, kernel parameters) can be tricky.
- Scalability can be an issue with large datasets.


#### Define and Train Support Vector Regressor (SVR)
```python
# Step 1: Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 2: Train the SVR model
svr = SVR(C=1.0, epsilon=0.1, kernel='rbf')
svr.fit(X_train_scaled, y_train)
```

#### Make predictions
```python
# Step 3: Standardize the test data
X_test_scaled = scaler.transform(X_test)

# Step 4: Make predictions
predictions_svr = svr.predict(X_test_scaled)
predictions_svr
```

#### Evaluate the model

```python
r2_svr = r2_score(y_test, predictions_svr)
print(f"  R^2: {r2_svr:.4f}")
```

#### Plot the predictions
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.gdp, label='Actual GDP', marker='o', color='blue')
plt.plot(y_test.index, predictions_svr, label='Predicted GDP', linestyle='--', color='red', marker='o')
plt.title('GDP Prediction with Support Vector Regressor (SVR)')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Boosting Methods

### Gradient Boosting Machines (GBM)

#### Assumptions:
- Uses decision trees as weak learners combined in a sequential manner.

#### How It Works:
- Iteratively adds new models that correct errors of previous ensembles, using gradient descent on the loss function.

**Pros:**
- Often high accuracy.
- Can handle complex relationships and interactions.

**Cons:**
- More complex to tune.
- Can overfit without proper regularization.

#### Define and Train Gradient Boosting Machines
```python
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)
```

#### Make predictions
```python
# Make predictions
predictions_gbr = gbr.predict(X_test)
predictions_gbr
```

#### Evaluate the model

```python
r2_gbr = r2_score(y_test, predictions_gbr)
print(f"  R^2: {r2_gbr:.4f}")
```

#### Plot the predictions
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.gdp, label='Actual GDP', marker='o', color='blue')
plt.plot(y_test.index, predictions_gbr, label='Predicted GDP', linestyle='--', color='red', marker='o')
plt.title('GDP Prediction with Gradient Boosting Machines')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

### XGBoost (Extreme Gradient Boosting)

#### Assumptions:
- Similar to GBM but with additional optimization and regularization.

#### How It Works:
- Optimized tree-based boosting algorithm with regularization for better generalization and faster training.

**Pros:**
- Very efficient and often achieves state-of-the-art performance.
- Handles missing values and can be parallelized easily.

**Cons:**
- Parameter tuning can still be complex.
- Slightly more complex implementation.

#### Define and Train Extreme Gradient Boosting
```python
xgbr = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgbr.fit(X_train, y_train)
```

#### Make predictions
```python
# Make predictions
predictions_xgbr = xgbr.predict(X_test_scaled)
predictions_xgbr
```

#### Evaluate the model

```python
r2_xgbr = r2_score(y_test, predictions_xgbr)
print(f"  R^2: {r2_xgbr:.4f}")
```

#### Plot the predictions
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.gdp, label='Actual GDP', marker='o', color='blue')
plt.plot(y_test.index, predictions_xgbr, label='Predicted GDP', linestyle='--', color='red', marker='o')
plt.title('GDP Prediction with Extreme Gradient Boosting')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


### LightGBM (Light Gradient Boosting Machine)

#### Assumptions:
- Similar to GBM but builds trees leaf-wise and uses histogram-based methods.

#### How It Works:
- More memory and computationally efficient than traditional GBM.

**Pros:**
- Very fast and scalable.
- Excellent accuracy on large datasets.

**Cons:**
- Might not perform well on very small datasets.
- Still requires careful hyperparameter tuning.



#### Define and Train Light Gradient Boosting Machine
```python
lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
lgb.fit(X_train, y_train)
```

#### Make predictions
```python
# Make predictions
predictions_lgb = lgb.predict(X_test)
predictions_lgb
```

#### Evaluate the model

```python
r2_lgb = r2_score(y_test, predictions_lgb)
print(f"  R^2: {r2_lgb:.4f}")
```

#### Plot the predictions
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.gdp, label='Actual GDP', marker='o', color='blue')
plt.plot(y_test.index, predictions_lgb, label='Predicted GDP', linestyle='--', color='red', marker='o')
plt.title('GDP Prediction with Light Gradient Boosting Machine')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Bagging

### Random Forest Regressor

#### Assumptions:
- Individual decision trees do not have strong assumptions.

#### How It Works:
- Builds many trees on bootstrap samples and averages their predictions.
- Uses random feature subsets for further diversity.

**Pros:**
- Reduces variance drastically compared to a single decision tree.
- Generally robust and accurate.

**Cons:**
- Less interpretable than a single decision tree.
- Can be computationally expensive.

#### Define and Train Random Forest Regressor
```python
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
```

#### Make predictions
```python
# Make predictions
predictions_rf = rf.predict(X_test)
predictions_rf
```

#### Evaluate the model

```python
r2_rf = r2_score(y_test, predictions_rf)
print(f"  R^2: {r2_rf:.4f}")
```

#### Plot the predictions
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.gdp, label='Actual GDP', marker='o', color='blue')
plt.plot(y_test.index, predictions_rf, label='Predicted GDP', linestyle='--', color='red', marker='o')
plt.title('GDP Prediction with Random Forest Regressor')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Stacking (Stacked Regression)

#### Assumptions:
- Combines multiple different models’ predictions as features for a final meta-model.

#### How It Works:
- Trains multiple base learners and then trains a meta-learner on their combined predictions to improve overall performance.

**Pros:**
- Can often improve predictive performance beyond any single model.

**Cons:**
- More complex workflow.
- Risk of overfitting if not done carefully.

#### Define and Train Stacked Regression
```python
base_models = [
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
    ('gbr', GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42))
]

stack = StackingRegressor(estimators=base_models, final_estimator=Ridge())
stack.fit(X_train, y_train)
```

#### Make predictions
```python
# Make predictions
predictions_stack = stack.predict(X_test)
predictions_stack
```

#### Evaluate the model

```python
r2_stack = r2_score(y_test, predictions_stack)
print(f"  R^2: {r2_stack:.4f}")
```

#### Plot the predictions
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.gdp, label='Actual GDP', marker='o', color='blue')
plt.plot(y_test.index, predictions_stack, label='Predicted GDP', linestyle='--', color='red', marker='o')
plt.title('GDP Prediction with Stacked Regression')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

### Machine Learning Model Comparison

```python
r2_scores = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression',
              'Decision Tree Regressor', 'K-Nearest Neighbors Regressor', 'Support Vector Regressor',
              'Gradient Boosting Machines', 'XGBoost Regressor', 'Light Gradient Boosting Machine',
              'Random Forest Regressor', 'Stacked Regression'],
    'R^2 Score': [r2_lr, r2_ridge, r2_lasso, r2_elastic, r2_dt, r2_knn, r2_svr, r2_gbr, r2_xgbr, r2_lgb, r2_rf, r2_stack]
})

r2_scores.sort_values(by='R^2 Score', ascending=False)
```
---

### Plot the R^2 scores
plt.figure(figsize=(12, 6))

plt.bar(r2_scores['Model'], r2_scores['R^2 Score'], color='skyblue')
plt.xlabel('R^2 Score')

plt.title('Model Comparison')
plt.xticks(rotation=90)
plt.show()

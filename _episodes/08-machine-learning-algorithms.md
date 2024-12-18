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
- **Linearity**: The relationship between predictors and the target is linear.
- **Independence of errors**: Residuals should be independent of each other.
- **Homoscedasticity**: Constant variance of residuals.
- **Normality of residuals** (for inference).

#### How It Works:
- Finds a linear combination of input features to predict the target value. Minimizes the sum of squared residuals.

**Pros:**
- Simple and interpretable.
- Fast to train and widely understood.

**Cons:**
- Sensitive to outliers.
- Might underfit if the relationship is non-linear.

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

### Lasso Regression

#### Assumptions:
- Similar to linear regression, but encourages sparsity in coefficients.

#### How It Works:
- Adds an L1 penalty (absolute sum of coefficients) to the cost function, which can set some coefficients to exactly zero.

**Pros:**
- Performs feature selection by eliminating irrelevant features.

**Cons:**
- Can be unstable for some datasets (feature selection might be too aggressive).

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

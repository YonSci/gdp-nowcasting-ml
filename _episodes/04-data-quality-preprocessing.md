---
title: Exploratory Data Analysis (EDA)
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

- Approaches to Handling Missing Data:
  
  - Deletion: Remove any row or observation that contains at least one missing value.
      - Pros:
         - Easy to implement.
      - Cons:
        - May result in significant data loss, reducing sample size.
        - Can lead to biased results.
        - Provides no insight into why data are missing.
      
   - Imputation: Replace missing values with estimated values derived from the available data.
  - Why Impute?: Preserves dataset size, potentially reduces bias, and retains the integrity of relationships among variables (if done correctly).  

- Common Imputation Techniques:

    - Basic Methods:
      
       - pandas Functions (fillna): Quickly fill with a fixed value or simple strategy.
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
  
   


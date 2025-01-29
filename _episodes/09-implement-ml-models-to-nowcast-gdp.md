---
title: Implementation of Machine Learning Models to Nowcast GDP
teaching: 130
exercises: 130
questions:
- "What are the fundamental concepts in ML?"

objectives:
- "Gain an understanding of fundamental machine learning concepts."

keypoints:
- "ML algorithms like linear regression, k-nearest neighbors,support vector Machine, xgboost and random forests are vital algorithms"

---
# Machine Learning for Economic Nowcasting: Real-Time GDP Estimation with Google Trends

## Why GDP nowcasting matters

- **Improve timeliness**: Calculating quarterly Gross Domestic Product (GDP) requires data from various sources, including government records and surveys. This data often takes 2-3 weeks even more to become available after the end of each quarter, making it difficult to have a real-time understanding of the economy.

- **Enhance accuracy**: Machine learning can integrate high-frequency data (Google Trends, financial transactions, satellite data) to improve estimates.

- **Optimize resource use**: Reduces dependency on costly and time-consuming traditional surveys by using alternative data sources.

- **Supporting economic policy**: Governments and policymakers need early economic insights (early warning signals) to make informed decisions and get early signals of economic changes for timely interventions. E.g: Governments can respond faster to economic slowdowns or booms.

- **Tracking economic shocks**: Detects economic downturns (e.g., COVID-19 impact, financial crises) before official statistics confirm them.
  
- **Improving investor confidence**: Real-time GDP estimates allow investors to make better decisions about trade, foreign direct investment (FDI), and market stability.

# Leveraging Machine Learning models and Google Trends data to nowcast Nigeria's quarterly GDP

## GDP Nowcasting Workflow

![](../assets/img/GDP-Nowcasting-Workflow.png)

1) **Data Collection & Preparation**

1.1) **Data Sources**:

   - **Quarterly Gross Domestic Product (GDP)**: Official quarterly Gross Domestic Product (GDP) data obtained from the Nigerian Bureau of Statistics (NBS) spanning the years 2010 to 2024.
  
![](../assets/img/gdp_quarterly_nigeria.png)
     
   - **Google Trends**: Google Trends data for Nigeria was harvested from the [Google Trends](https://trends.google.com/trends/) dataset for the same period utilizing a [web application](https://mlops-gpd-nowcasting-88t9uagbxrtgq2ajmbpcw4.streamlit.app/) developed internally within the African Centre for Statistics (ACS) to facilate the Google Trends data collection.

1.2)  **Data Loading**

- Load raw datasets (GDP & Google Trends time-series data).
- Check for missing values & detect anomalies.

2) **Data Preprocessing**

- Converted monthly Google Trends data to quarterly.
- Normalization/standardization
- Removing the long-term trends 
- Removing Seasonality

3) **Feature Engineering**

- Calculate GDP growth rate
- Created lags features for potential leading signals
  

4) **Train–Test Split**

- Partitioned the quarterly time series to keep 80% for training, 20% for out-of-sample validation.

5) **Model Training & Forecasting**
   
- Ran 10 ML models (Ridge, Lasso, ElasticNet, KNN, Decision Tree, ExtraTrees, GBM, RF, XGB, LGBM). Each was tuned via cross-validation.

6) **Hyperparameter Tuning & Cross Validation**

6.1)  **Optimization**: Grid search  to find the best parameters

6.2) **Cross-Validation Strategy**: Time-series CV (e.g., sklearn.TimeSeriesSplit).

7) **Model Evaluation**

- Calculate evaluation marices (R², MSE, MAE) on both the training set and out-of-sample test set.

8) **Model Uncertainty** (Confidence Intervals)
   - Employed bootstrap resampling on the final model’s predictions, deriving 5–95% intervals.
     
10) **Visualization**

- Compared actual vs. predicted GDP levels over time, shading the forecast intervals.

## Key Results

## Future Enhancements

## Final Wrap

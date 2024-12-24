---
title: Download Google Trends Data
teaching: 130
exercises: 130
questions:
- "What are the fundamental concepts in ML?"

objectives:
- "Gain an understanding of fundamental machine learning concepts."

keypoints:
- "ML algorithms like linear regression, k-nearest neighbors,support vector Machine, xgboost and random forests are vital algorithms"

---

# Introduction to Google Trends Data

`Google Trends` is a powerful tool that provides insights into the relative popularity of search queries over time across various regions and languages. It is widely used to analyze search trends, compare keywords, and explore interest in topics. The data is normalized on a scale of 0 to 100, with 100 representing the peak popularity of a term within the selected parameters.  

Google Trends data is valuable for various applications, including `market research`, `forecasting`, `sentiment analysis`, and `academic research`. By analyzing trends, businesses and researchers can identify patterns, understand user behavior, and make informed decisions based on public interest.

Key features of Google Trends data include:

- `Interest Over Time`: Displays how search interest in a term changes over a specific period.    
- `Regional Interest`: Shows interest levels by geography, such as countries or cities.  
- `Related Queries and Topics`: Highlights associated search terms and their popularity.  
- `Search Categories and Types`: Enables filtering by category or search type (e.g., web search, image search, YouTube search).  
  
With both manual exploration and automated methods using tools like the pytrends Python library, Google Trends data is accessible and versatile for a wide range of analyses.  


# Download Google Trends Data

## Download Google Tresnds Data  Directly from Google Trends Account.  

Step1 : Logging in with a Google account.  

 - Logging in with a Google account is optional but can provide a smoother experience and help avoid potential limitations during data downloads.
   
Step 2: Open the [Google Trends website](https://trends.google.com/trends/) in your web browser.  

![](../assets/img/gt1.png)

Step 3: Click the Explore button to begin your search.  

![](../assets/img/gt2.png)

Step 4: Search for Keywords. Enter a key term or word in the search bar.  

![](../assets/img/g3.png)

- Example terms: Economic crisis, Recession, Financial crisis, Inflation, Unemployment. List of Keywords can be found here 

Step 5: Country/Region: Select a specific country.  

Step 6: Time Period: Choose a predefined range (e.g., past 7 days, 12 months) or specify a custom range.    

Step 7: Category: Set to "All Categories" unless a specific category is relevant to your search.    

Step 8: Search Type: Select "Web Search" for general interest trends or adjust to other types like "Image Search" or "YouTube Search" as needed.    

Step 9: Review Data Visualization: Analyze the data presented in various formats, such as interest over time, geographic distribution, and related topics or queries.  

Step 10: Download Data: Click the Download icon in the top-right corner of the chart. The data will be saved in CSV format on your computer. 

Step 11: Open and Analyze the CSV File: Open the file using a spreadsheet application like Google Sheets, Microsoft Excel, or another data analysis tool.  

## Automated Google Tresnds Data Download using web appliction devloped at UNECA


## Automated Google Tresnds Data Download using the `pytrends` Python library



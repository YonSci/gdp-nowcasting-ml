---
title: Python Core Libraries
teaching: 30
exercises: 30
questions:
- "What is Python, and why is it popular?"
objectives:
- "Learn the basics of the Pandas library."

keypoints:
- "Pandas is a powerful library for data manipulation and analysis."


---

# Python Core Libraries

## NumPy


## Pandas

### Agenda 

- Learn the basics of the Pandas library.
- Understand how to work with Series and DataFrames.
- Perform data manipulation tasks like filtering, sorting, and aggregating.
- Learn how to handle missing data.
  
### Introduction to Pandas

- `Pandas` is a powerful library for `data manipulation` and `analysis`.  
- The two main data structures are:  
   - `Series`: A one-dimensional labeled array.   
   - `DataFrame`: A two-dimensional table-like data structure.  

#### Import the Pandas library
```python
import pandas as pd

# Check the version of Pandas
print(pd.__version__)
```

### Pandas Series

- A Series is a one-dimensional array with labels (index).
- It is similar to a column in a spreadsheet.

```python
# Create a Series from a list
data = [10, 20, 30, 40]
series = pd.Series(data, index=["A", "B", "C", "D"])
print(series)

# Accessing elements in a Series
print(series["B"])  # Output: 20
```

### Pandas DataFrame

- A DataFrame is a two-dimensional structure with rows and columns.  
- Think of it as a table in a database or an Excel sheet.  

#### Create a DataFrame from a dictionary
```python
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}

df = pd.DataFrame(data)
print(df)
```

#### Viewing DataFrame

- Use `DataFrame.head()` and `DataFrame.tail()` to view the top and bottom rows of the frame respectively

```python
df.head()

df.tail()
```

####  Display DataFrame information

```python
df.info()
```

#### Display the data types 

```python
df.dtypes
```

####  Display statistical summary of DataFrame

```python
df.describe()
```

#### Display the DataFrame index and columns

- Use DataFrame.index or DataFrame.columns to get the DataFrame index and columns

##### Get the DataFrame index
```python
df.index
```

##### Get the DataFrame columns
```python
df.columns
```

#### Accessing a DataFrame
```python
# Accessing a column
print(df["Name"])

# Accessing rows using loc
print(df.loc[1])  

# Accessing rows using iloc
print(df.iloc[2]) 
```

#### Adding New Columns

```python
df['Experience'] = np.random.randint(1, 5, size=len(df))

df.head()
```

#### Dropping Columns

- Use the `drop()` method to remove columns.  
- Specify `axis=1` to indicate columns.    
- Use `inplace=True` to modify the DataFrame directly (optional).  

```python
# Dropping a Single Column
df_dropped = df.drop("City", axis=1)
print(df_dropped)

# Alternatively, drop in place (modifies the original DataFrame)
df.drop("City", axis=1, inplace=True)
print(df)

```

```python
# Dropping Multiple Columns
df = pd.DataFrame(data)  # Recreate the original DataFrame
df_dropped = df.drop(["Age", "City"], axis=1)
print(df_dropped)
```

#### Dropping Rows

- Use the `drop()` method to remove rows.  
- Specify `axis=0` to indicate rows (default behavior).    
- Use `inplace=True` to modify the DataFrame directly (optional).  

```python
# Drop the row with index 1
df_dropped = df.drop(1, axis=0)
print(df_dropped)

# Alternatively, drop in place
df.drop(1, axis=0, inplace=True)
print(df)
```

```python
# Drop multiple rows
df = pd.DataFrame(data)  # Recreate the original DataFrame
df_dropped = df.drop([0, 2], axis=0)
print(df_dropped)
```


### Reading and Writing Data

- Pandas supports reading and writing data in multiple formats (CSV, Excel, JSON, etc.).

#### Read data from a CSV file
```python
df = pd.read_csv("example.csv")
print(df)
```

#### Write data to a CSV file
```python
df.to_csv("output.csv", index=False)
```

#### Reading Excel Files

```python
# Read data from an Excel file
df = pd.read_excel("example.xlsx", sheet_name="Sheet1")  # Specify the sheet name

# Read the first 5 rows (head) of the DataFrame
print(df.head())
```

#### Writing Excel Files
 - To save a DataFrame to an Excel file, use the `to_excel()` function.

```python
# Save DataFrame to an Excel file
df.to_excel("output.xlsx", index=False)  # Save without including the index
```

##### Writing Multiple Sheets

- To write multiple DataFrames to different sheets in the same Excel file, use ExcelWriter.

```python
# Create multiple DataFrames
df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
df2 = pd.DataFrame({"X": [7, 8, 9], "Y": [10, 11, 12]})

# Write to multiple sheets
with pd.ExcelWriter("multi_sheet_output.xlsx") as writer:
    df1.to_excel(writer, sheet_name="Sheet1", index=False)
    df2.to_excel(writer, sheet_name="Sheet2", index=False)
```

#### Reading JSON Files

```python
# Read data from a JSON file
df = pd.read_json("example.json")

# Display the first 5 rows
print(df.head())
```

#### Writing JSON Files

- To save a DataFrame to a JSON file, use the `to_json()` function.

```python
# Save DataFrame in a column-oriented JSON format
df.to_json("column_oriented.json", orient="columns")
```


### Data Manipulation

- `Filtering`: Select specific rows or columns.  
- `Sorting`: Sort data based on column values.  
- `Aggregating`: Perform operations like sum, mean, count, etc.

#### Filtering

```python
# Filter rows where Age > 25
filtered_df = df[df["Age"] > 25]
print(filtered_df)
```

```python
# Filter rows where Age > 25 and City is "Chicago"
filtered_df = df[(df["Age"] > 25) & (df["City"] == "Chicago")]
print(filtered_df)
```

#### Sort by Age in descending order
```python
sorted_df = df.sort_values("Age", ascending=False)
print(sorted_df)
```

#### Aggregations

```python
# Calculate the mean age
mean_age = df["Age"].mean()
print(f"Mean Age: {mean_age}")
```

### Selecting Columns and Rows in Pandas

#### Selecting Columns

- You can select one or multiple columns from a DataFrame.    
- Columns can be accessed using `square brackets []` or `dot notation` (only for single-column names without spaces).

##### Selecting a Single Column

```python
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}
df = pd.DataFrame(data)

# Select a single column
age_column = df["Age"]
print(age_column)

# Alternatively, use dot notation
city_column = df.City
print(city_column)
```

##### Selecting Multiple Columns

```python
subset = df[["Name", "City"]]
print(subset)
```

#### Renaming Columns

```python
df.rename(columns={"Age": "Years"}, inplace=True)
print(df)
```

### Selecting Rows

- Use `.loc[]` for label-based selection (index labels).    
- Use `.iloc[]` for position-based selection (index positions).

#### Selecting Rows by Index Label (.loc[])  

```python
# Select a single row by index label
row_1 = df.loc[1]
print(row_1)

# Select multiple rows by index labels
rows = df.loc[[0, 2]]
print(rows)
```

#### Selecting Rows by Index Position (.iloc[])

```python
# Select a single row by index position
row_0 = df.iloc[0]
print(row_0)

# Select multiple rows by index positions
rows = df.iloc[[0, 2]]
print(rows)
```


#### Selecting Specific Rows and Columns

- Combine .loc[] or .iloc[] to select specific rows and columns.

##### Using .loc[] for Specific Rows and Columns

```python
# Select rows 0 and 2, and columns "Name" and "City"
subset = df.loc[[0, 2], ["Name", "City"]]
print(subset)
```

##### Using .iloc[] for Specific Rows and Columns
```python
# Select rows 0 and 2, and columns at positions 0 and 2
subset = df.iloc[[0, 2], [0, 2]]
print(subset)

```

### Handling Missing Data

- Use `isna()` to identify missing values.  
- Fill missing values with fillna().
- Remove rows/columns with missing values using dropna().

#### Handling Missing Data

#### Create a DataFrame with missing values
```python
data = {
    "Name": ["Alice", "Bob", None],
    "Age": [25, None, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}
df = pd.DataFrame(data)
```

#### Check for missing values
```python
print(df.isna())
```

#### Fill missing values
```python
filled_df = df.fillna("Unknown")
print(filled_df)
```

#### Drop rows with missing values
```python
cleaned_df = df.dropna()
print(cleaned_df)
```

### Summary of Pandas Selection Methods

| Selection Type               | Syntax Example                         | Notes                                   |
|------------------------------|-----------------------------------------|-----------------------------------------|
| **Single Column**            | `df["column_name"]`                    | Access column as a Series.             |
| **Multiple Columns**         | `df[["col1", "col2"]]`                 | Returns a new DataFrame.               |
| **Single Row (label)**       | `df.loc[label]`                        | Label-based row selection.             |
| **Single Row (position)**    | `df.iloc[position]`                    | Position-based row selection.          |
| **Specific Rows and Columns**| `df.loc[[rows], [columns]]`            | Label-based row/column selection.      |
| **Filtering Rows**           | `df[condition]`                        | Filter rows based on conditions.       |


## Matplotlib


## Exercise

1) Create a Pandas Series with the following data: [100, 200, 300]. Assign the indices as ["X", "Y", "Z"].  
2) Create a DataFrame with columns Product, Price, and Stock using the following data:

  Products: ["Laptop", "Tablet", "Smartphone"]
  Prices: [1000, 500, 800]
  Stock: [50, 100, 200]
  Filtering and Sorting

3) Filter rows in the DataFrame where Price > 600.  
4) Sort the DataFrame by Stock in ascending order.
5) Create a DataFrame with some missing values and write code to:  
   - Replace missing values with "Not Available".  
   - Remove rows with missing values    


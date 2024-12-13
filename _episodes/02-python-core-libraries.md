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

---

## NumPy

### Agenda 

- Learn the basics of `NumPy`, a library for `numerical computing` in Python.  
- Understand and `create arrays`, the core data structure in NumPy.  
- Perform basic `mathematical operations` on arrays.  
- Learn advanced operations like `reshaping`, `slicing`, and `broadcasting`.    


### Introduction to NumPy

- `NumPy` stands for "Numerical Python" and provides `fast`, `efficient`, and `flexible` array operations.  
- The core object in NumPy is the `ndarray` (N-dimensional array).
- Designed for scientific computation: useful `linear algebra`, `Fourier transform`, and `random number capabilities` etc.


#### Import the NumPy library

```python
import numpy as np

# Check the version of NumPy
print(np.__version__)
```

#### Creating NumPy Arrays

##### Creating Arrays from Lists

###### Create a 1D array

```python
arr = np.array([1, 2, 3, 4])
print(arr)
```

###### Create a 2D array

```python
arr_2d = np.array([[1, 2], [3, 4]])
print(arr_2d)
```

###### Create a 3D Arrays

```python
c = np.array([[[1,1,2], [2,3,3]], [[1,1,1], [1,1,1]], [[1,1,1], [1,1,1]]])
```

##### Creating Arrays with NumPy Functions

###### Create an array of zeros

```python
zeros = np.zeros((2, 3))  # 2 rows, 3 columns
print(zeros)
```

###### Create an array of ones

```python
ones = np.ones((3, 2))  # 3 rows, 2 columns
print(ones)
```

###### Random numbers

```python
a = np.random.rand(4)       # uniform in [0, 1]

b = np.random.randn(4)      # Gaussian
```


###### Create an array with a range of numbers

```python
range_arr = np.arange(0, 10, 2)  # Start at 0, stop before 10, step by 2
print(range_arr)
```

###### Create an array with evenly spaced numbers

```python
linspace_arr = np.linspace(0, 1, 5)  # Start at 0, stop at 1, 5 evenly spaced values
print(linspace_arr)
```




#### Basic Array Operations

##### Mathematical Operations

###### Perform basic math operations

```python
arr = np.array([1, 2, 3, 4])
print(arr + 2)  # Add 2 to each element
print(arr * 3)  # Multiply each element by 3
```

##### Perform element-wise operations with another array

```python
arr2 = np.array([5, 6, 7, 8])
print(arr + arr2)  # Element-wise addition
print(arr * arr2)  # Element-wise multiplication
```

##### Array Properties

##### Get properties of an array

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # Shape of the array (rows, columns)
print(arr.size)   # Total number of elements
print(arr.dtype)  # Data type of elements
```

#### Indexing and Slicing

##### Indexing

###### Access elements of a 1D array

```python
arr = np.array([10, 20, 30, 40])
print(arr[0])  # First element
print(arr[-1])  # Last element
```

###### Access elements of a 2D array

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d[0, 1])  # Element in first row, second column
print(arr_2d[1, 2])  # Element in second row, third column
```

##### Slicing

###### Slice a 1D array

```python
arr = np.array([10, 20, 30, 40, 50])
print(arr[1:4])  # Elements from index 1 to 3
print(arr[:3])   # First three elements
```

###### Slice a 2D array

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr_2d[:2, 1:])  # First two rows, columns 2 and 3
print(arr_2d[1:, :2])  # Last two rows, first two columns
```

#### Reshaping Arrays

##### Changing the Shape of an Array

###### Reshape a 1D array to a 2D array

```python
arr = np.arange(1, 10)
reshaped = arr.reshape(3, 3)  # 3 rows, 3 columns
print(reshaped)
```

###### Flatten a 2D array back to 1D

```python
flattened = reshaped.flatten()
print(flattened)
```

#### Broadcasting

- Broadcasting allows operations between arrays of different shapes.
  
##### Add a scalar to a 2D array

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d + 10)
```
##### Add a 1D array to a 2D array

```python
row = np.array([1, 2, 3])
print(arr_2d + row)
```

#### Asking For Help

```python
np.add?
np.array?
np.arr*?
np.con*?
```

### Key Points:

- NumPy arrays are efficient for numerical computations.    
- Perform operations like addition, multiplication, and broadcasting directly on arrays.    
- Use slicing and indexing to access or modify specific elements.    

---

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

##### Print the head of the DataFrame
```python
df.head()
```

##### Print the tail of the DataFrame
```python
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

---

## Matplotlib

### Agenda

- Learn the basics of the Matplotlib library.      
- Understand how to create various types of plots (line, bar, scatter, etc.).    
- Customize plots with titles, labels, legends, and styles.    
- Save plots as image files (PNG, PDF, SVG, EPS, and PGF.).  

### Introduction to Matplotlib 

- Matplotlib is a python 2D and 3D plotting library which produces scientific figures and publication quality figures.
- The core component is the `pyplot` module, often imported as plt.
- It is modeled closely after "Matlab™". Therefore, the majority of plotting commands in pyplot have Matlab™ analogs with similar arguments.
- You can create a wide variety of plots, including `line plots`, `bar charts`, `histograms`, `scatter plots`, and more.  

#### Importing Matplotlib

```python
# Import Matplotlib's pyplot module
import matplotlib.pyplot as plt

# Check the Matplotlib version
print(plt.__version__)
```

#### Line Plot

##### Creating a Simple Line Plot

```python
# Data for the plot
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

# Create a line plot
plt.plot(x, y)
plt.title("Simple Line Plot")  # Add a title
plt.xlabel("X-axis")           # Add an X-axis label
plt.ylabel("Y-axis")           # Add a Y-axis label
plt.show()                     # Display the plot
```

![Line Plot one](../assets/imag/line1.pmg)

##### Customizing the Line Plot

```python
# Customize line style, color, and markers
plt.plot(x, y, color="red", linestyle="--", marker="o", label="Data")
plt.title("Customized Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()  # Add a legend
plt.grid(True)  # Add grid lines
plt.show()
```

#### Bar Chart

##### Creating a Bar Chart

```python
# Data for the bar chart
categories = ["A", "B", "C", "D"]
values = [5, 7, 8, 6]

# Create a bar chart
plt.bar(categories, values, color="blue")
plt.title("Bar Chart")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()
```

##### Horizontal Bar Chart

```python
# Create a horizontal bar chart
plt.barh(categories, values, color="green")
plt.title("Horizontal Bar Chart")
plt.xlabel("Values")
plt.ylabel("Categories")
plt.show()
```

#### Scatter Plot

##### Creating a Scatter Plot

```python
# Data for the scatter plot
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11]
y = [99, 86, 87, 88, 100, 86, 103, 87, 94, 78]

# Create a scatter plot
plt.scatter(x, y, color="purple", marker="x")
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

#### Histogram

##### Creating a Histogram

```python
# Data for the histogram
data = [22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27]

# Create a histogram
plt.hist(data, bins=5, color="orange", edgecolor="black")
plt.title("Histogram")
plt.xlabel("Data")
plt.ylabel("Frequency")
plt.show()
```

#### Pie Chart

##### Creating a Pie Chart

```python
# Data for the pie chart
sizes = [30, 20, 35, 15]
labels = ["Apples", "Bananas", "Cherries", "Dates"]

# Create a pie chart
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title("Pie Chart")
plt.show()
```

#### Subplots
- Subplots allow you to create multiple plots in a single figure.  
- Use `plt.subplot()` for simple grids and `plt.subplots()` for more control.

##### Creating Simple Subplots

```python
import matplotlib.pyplot as plt
import numpy as np

# Data for plots
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create subplots
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.plot(x, y1, label="sin(x)", color="blue")
plt.title("Sine Wave")
plt.legend()

plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(x, y2, label="cos(x)", color="red")
plt.title("Cosine Wave")
plt.legend()

plt.tight_layout()  # Adjust layout to avoid overlapping
plt.show()
```

##### Using plt.subplots() for More Control

```python
# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Sine wave
axes[0, 0].plot(x, y1, color="blue")
axes[0, 0].set_title("Sine Wave")

# Cosine wave
axes[0, 1].plot(x, y2, color="red")
axes[0, 1].set_title("Cosine Wave")

# Exponential
y3 = np.exp(x / 10)
axes[1, 0].plot(x, y3, color="green")
axes[1, 0].set_title("Exponential Function")

# Logarithmic
y4 = np.log(x + 1)
axes[1, 1].plot(x, y4, color="purple")
axes[1, 1].set_title("Logarithmic Function")

# Add overall titles and adjust layout
fig.suptitle("Multiple Plots in One Figure", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust space for the title
plt.show()
```

#### 3D Plots (Optional)

- Matplotlib provides 3D plotting functionality through the mpl_toolkits.mplot3d module.  
- Use ax.plot_surface() for surface plots and ax.scatter() for 3D scatter plots.

##### 3D Line Plot
```python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Data for a 3D line
z = np.linspace(0, 1, 100)
x = z * np.sin(25 * z)
y = z * np.cos(25 * z)

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(x, y, z, label="3D Spiral", color="blue")
ax.set_title("3D Line Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()

plt.show()
```

##### 3D Surface Plot

```python
# Create data for a surface plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# Create a 3D surface plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(x, y, z, cmap="viridis")
fig.colorbar(surf)  # Add a color bar for reference
ax.set_title("3D Surface Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

plt.show()
```

##### 3D Scatter Plot

```python
# Create random 3D data
np.random.seed(42)
x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x, y, z, color="red", marker="o", label="Data Points")
ax.set_title("3D Scatter Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()

plt.show()
```



#### Saving a Plot

##### Saving a Plot as an Image

```python
# Create a plot
plt.plot(x, y)
plt.title("Line Plot")

# Save the plot as a PNG file
plt.savefig("line_plot.png")

# Save the plot as a high-quality image
plt.savefig("line_plot_high_res.png", dpi=300)

# Display the plot
plt.show()
```

### Key Points:

- Matplotlib provides tools for creating a variety of `plots` and `visualizations`.  
- Use `plt.plot()` for line plots, `plt.bar()` for bar charts, `plt.scatter()` for scatter plots, and `plt.hist()` for histograms.  
- Customize plots with `titles`, `axis` `labels`, `legends`, and `grid lines`.  
- Save your plots using `plt.savefig()` in various formats.
  
---

## Exercise

### Numpy

1) Create a 1D array of integers from 5 to 15.  
2) Create a 2D array of shape (3, 3) filled with ones.  
3) Multiply all elements in the array [2, 4, 6, 8] by 3.  
4) Add the arrays [1, 2, 3] and [4, 5, 6].  
5) Access the last row of the array [[1, 2, 3], [4, 5, 6], [7, 8, 9]].  
6) Slice the array np.arange(10) to get only even numbers.  
7) Reshape the array np.arange(12) into a (3, 4) array.  
8) Flatten the reshaped array back into 1D.    

### Pandas

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
   - Remove rows with missing values.  
   
### Matplotlib

1) Create a line plot for x = [1, 2, 3, 4, 5] and y = [10, 20, 25, 30, 35]. Customize the plot by changing the line color, adding markers, and a legend.  
2) Create a bar chart for categories ["Math", "Science", "History", "English"] and values [85, 90, 75, 80]. Add a title, and X and Y axis labels.  
3) Plot x = [10, 20, 30, 40, 50] and y = [5, 15, 25, 35, 45] as a scatter plot. Change the marker style and color.  
4) Create a histogram for the dataset [5, 7, 8, 9, 5, 3, 4, 5, 7, 8, 9, 10] with 4 bins.  
5) Create a pie chart for sizes = [40, 30, 20, 10] with labels ["A", "B", "C", "D"]    

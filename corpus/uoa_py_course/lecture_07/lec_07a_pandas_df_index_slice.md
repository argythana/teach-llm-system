<!-- source: lectures_07_13_pandas_plots_scikit/lecture_07_pandas/reading_material/lec_07a_pandas_df_index_slice.ipynb @ 0cc874704aaa -->

# Lecture 07a: Introduction to Pandas — DataFrames, Indexing & Slicing

## What is [pandas?](https://pandas.pydata.org/)

[Pandas wikipedia entry](https://en.wikipedia.org/wiki/Pandas_(software))

**Necessary homework reading:**
1) From the [User Guide](https://pandas.pydata.org/docs/user_guide/index.html), study: [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html). As the title says, it takes no more than 2 hours and 10 minutes!
2) More on [indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html#)   

Advanced Reading: more complex recipes in the pandas [Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html#cookbook).

### Lecture outline:
1. Open a `.csv` file and read the data as a pandas DataFrame.
2. Basic DataFrame methods, attributes, and descriptive statistics.
3. Columns and Index — understanding the structure.
4. Select (Locate) data with `.loc` (label-based) and `.iloc` (position-based).
5. Alternative slicing syntax (bracket notation, dot notation).
6. Transpose a DataFrame — `.loc` vs `.iloc` with string labels.

### Datasets used:
- `temp_to_coffee.csv` — a small, simple dataset to get started.
- `predict_heart_disease_train.csv` — a medical dataset with 426 patients and 15 features for the main examples.

```python
!pwd  # Present working directory. This is where your notebook is running on your computer.
```

#### Module not found error if not installed.
**To Install it, open CLI, go to course folder, activate venv, type`pip install pandas` .**

```python
# Import module_name as alias (useful convention)
import pandas as pd
```

## 1. Open a `.csv` file and read the data as a pandas dataframe.

#### If you get a very red, angry "FileNotFoundError" (file not found Error):  

Guess what is the problem:   
a) You don't have the file in the correct folder.   
b) You don't use the correct file name.   
c) YOu don't run the notebook from the correct folder.

**d) Our teacher is sooo bad, and it is his fault!**

e) All of the above.   

e) Is is the correct answer, if you are a beginner, and that's normal because it is mainly **d)**.

```python
# First download the file from the eclass directory!

# Uncomment to use the code below if you put the data file in the same folder as the notebook.
df = pd.read_csv("temp_to_coffee.csv")
```

```python
# Use the code below if you put the data file in a directory that has the same structure as I do.
# If not, comment it out.
# df = pd.read_csv("../../data/temp_to_coffee.csv")
```

```python
# see the dataframe
df
```

```python
# first 4 rows
df.head(2)
```

```python
type(df)
```

## 2. Basic DataFrame methods, attributes, and descriptive statistics.
These work on any DataFrame, regardless of the dataset.

```python
# Number of rows (observations).
len(df)
```

```python
# The names of the columns (features) in the data.
df.columns
```

```python
# Number of non-null observations per column.
df.count()
```

```python
# Total number of data points (rows × columns).
df.size
```

```python
# Shape: (rows, columns) as a tuple.
df.shape
```

```python
# Number of rows only.
df.shape[0]
```

```python
# Number of columns only.
df.shape[1]
```

```python
# The row index — a RangeIndex starting from 0.
df.index
```

```python
type(df.index)
```

```python
# Descriptive statistics for all numeric columns.
df.describe()
```

```python
# Custom percentiles — pass a list of percentile values.
df.describe(percentiles=[0.1, 0.2, 0.4, 0.95])
```

### A real-world DataFrame: Heart Disease Prediction
We now load a larger, more realistic dataset: **predict_heart_disease_train.csv**.  
It contains medical data for 630k patients, with features like Age, Blood Pressure, Cholesterol, and a target column `Heart Disease` (Presence/Absence).

We reuse the same variable name `df`. Why? **Convention, Reproducibility, Readability.**  
In practice, you work with one main DataFrame at a time, and `df` is the standard name for it.

```python
# Explore available read methods by uncommenting and pressing Tab:
# pd.read_

# Comment it out again after exploring.
```

```python
# Uncomment to see the full docstring for read_csv:
# pd.read_csv?

# Comment it out again.
```

```python
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# NOTE: We intentionally omit index_col here to demonstrate the duplicated index issue below.
df = pd.read_csv("predict_heart_disease_train.csv")

# Use the code below if you put the data file in the data/ directory.
# df = pd.read_csv("../../data/predict_heart_disease_train.csv")
```

```python
# Display the full DataFrame. Notice: 630000 rows × 15 columns.
df
```

#### Spot the problem: the `id` column duplicates the index!

Look at the DataFrame above. The leftmost bold numbers (0, 1, 2, ...) are the **auto-generated RangeIndex**.  
The `id` column also contains 0, 1, 2, ... — **the exact same values**.

**Root cause:** The CSV file has an `id` column that was meant to be the row identifier.  
But `pd.read_csv()` doesn't know that — it always creates its own default integer index.  
So you end up with **two copies** of the same information: one as the index, one as a column.

**Fix:** Use the `index_col` parameter to tell pandas which column IS the index.

```python
# Fix: re-read with index_col="id" so the id column becomes the index.
df = pd.read_csv("predict_heart_disease_train.csv", index_col="id")
df.head(3)  # Now "id" is the index, not a separate column.
```

```python
# A random sample of 5 rows — useful for a quick peek at the data.
df.sample(5)
```

```python
# Column names — notice some have spaces (e.g. "Chest pain type", "Max HR").
df.columns
```

```python
df.head(3)  # First 3 rows to see the column names and index.
```

```python
df.tail(3)  # Last 3 rows to see the index and data at the end of the DataFrame.
```

```python
len(df)
```

```python
# Info on column types, non-null counts, and memory usage.
# This dataset has no missing values — all columns show 426 non-null entries.
df.info()
```

## 3. Columns and Index 
**Columns** = features, dimensions, variables (independent X, dependent y).  
In our heart disease dataset: Age, BP, Cholesterol, etc. are **features (X)**; `Heart Disease` is the **target (y)**.

**Integer Index** = row numbers, representing individual observations/patients (usually).  
By default, pandas creates a `RangeIndex` starting from 0.

```python
# Row index: RangeIndex(start=0, stop=426, step=1)
df.index
```

```python
# Data types per column — mostly int64/float64 for medical measurements, object for "Heart Disease".
df.dtypes
```

## 4. Select (Locate) data: `.loc` vs `.iloc`
Recommended [StackOverflow answer](https://stackoverflow.com/a/31593712) on the difference.

| Method | Based on | End is... | Example |
|--------|----------|-----------|---------|
| `.iloc` | **Position** (integer) | Exclusive | `df.iloc[0:2]` → rows 0, 1 |
| `.loc`  | **Label** (name)       | Inclusive  | `df.loc[0:2]` → rows 0, 1, 2 |

**Syntax: (Rows, Columns)** — row index first, column index second.

### 4.1 Slice rows

```python
# .iloc: first row (position 0). Returns a Series.
df.iloc[0:3]
```

```python
# .loc: row with label 0. Same result here because labels happen to be integers.
df.loc[0:3]
```

```python
# Second row (position 1).
df.iloc[1]
```

```python
# .iloc is EXCLUSIVE (like Python lists): 0:2 gives rows 0 and 1.
df.iloc[0:2]
```

```python
# .loc is INCLUSIVE: 0:2 gives rows 0, 1, AND 2.
df.loc[0:2]
```

### 4.2 Slice columns

```python
# Select a single column by name. Single brackets → returns a Series.
df["Age"]
```

```python
type(df["Age"])  # Single brackets → Series
```

```python
# Double brackets → returns a DataFrame (even for a single column).
df[["Age"]]
```

```python
type(df[["Age"]])  # Double brackets → DataFrame
```

```python
# Select multiple columns by passing a list of column names.
df[["Age", "Cholesterol", "BP"]]
```

### 4.3 Slice rows and columns

```python
# .loc: rows 0-5, columns from "Age" to "BP" (inclusive on both ends).
# This gives us: Age, Sex, Chest pain type, BP
df.loc[0:5, "Age":"BP"]
```

```python
# All rows, columns from "Age" to "BP".
df.loc[:, "Age":"BP"]
```

```python
# First 6 rows, columns from "Sex" to "Cholesterol".
df.loc[:5, "Sex":"Cholesterol"]
```

```python
type(df.loc[:, "Age":"Sex"])
```

```python
# .loc is INCLUSIVE — because it works with "names" (labels), not positions.
df.loc[0:3, "Age":"BP"]

# Inclusive: rows 0, 1, 2, 3 AND columns Age, Sex, Chest pain type, BP.
```

```python
# First 11 rows, ALL columns.
df.loc[0:10, :]
```

```python
# Rows 0-6, and a LIST of specific column names.
df.loc[0:6, ["Age", "Cholesterol"]]
```

```python
# A single value: row 8, column "Age".
df.loc[8, "Age"]
```

## 5. Alternative syntax to slice rows and columns.   
You may use similar notation to lists and strings indexing, without `.loc` nor `.iloc`.

**Dot notation caveat:** `df.Age` works (no spaces), but `df.Chest pain type` does **NOT**.  
Columns with spaces **must** use bracket notation: `df["Chest pain type"]`.

```python
# Dot notation: value of "Age" for row 8.
df.Age[8]
```

```python
# Basic indexing as we do it in lists and strings works similarly on df rows.
# EXCLUSIVE because it works with ordered indexing in numbers.

df[10:13]
```

```python
# Dot notation + slice — EXCLUSIVE (like list indexing).
df.Age[10:13]
```

```python
# Same result, bracket notation instead of dot notation.
# EXCLUSIVE — works like list/string indexing.
df["Age"][10:13]
```

```python
# Rows first, then column — same result, different order of operations.
df[10:13]["Age"]
```

```python
# Select columns first, then slice rows.
df[["Age", "Sex"]][10:13]
```

```python
# Slice rows first, then select columns — same result, different order.
df[10:13][["Age", "Sex"]]
```

```python
# Negative indexing: last 10 to last 5 Cholesterol values.
df.Cholesterol[-10:-5]
```

```python
type(df.Cholesterol[-10:-5])
```

```python
df[5:10]
```

```python
# Chained slicing: slice rows, then slice rows again.
# First: rows 5-9. Then: first 3 of those → rows 5, 6, 7.
# Tricky! The second slice uses POSITIONAL indexing on the result.

df[5:10][0:3]
```

```python
# Step 1: Slice rows 5 to 9.
slice_on_df_rows = df[5:10]
```

```python
slice_on_df_rows
```

```python
# Step 2: Take first 3 rows from the previous slice → rows 5, 6, 7.
second_slice = slice_on_df_rows[0:3]
second_slice

# Understanding chained slicing is important — it will save you hours of debugging.
```

### Can you spot the difference between `.loc` and bracket slicing?
Hint: count the rows returned in each cell below.

```python
# .loc is INCLUSIVE — returns rows 0 through 5.
df.loc[:5, "Age"]
```

```python
# Bracket/slice notation is EXCLUSIVE — returns rows 0 through 4.
df.Age[:5]
```

```python
df[2:5]["Age"]
```

## 6. Transpose a DataFrame
Sometimes useful for reshaping data.  
Here we use it to practice `.loc` and `.iloc` when row labels are **strings** (column names become row labels after transposing).

```python
# Transpose: rows ↔ columns. Just a view, does NOT modify df.
df.transpose()#.head(4)  # alternative syntax: df.T
```

```python
# df has NOT changed in memory — transpose() returns a view.
df
```

```python
# Nothing but a view of the transpose of the df.
df.T
```

```python
# df has not changed in memory.
df
```

```python
# Assign the transposed view to a new variable (df stays unchanged).
transposed_df = df.T  # .T is shorthand for .transpose()
```

```python
transposed_df
```

```python
# .iloc still works — it uses positional (integer) indexing.
transposed_df.iloc[0:4]
```

```python
# .loc uses labels — now the row labels are the original column names (strings!).
transposed_df.loc["Age":"Sex"]
```

```python
# Range of row labels AND range of column labels (columns are now the patient indices).
transposed_df.loc["Age":"BP", 1:6]
```

```python
# Index of transposed_df = the original column names (strings).
transposed_df.index
```

```python
# Columns of transposed_df = the original row indices (integers).
transposed_df.columns
```

```python
# Select a single row by its string label.
transposed_df.loc["Age"]
```

```python
# Range of row labels — from "Age" to "BP" (inclusive).
transposed_df.loc["Age":"BP"]
```

```python
# A LIST of specific row labels.
transposed_df.loc[["Age", "BP", "Heart Disease"]]
```

#### Revision example

```python
# This raises an error! Can you figure out why?
# +5000 points if you spot it before running the cell.

transposed_df.loc[["Age", "BP"]]
```

```python
# Explanation:
# transposed_df.loc["Age", "BP"] means: row="Age", column="BP".
# But the COLUMNS of transposed_df are integers (0, 1, 2, ..., 425), not feature names!
# "BP" is a ROW label, not a column label.
# Fix: use double brackets to select a LIST of rows → transposed_df.loc[["Age", "BP"]]
```

```python
# Correct: double brackets → list of row labels.
transposed_df.loc[["Age", "BP"]]
```

```python
# This works: row label "Age", column label 2 (patient index 2).
transposed_df.loc["Age", 2]
```

#### Summary: selecting multiple rows or columns by name requires a **list** inside `.loc`.
- Single label: `transposed_df.loc["Age"]` → one row  
- List of labels: `transposed_df.loc[["Age", "BP"]]` → multiple rows  
- Range of labels: `transposed_df.loc["Age":"BP"]` → continuous range

```python
# List of row labels AND list of column labels (patient indices).
transposed_df.loc[["Age", "Cholesterol"], [2, 3, 4, 5]]
```

```python
# Chained bracket slicing on the transposed df.
# First slice: rows up to "Cholesterol". Second slice: columns 3 to 5 (positional).
# Be very careful: BOTH slices here act on ROWS, not rows then columns!

transposed_df[:"Cholesterol"][3:6]
```

<!-- source: lectures_07_13_pandas_plots_scikit/lecture_07_pandas/reading_material/lec_07b_pandas_methods.ipynb @ 0cc874704aaa -->

# Lecture 07b: Pandas Methods — View vs Modify, Rename, Drop, Filter, Group

**Dataset:** `predict_heart_disease_train.csv` — 630,000 patients, 14 medical features + target.

### Outline:
1. View method output VS DataFrame modification (assignment needed!).
2. Basic methods: rename columns, handle missing values, drop columns.
3. Replace values in a column using a dictionary.
4. Select data using boolean conditions (AND `&`, OR `|`).
5. Set a column as index.
6. Group by a column or list of columns (`groupby`).

```python
# Present working directory. This is the location of the notebook in your computer.
!pwd
```

```python
import pandas as pd
```

### Read the heart disease CSV file.
Use `index_col="id"` to avoid the duplicated index issue we saw in Lecture 07a.

```python
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
df = pd.read_csv("predict_heart_disease_train.csv", index_col="id")

# Use the code below if you put the data file in the data/ directory.
# df = pd.read_csv("../../data/predict_heart_disease_train.csv", index_col="id")
```

```python
# 426 rows, 14 columns. All non-null — this dataset is clean (for now).
df.info()
```

```python
# Notice: column names have spaces and mixed case — we'll fix that later.
df.head(3)
```

## 1. View versus "modify" — a critical distinction

Most pandas methods return a **new object** (a "view") and do **NOT** change the original DataFrame.  
To actually modify the DataFrame, you must **assign the result** back to a variable.

This is the single most common source of confusion for beginners.

```python
# Column names — notice the spaces. We'll rename them in Section 2.
df.columns
```

```python
# Selecting a single row returns a Series (not a DataFrame).
# This is patient 0's data.
df.loc[0]
```

```python
# A single row → pandas Series.
# https://pandas.pydata.org/docs/reference/api/pandas.Series.html
type(df.loc[0])
```

```python
# Multiple rows → DataFrame. .loc is inclusive.
df.loc[0:2]
```

```python
# Skip the first 5 patients. This returns a VIEW — does NOT modify df.
# The index still starts from 5 (not reset to 0).
df.loc[5:]
```

```python
# df is unchanged — the .loc above was just a view!
df.head(2)
```

```python
# Chaining .reset_index() also does NOT modify df — still just a view.
df.loc[5:].reset_index(drop=True)
```

```python
# dStill unchanged! Methods return new objects, they don't modify the original.
df.head(2)
```

### 1.1 To actually modify: create a copy, use assignment, reset the index

Use `.copy()` to create an independent copy. Then use **assignment** (`=`) to store the result.

```python
# .copy() creates an independent copy — changes to copy_df won't affect df.
copy_df = df.copy()
```

```python
# NOW we modify by assignment: skip first 5 rows, reset index to start from 0.
# drop=True discards the old index (doesn't add it as a column).
copy_df = copy_df.loc[5:].reset_index(drop=True)
```

```python
# Index now starts from 0. First row is the patient that was previously at index 5.
copy_df.head(2)
```

## 2. Basic methods on a pandas DataFrame

### 2.1 Rename columns

**Why rename?** Our columns have spaces (`"Chest pain type"`) and mixed case (`"Max HR"`).  
This prevents dot notation (`df.Chest pain type` → SyntaxError!) and makes code harder to write.  
**Best practice:** lowercase, underscores, no spaces → `snake_case`.

```python
# View only — rename a single column. Does NOT modify copy_df.
copy_df.rename(columns={"Age": "age"}).head(2)  # rename takes a dict: {"old_name": "new_name"}
```

```python
# Column name is still "Age" — rename() returned a view, not a modification.
copy_df.head(2)
```

```python
# Still the original names — spaces, mixed case.
copy_df.columns
```

```python
# Step 1: Get the current column names as a list.
list_of_column_names = list(copy_df.columns)
list_of_column_names
```

```python
# # Dict comprehension — creates the skeleton, but you'd still need to fill in new names.
# dict_of_column_renames = {col: "new_name" for col in list_of_column_names}
# dict_of_column_renames
```

```python
# Step 2: Create a list of new snake_case names.
# Rule: lowercase, underscores instead of spaces, no special characters.
list_of_new_column_names = [
    "age", "sex", "chest_pain_type", "bp", "cholesterol",
    "fbs_over_120", "ekg_results", "max_hr", "exercise_angina",
    "st_depression", "slope_of_st", "num_vessels_fluro", "thallium", "heart_disease",
]

list_of_new_column_names
```

```python
# Step 3: zip the two lists and create a dictionary: {"old_name": "new_name"}.
dict_of_column_renames = dict(
    zip(list_of_column_names, list_of_new_column_names)
)
```

```python
# The mapping: original name → snake_case name.
dict_of_column_renames
```

```python
# Preview: rename returns a VIEW. copy_df is NOT modified yet.
copy_df.rename(columns=dict_of_column_renames).head(2)
```

```python
# Confirm: columns are still the original names.
copy_df.columns
```

```python
# NOW we modify by assignment. This actually changes copy_df.
# Warning: don't chain .head(2) here or you'll overwrite copy_df with just 2 rows!
copy_df = copy_df.rename(columns=dict_of_column_renames)
```

```python
# copy_df.rename?
```

```python
# Now columns are snake_case. Dot notation works: copy_df.age, copy_df.bp, etc.
copy_df.head(3)
```

```python
copy_df.max_hr.mean()
```

### 2.2 Find and handle missing values (NaN)

In real-world data, missing values are **very common**. Pandas represents them as `NaN` (Not a Number).

Our heart disease dataset is clean (no NaN), so we'll first **verify** that, then **intentionally inject** some NaN values to practice the `dropna()` workflow.

Advanced Reading: [Is inplace harmful or not?](https://stackoverflow.com/a/59242208)

```python
# .isnull() returns True/False for every cell. All False → no missing data.
copy_df.isnull()
```

```python
# .any(axis=0): is there at least one NaN per COLUMN? All False → clean data.
copy_df.isnull().any(axis=0)
```

```python
# .any(axis=1): is there at least one NaN per ROW? All False → clean data.
copy_df.isnull().any(axis=1)
```

```python
# Filter rows that have ANY NaN — empty DataFrame because our data is clean.
copy_df[copy_df.isnull().any(axis=1)]
```

```python
# Let's intentionally inject NaN to practice the dropna workflow.
import numpy as np

# Create a dirty copy. Set some cholesterol values to NaN.
dirty_df = copy_df.copy()

dirty_df.loc[0:2, "cholesterol"] = np.nan    # rows 0, 1, 2
dirty_df.loc[10:12, "bp"] = np.nan           # rows 10, 11, 12
dirty_df.head(13)
```

```python
# Now isnull().any() shows which columns have missing values.
dirty_df.isnull().any(axis=0)
```

```python
type(dirty_df[dirty_df.isnull().any(axis=1)])
```

```python
# Show only rows with NaN. 6 rows have missing values.
dirty_df[dirty_df.isnull().any(axis=1)]
```

```python
# .info() shows the non-null counts — cholesterol and bp now have fewer than 421.
dirty_df.info()
```

```python
# Uncomment to see dropna() documentation:
# dirty_df.dropna?
```

```python
# dropna(axis=0): drop ROWS with any NaN. Just a view!
dirty_df.dropna(axis=1)
```

```python
# dropna(axis=1): drop COLUMNS that contain any NaN. Removes cholesterol and bp entirely!
dirty_df.dropna(axis=0)
```

```python
# dirty_df is STILL unchanged — dropna() returned views, not modifications.
len(dirty_df)
```

```python
# Avoid inplace=True — it's discouraged and may be deprecated.
# dirty_df.dropna(inplace=True)  # DON'T do this
```

```python
# Modify by assignment: drop rows with NaN.
dirty_df = dirty_df.dropna()
```

```python
# 6 rows dropped (3 with NaN cholesterol + 3 with NaN bp).
len(dirty_df)
```

```python
# The index is NOT reset — gaps at rows 0, 1, 2, 10, 11, 12.
# Always mind the index after dropping rows!
dirty_df.head(5)
```

```python
# reset_index() WITHOUT drop=True: the old index becomes a new column!
dirty_df = dirty_df.reset_index()  # try: .reset_index(drop=True) to discard the old index.
```

```python
# A new "index" column appeared — the old index values. Usually unwanted.
dirty_df.head(20)
```

```python
# Notice the "index" column at the left — old index values retained as data.
dirty_df.tail(5)
```

### 2.3 Drop a column by name

Use `.drop("column_name", axis=1)` to remove a column. `axis=1` means "operate on columns".

```python
# Drop the unwanted "index" column from dirty_df (created by reset_index).
dirty_df = dirty_df.drop("index", axis=1)
```

```python
# "index" column is gone. We continue with copy_df (the clean version) for the rest.
dirty_df.tail(2)
```

## 3. Replace values in a column, using a dictionary

Encoded values like `Sex=0/1` are common in medical datasets but not human-readable.  
We use `.replace()` with a dictionary to map coded values → meaningful labels.

[map or replace with a dict — StackOverflow](https://stackoverflow.com/a/49259581)

```python
copy_df.age.unique()
```

```python
# What unique values does the "sex" column have?
copy_df.sex.unique()  # 0 and 1 — not very informative!
```

```python
# Inline approach — works but harder to read with many replacements:
# copy_df.sex = copy_df.sex.replace({1: "Male", 0: "Female"})
```

```python
# Better: define the mapping as a separate dictionary.
sex_replacements = {1: "Male", 0: "Female"}

# Also prepare replacements for other encoded columns.
fbs_replacements = {1: "Yes", 0: "No"}
angina_replacements = {1: "Yes", 0: "No"}
```

```python
# Apply the replacements. This MODIFIES copy_df (assignment to column).
copy_df.sex = copy_df.sex.replace(sex_replacements)

copy_df.fbs_over_120 = copy_df.fbs_over_120.replace(fbs_replacements)

copy_df.exercise_angina = copy_df.exercise_angina.replace(angina_replacements)
```

```python
# Verify: sex now shows "Male" and "Female" instead of 1 and 0.
copy_df.sex.unique()
```

```python
# Much more readable! Sex, fbs_over_120, exercise_angina are now descriptive.
copy_df.sample(10)
```

## 4. Select data using boolean conditions

Boolean indexing: create a True/False mask, then use it to filter rows.  
This is one of the most powerful and frequently used pandas operations.

```python
# Step 1: Create a boolean mask — True where condition is met, False otherwise.
copy_df["heart_disease"] == "Presence"
```

```python
# Step 2: Use the mask to filter — only patients WITH heart disease.
copy_df[copy_df["heart_disease"] == "Presence"]
```

```python
# How many patients have heart disease?
len(copy_df[copy_df["heart_disease"] == "Presence"])
```

```python
# Same result with dot notation (works because "heart_disease" has no spaces).
copy_df[copy_df.heart_disease == "Presence"]
```

```python
# Numeric condition: patients older than 60.
copy_df[copy_df.age > 60]
```

```python
# Store filtered results. Notice: the index is NOT reset (gaps in row numbers).
df_sick = copy_df[copy_df["heart_disease"] == "Presence"]
```

```python
df_sick
```

### 4.1 The OR operator: `|`

Use OR to create a **union** — patients matching **any** of the conditions.

```python
(copy_df[
    (copy_df.age == 64)
    # | (copy_df.age == 44)df_male_sick
    | (copy_df.age == 54)
    | (copy_df.age == 34)
])
```

```python
# Patients with chest pain type 3 OR 4 (the more severe types).
# Each condition must be in parentheses when using | or &.
(copy_df[
    (copy_df.chest_pain_type == 3)
    # | (copy_df.chest_pain_type == 4)
])

# OR gives the UNION: rows matching EITHER condition.
```

### 4.2 The AND operator: `&`

Use AND to create an **intersection** — patients matching **all** conditions simultaneously.

```python
# Male patients with heart disease.
copy_df[
    (copy_df.sex == "Male")
    & (copy_df.heart_disease == "Presence")
]
```

```python
# Union: patients older than 65 OR with high cholesterol (> 300).
copy_df[
    (copy_df.age > 65)
    | (copy_df.cholesterol > 300)
]

# Union = "either or both" conditions are True.
```

```python
# Intersection: patients older than 55 AND with high blood pressure (> 140).
copy_df[
    (copy_df.age > 55)
    & (copy_df.bp > 140)
]

# Intersection = BOTH conditions must be True.
```

```python
# Store filtered result and reset the index.
df_male_sick = copy_df[
    (copy_df.sex == "Male")
    & (copy_df.heart_disease == "Presence")
].reset_index(drop=True)

df_male_sick
```

## 5. Set a column as index

```python
# Set "heart_disease" as the index (replaces the default integer index).
# This is a VIEW — does NOT modify copy_df.
copy_df.set_index("heart_disease")
```

```python
# copy_df is unchanged — set_index returned a view, not a modification.
copy_df
```

## 6. Group by a column or a list of columns

`groupby()` splits the DataFrame into groups, then applies an aggregation (e.g., `.size()`, `.mean()`).

```python
# Group by heart_disease → count patients in each category.
# .size() returns a Series. Suited for categorical columns.
by_disease = copy_df.groupby(
    by="heart_disease"
    ).size().sort_values(ascending=False)

by_disease
```

```python
# Group by heart_disease → count patients in each category.
# .size() returns a Series. Suited for categorical columns.
by_disease = copy_df.groupby(by="age").size()#.sort_values(ascending=False)

by_disease
```

```python
# Same result, more verbose — useful for debugging step by step.
by_disease = copy_df.groupby(by="heart_disease").size()
by_disease = by_disease.sort_values(ascending=False)

by_disease
```

```python
# .head(n) works on Series too.
by_disease.head(2)
```

```python
# groupby().size() returns a pandas Series, not a DataFrame.
type(by_disease)
```

```python
# Group by sex — how many Male vs Female patients?
by_sex = copy_df.groupby("sex").size().sort_values()

by_sex
```

```python
# Group by TWO columns: sex first, then heart_disease.
# The order of columns matters — it determines the hierarchy.
by_sex_disease = copy_df.groupby(
    ["sex", "heart_disease"]
    ).size().sort_values(ascending=False)

by_sex_disease
```

```python
# Reverse the column order: heart_disease first, then sex.
by_disease_sex = copy_df.groupby(
    by=["heart_disease", "sex"]
    ).size().sort_values(ascending=False)

by_disease_sex
```

```python
# Convert the Series to a DataFrame for easier manipulation.
by_sex_disease = by_sex_disease.to_frame()
by_sex_disease
```

```python
# Now it's a DataFrame, not a Series.
type(by_sex_disease)
```

#### Resources

- [pandas documentation — DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
- [pandas documentation — groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html)
- [pandas documentation — replace](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html)
- [Boolean indexing — pandas user guide](https://pandas.pydata.org/docs/user_guide/indexing.html#boolean-indexing)

```python
# Save a result to a CSV file. By default, the index is included as a column in the output.
by_sex_disease.to_csv("by_sex_disease.csv")
```

```python
# save copy_df to a CSV file. By default, the index is included as a column in the output.
copy_df.to_csv("copy_df.csv")
```

<!-- source: lectures_07_13_pandas_plots_scikit/lecture_07_pandas/practice_exercises/practice_lec_07a_join_dfs_solutions.ipynb @ 0cc874704aaa -->

Pandas provides several ways to join two DataFrames:

1. **Concatenation**: Concatenation basically glues the DataFrames together. You can select the axis (default is 0) according to your needs. This can be done using the `concat()` function.

2. **Appending**: Appending a DataFrame to another one is essentially concatenation, but the `append()` function is used instead.

3. **Merging**: Merging is a more powerful way to join two DataFrames, allowing you to join on any column(s) or index. This can be done using the `merge()` function.

4. **Joining**: Joining is a convenient method for combining the columns of two potentially differently-indexed DataFrames into a single result DataFrame. This is very similar to the `merge()` function, but the `join()` function is used instead.

Here are examples of each:

```python
import pandas as pd

# Assume we have two dataframes df1 and df2

# Concatenation
result = pd.concat([df1, df2])

# Appending
result = df1.append(df2)

# Merging
# Assume 'key' is a common column in df1 and df2
result = pd.merge(df1, df2, on='key')

# Joining
# Assume df1 and df2 have a common index
result = df1.join(df2, lsuffix='_df1', rsuffix='_df2')
```

Remember to choose the method that best fits your specific needs and constraints.

Sure, let's create two small pandas DataFrames with country names and their GDPs, and perform an inner join on them. 

Here's the Python code that follows this plan:

```python
import pandas as pd

# Create two small dataframes
df1 = pd.DataFrame({
    'Country': ['USA', 'China', 'Japan', 'Germany'],
    'GDP': [21.44, 15.42, 4.97, 3.86]  # in trillions of dollars
})

df2 = pd.DataFrame({
    'Country': ['Germany', 'India', 'UK', 'France'],
    'GDP': [3.86, 2.94, 2.83, 2.71]  # in trillions of dollars
})

# Perform an inner join on the 'Country' column
df_inner = pd.merge(df1, df2, on='Country', how='inner')

print(df_inner)
```

In this example, the `merge()` function is used to perform an inner join on the 'Country' column. The `how='inner'` argument specifies that we want an inner join. The resulting DataFrame, `df_inner`, contains only the rows with countries that are present in both `df1` and `df2`.

Sure, let's perform a left join on the two DataFrames. A left join returns all the rows from the left dataframe and the matched rows from the right dataframe. If there is no match, the result is `NaN` in the columns of the right dataframe.

Here's the Python code that follows this plan:

```python
# Perform a left join on the 'Country' column
df_left = pd.merge(df1, df2, on='Country', how='left')

print(df_left)
```

In this example, the `merge()` function is used to perform a left join on the 'Country' column. The `how='left'` argument specifies that we want a left join. The resulting DataFrame, `df_left`, contains all rows from `df1` and the matching rows from `df2`. If there is no match, the result is `NaN` in the columns of `df2`.

Sure, let's perform a right join on the two DataFrames. A right join returns all the rows from the right dataframe and the matched rows from the left dataframe. If there is no match, the result is `NaN` in the columns of the left dataframe.

Here's the Python code that follows this plan:

```python
# Perform a right join on the 'Country' column
df_right = pd.merge(df1, df2, on='Country', how='right')

print(df_right)
```

In this example, the `merge()` function is used to perform a right join on the 'Country' column. The `how='right'` argument specifies that we want a right join. The resulting DataFrame, `df_right`, contains all rows from `df2` and the matching rows from `df1`. If there is no match, the result is `NaN` in the columns of `df1`.

Sure, let's perform a full join (also known as an outer join) on the two DataFrames. A full join returns all the rows from both dataframes, and if there is no match, the result is `NaN` in the columns of the dataframe that does not have a match.

Here's the Python code that follows this plan:

```python
# Perform a full join on the 'Country' column
df_full = pd.merge(df1, df2, on='Country', how='outer')

print(df_full)
```

In this example, the `merge()` function is used to perform a full join on the 'Country' column. The `how='outer'` argument specifies that we want a full join. The resulting DataFrame, `df_full`, contains all rows from both `df1` and `df2`. If there is no match, the result is `NaN` in the columns of the dataframe that does not have a match.

There are four main types of joins that you can perform with pandas: inner, left, right, and full (or outer). We've already covered these in the previous examples. 

However, there are variations of these joins that you can perform based on multiple keys or conditions. For example, you can perform a join on multiple columns by passing a list of column names to the `on` parameter. 

Here's an example of a multi-key join:

```python
# Create two small dataframes with multiple keys
df1 = pd.DataFrame({
    'Country': ['USA', 'China', 'Japan', 'Germany'],
    'Year': [2020, 2020, 2020, 2020],
    'GDP': [21.44, 15.42, 4.97, 3.86]  # in trillions of dollars
})

df2 = pd.DataFrame({
    'Country': ['Germany', 'India', 'UK', 'France'],
    'Year': [2020, 2020, 2020, 2020],
    'Population': [83.02, 1380, 66.65, 67.06]  # in millions
})

# Perform an inner join on the 'Country' and 'Year' columns
df_inner = pd.merge(df1, df2, on=['Country', 'Year'], how='inner')

print(df_inner)
```

In this example, the `merge()` function is used to perform an inner join on the 'Country' and 'Year' columns. The resulting DataFrame contains only the rows with countries and years that are present in both `df1` and `df2`.

# Concatenate along the rows (axis=0)
df_concat_rows = pd.concat([df1, df2], axis=0)

print("Concatenation along rows:")
print(df_concat_rows)

# Concatenate along the columns (axis=1)
df_concat_cols = pd.concat([df1, df2], axis=1)

print("\nConcatenation along columns:")
print(df_concat_cols)

Sure, the `append()` function in pandas can be used to append rows of one DataFrame to the end of another DataFrame, resulting in a new DataFrame. It's essentially a shortcut to `concat()` along `axis=0`.

Here's the Python code that follows this plan:

```python
# Append df2 at the end of df1
df_append = df1.append(df2)

print(df_append)
```

In this example, the `append()` function is used to append `df2` at the end of `df1`. The resulting DataFrame, `df_append`, contains all rows from `df1` followed by all rows from `df2`.

```python
import pandas as pd

# Create two dataframes with no common columns or rows
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2']})

df2 = pd.DataFrame({'C': ['C0', 'C1', 'C2'],
                    'D': ['D0', 'D1', 'D2']})

# Introduce a temporary key for both dataframes
df1['key'] = 0
df2['key'] = 0

# Perform the merge, which results in a Cartesian product
df_cross = pd.merge(df1, df2, on='key')

# Drop the temporary key column
df_cross.drop('key', axis=1, inplace=True)

print(df_cross)
```

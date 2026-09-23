<!-- source: lectures_07_13_pandas_plots_scikit/lecture_07_pandas/practice_exercises/practice_lec_07a_join_dfs.ipynb @ 0cc874704aaa -->

# Lecture 07 practice exercise

**REQUIRES ADDITIONAL READING OF METHODS NOT PRESENTED IN CLASS**  

Pandas provides several ways to join DataFrames:

1. **Concatenation**: Concatenation basically glues the DataFrames together.  
   Concatenation is almost a synonym for addition.  
   You can select the axis (default is 0) according to your needs.  
   This can be done using the `concat()` function.

3. **Appending**: Appending a DataFrame to another one is essentially concatenation, but the `append()` function is used instead.  
   It's essentially a shortcut to `concat()` along `axis=0`.

5. **Merging**: Merging is a more powerful way to join two DataFrames, allowing you to join using as criteria any column(s) or index.
   This can be done using the `merge()` function.

7. **Joining**: Joining is a convenient method for combining the columns of two potentially differently-indexed DataFrames into a single result DataFrame.   
Similar to the `merge()` function, but the `join()` function is used instead.

Examples:

```python
import pandas as pd

# Assume we have two dataframes df1 and df2

# Concatenation
result = pd.concat([df1, df2])

# Appending
result = df1.append(df2)

# Merging
# Assume 'key_column_name' is a common column ΝΑΜΕ in df1 and df2
result = pd.merge(df1, df2, on='key_column_name')

# Joining
# Assume df1 and df2 have a common index
result = df1.join(df2, lsuffix='_df1', rsuffix='_df2')
```

## 1) Practice on df merging

```python
import pandas as pd

# Two dataframes with countries and their GDP.
# Create two dfs using the DataFrame function and dictionary notation.

df1 = pd.DataFrame({
    'Country': ['USA', 'China', 'Japan', 'Germany'],
    'GDP': [21.44, 15.42, 4.97, 3.86]  # In trillions of dollars.
})

df2 = pd.DataFrame({
    'Country': ['Germany', 'India', 'UK', 'France'],
    'GDP': [3.86, 2.94, 2.83, 2.71]  # In trillions of dollars.
})
```

```python
# Perform an inner join on the 'Country' column
```

```python
# Perform a left join on the 'Country' column
```

```python
# Perform a right join on the two DataFrames.
```

```python
# Perform a full join (also known as an outer join) on the two DataFrames.
```

```python
# Concatenate along the rows (axis=0)
```

```python
# Concatenate along the columns (axis=1)
```

```python
# Append df2 at the end of df1
```

## 2) Advanced practice on a multi-key join:

```python
# Two dataframes with countries, their GDP, the year and the population only for some countries.
# Create two dfs using the DataFrame function and dictionary notation.

df1 = pd.DataFrame({
    'Country': ['USA', 'China', 'Japan', 'Germany', 'UK'],
    'Year': [2020, 2020, 2020, 2020, 2020],
    'GDP': [21.44, 15.42, 4.97, 3.86, 2.83]  # in trillions of dollars
})

df2 = pd.DataFrame({
    'Country': ['Germany', 'India', 'UK', 'France'],
    'Year': [2020, 2020, 2020, 2020],
    'Population': [83.02, 1380, 66.65, 67.06]  # in millions
})
```

```python
# Perform an inner join on the 'Country' and 'Year' columns.
```

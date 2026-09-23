<!-- source: lectures_07_13_pandas_plots_scikit/lecture_08_EDA_plots/reading_material/lec_08d_example_EDA_iris.ipynb @ 0cc874704aaa -->

# Lecture 08d: Exploratory Data Analysis on the 'iris' dataset

### A refactoring of a nice blog that could be improved a lot.
[this blog post](https://medium.com/codex/machine-learning-k-nearest-neighbors-algorithm-with-python-df94b374ad41)

### The *iris* plants dataset, loaded from seaborn
[iris flowers](https://en.wikipedia.org/wiki/Iris_(plant))
[iris setosa](https://en.wikipedia.org/wiki/Iris_setosa)
[Iris Versicolour](https://en.wikipedia.org/wiki/Iris_versicolor)
[Iris Virginica](https://en.wikipedia.org/wiki/Iris_virginica)

[Info about dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.names)

### Import necessary modules.

```python
# pandas: the main library for tabular data manipulation in Python
import pandas as pd
# numpy: for numerical operations (arrays, math functions)
import numpy as np
# seaborn: a high-level plotting library built on top of matplotlib
import seaborn as sns

# matplotlib: the foundational Python plotting library
import matplotlib.pyplot as plt
from matplotlib import style
```

## Examine the data, using graphs (EDA) and statistics.

```python
# seaborn comes with built-in practice datasets — very handy for learning.
# load_dataset() downloads and returns a pandas DataFrame.
df = sns.load_dataset('iris')
```

```python
type(df)
```

```python
# seaborn loads training datasets and returns a pandas dataframe which contains X and y.
df.sample(5)
```

### 1. Basic info about the data.

```python
df.columns
```

```python
df.info()
```

```python
# .describe() gives count, mean, std, min, 25%, 50%, 75%, max for each numeric column.
# WARNING: Running describe() on ALL the data offers no insights about the 3 different
# species — it mixes them all together! We fix this in the next cell with groupby().
df.describe()
```

```python
# Much better! groupby("species").describe() gives separate stats for each species.
# .T transposes the table so species become columns — easier to compare side by side.
df.groupby("species").describe().T
```

### 2. Various plots for EDA

```python
# Matplotlib styles
style.available
```

```python
# Set a consistent visual style for all matplotlib/seaborn plots in this notebook.
# rcParams lets you set global defaults like figure size.
style.use("seaborn-v0_8-whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)  # width=14 inches, height=8 inches
```

#### A2. Sepal scatter visualization

```python
# sns.scatterplot?
```

```python
# Write functions with many parameters using one line for each parameter: Readability and Flexibility. Faster, cleaner coding.
sns.scatterplot(data=df,
                x='sepal_length',
                y='sepal_width',
                hue='species',
                palette=['Red', 'Blue', 'Limegreen'],
                # palette = 'Set2', # Try this one.
                edgecolor='black',
                s=100,
                alpha=0.85  # try setting 0.5
)

plt.title('Sepal Length and Sepal width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend(markerscale=1.5, loc="upper right", prop={'size': 14});  # Try different marker scales.

# plt.savefig("sepal.png")  # Can you guess what does this line of code do?
```

```python
# Spot the three differences in the code between this plot and the one above:
# a)
# b)
# c)

# Write functions with many parameters using one line for each parameter: Readability and Flexibility. Faster, cleaner coding.
sns.scatterplot(
    x='sepal_length',
    y='sepal_width',
    data=df,  # a)
    hue='species',
    palette=['red', 'blue', 'limegreen'],  # b)
    edgecolor='b', # c) Can you guess if this is "black" or "blue"? Explicit is better than implicit.
    s=100,
    alpha=0.85
)

plt.title('Sepal Length to Sepal width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend(markerscale=1.5, loc="upper right", prop={'size': 14});

# plt.savefig('sepal.png')  # Can you guess what does this line of code do?
```

#### A3. Petal scatter visualization

```python
sns.scatterplot(
    x='petal_length', y='petal_width', data=df, hue='species',
    palette = ['Red', 'Blue', 'limegreen'],
    edgecolor = 'w', s = 150, alpha = 0.7
)

plt.title('Petal Length To Petal Width')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc = 'upper left', fontsize = 12);

# plt.savefig('petal.png')
```

```python
# sns.heatmap?
```

#### A4. Data Heatmap

```python
# A correlation heatmap shows how strongly each pair of numeric features is related.
# Values close to +1 or -1 mean strong correlation; close to 0 means weak/no correlation.
# numeric_only=True is required since pandas 2.0 (it won't silently ignore non-numeric columns).
df_corr = df.corr(numeric_only=True)

sns.heatmap(
    df_corr, annot=True, cmap='Blues',
    xticklabels=df_corr.columns.values,
    yticklabels=df_corr.columns.values
);

plt.title('Iris Data Heatmap', fontsize=15);
plt.xticks(fontsize=12);
plt.yticks(fontsize=12);

# plt.savefig('heatmap.png')  # Uncomment to save the plot as an image file
```

#### A5. Scatter Matrix

## Examine possibilities for feature engineering.
If new features are Linear combinations of existing features, can't have both in linear models.

petal lenght to sepal lenght.
petal width to sepal width.
Approximate area of sepal and pedal (sq.cm.)? Width * length

```python
# A pair plot creates scatter plots for every pair of numeric features,
# with histograms on the diagonal. Coloring by 'species' reveals clusters.
# This is one of the most powerful EDA tools for spotting patterns.
sns.pairplot(data=df, hue='species')#, palette=['Red', 'Blue', 'limegreen']);

# plt.savefig('iris_pairplot.png')
```

```python
df["approx_sepal_area"] = df.sepal_length * df.sepal_width
df["approx_petal_area"] = df.petal_length * df.petal_width
df["approx_total_area"] = df.approx_petal_area * df.approx_sepal_area
```

```python
# use the species to 
sns.pairplot(data=df[["approx_sepal_area", "approx_petal_area", "approx_total_area", 'species']], hue='species', palette=['Red', 'Blue', 'limegreen'], height=8);
```

```python
df["approx_sepal_ratio"] = df.sepal_length / df.sepal_width
df["approx_petal_ratio"] = df.petal_length / df.petal_width
df["approx_total_ratio"] = df.approx_petal_area / df.approx_sepal_area
```

```python
# use the species to 
sns.pairplot(data=df[["approx_sepal_ratio", "approx_petal_area", "approx_total_area", 'species']], hue='species', palette=['Red', 'Blue', 'limegreen'], height=8);
```

```python
# 5. Distribution plot for all species of iris

ax1 = plt.subplot(211)
sns.kdeplot(df['sepal_length'], color = 'r', shade = True)
sns.kdeplot(df['sepal_width'], color = 'b', shade = True)
plt.xlabel('Sepal width and sepal length');
plt.legend(["Length", "Width"])


ax2 = plt.subplot(212)
sns.kdeplot(df['petal_length'], color = 'coral', shade = True)
sns.kdeplot(df['petal_width'], color = 'green', shade = True);
plt.xlabel('Petal width and sepal length');
plt.legend(["Length", "Width"])

# plt.savefig('dist.png')
```

## B. Examine outliers
BOXPLOTS

`

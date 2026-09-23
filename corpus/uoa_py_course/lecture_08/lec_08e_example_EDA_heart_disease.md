<!-- source: lectures_07_13_pandas_plots_scikit/lecture_08_EDA_plots/reading_material/lec_08e_example_EDA_heart_disease.ipynb @ 0cc874704aaa -->

# Lecture 08e: Exploratory Data Analysis — Heart Disease Dataset

In this notebook we perform a **comprehensive EDA** on a real-world medical dataset.  
The goal is to understand the data *before* building any Machine Learning model.

### Why EDA matters
- Spot **data quality issues** (missing values, duplicates, incorrect types).
- Understand **distributions** — are features normally distributed? Skewed?
- Find **relationships** between features and the target variable.
- Generate **hypotheses** that guide feature engineering and model choice.

### Dataset: Predict Heart Disease
Each row represents a patient. Features include age, sex, blood pressure, cholesterol,
and various cardiac test results. The target column indicates **Presence** or **Absence**
of heart disease.

| Feature | Description |
|---|---|
| Age | Patient age in years |
| Sex | 1 = male, 0 = female |
| Chest pain type | 1–4 (typical angina, atypical, non-anginal, asymptomatic) |
| BP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FBS over 120 | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| EKG results | 0, 1, or 2 (normal, ST-T abnormality, LV hypertrophy) |
| Max HR | Maximum heart rate achieved |
| Exercise angina | Exercise-induced angina (1 = yes, 0 = no) |
| ST depression | ST depression induced by exercise relative to rest |
| Slope of ST | 1 = upsloping, 2 = flat, 3 = downsloping |
| Number of vessels fluro | Number of major vessels (0–3) colored by fluoroscopy |
| Thallium | 3 = normal, 6 = fixed defect, 7 = reversible defect |
| Heart Disease | **Target**: Presence or Absence |

---
## 0. Setup — Import Libraries

```python
# pandas: the main library for working with tabular data in Python
import pandas as pd

# numpy: for numerical calculations (we'll use it for a few things)
import numpy as np

# seaborn: high-level statistical plotting built on matplotlib
import seaborn as sns

# matplotlib: the foundational Python plotting library
import matplotlib.pyplot as plt

# plotly express: one-line interactive plots
import plotly.express as px

# Suppress FutureWarnings for cleaner output
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

```python
# Set a clean visual style for all seaborn/matplotlib plots in this notebook.
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)  # Default figure size: 12 inches wide, 6 tall
```

---
## 1. Load & First Look at the Data

```python
# Read the CSV file into a pandas DataFrame.
# The dataset is in the same folder as this notebook.
df = pd.read_csv("predict_heart_disease_train.csv")

# .shape returns (rows, columns) — a quick sanity check.
print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
```

```python
# .head() shows the first 5 rows — always the first thing to do after loading.
df.head()
```

```python
# .sample() shows random rows — useful to see variety in the data.
# random_state ensures everyone gets the same "random" sample (reproducibility).
df.sample(5, random_state=42)
```

```python
# .info() shows column names, data types, and non-null counts.
# This is ESSENTIAL for spotting:
#   - wrong data types (e.g., a number stored as text)
#   - missing values (non-null count < total rows)
df.info()
```

```python
# Check for missing values in each column.
# A clean dataset has 0 missing values everywhere.
df.isnull().sum()
```

```python
# Check for duplicate rows. Duplicates can bias our analysis.
n_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {n_duplicates:,}")
```

```python
# The 'id' column is just a row identifier —data=presence_df, it carries no predictive information.
# We drop it to keep our analysis clean.
df = df.drop(columns=["id"])
```

---
## 2. Descriptive Statistics

Before making any plots, let's look at the **numbers**.

```python
# .describe() gives count, mean, std, min, 25%, 50% (median), 75%, max.
# This tells us the central tendency and spread of each numeric feature.
df.describe().round(2)
```

```python
# How balanced is our target variable?
# A heavily imbalanced target can mislead models and metrics.
print("Target distribution:")
print(df["Heart Disease"].value_counts())
print(f"\nPercentages:")
print(df["Heart Disease"].value_counts(normalize=True).round(3) * 100)
```

```python
# Compare descriptive stats between patients WITH and WITHOUT heart disease.
# This is much more informative than looking at the overall stats.
# .T transposes the table so groups become columns — easier to compare.
df.groupby("Heart Disease").describe().T
```

**Key observations from descriptive stats:**
- Compare the mean Age, BP, Cholesterol, Max HR between Presence and Absence groups.
- Which features show the biggest differences between the two groups?

---
## 3. Target Variable — Heart Disease Distribution

```python
# A count plot shows how many rows fall into each category.
# This is the first plot to make for any classification problem.
fig = px.histogram(
    df,
    x="Heart Disease",
    color="Heart Disease",
    color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"},
    title="Heart Disease: Presence vs Absence",
    labels={"count": "Number of Patients"},
    text_auto=True  # Show count numbers on bars
)
fig.update_layout(showlegend=False)
fig.show()
```

---
## 4. Univariate Analysis — One Feature at a Time

We examine the distribution of each feature individually.

### 4.1 Age Distribution

```python
# A histogram shows how values are distributed across ranges (bins).
# Adding color='Heart Disease' reveals differences between the two groups.
# marginal='box' adds a box plot above, showing median, quartiles, and outliers.
fig = px.histogram(
    df,
    x="Age",
    color="Heart Disease",
    color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"},
    marginal="box",
    nbins=30,  # Number of bins in the histogram
    barmode="overlay",  # Overlap the two distributions to compare
    opacity=0.7,
    title="Age Distribution by Heart Disease Status"
)
fig.show()
```

### 4.2 Blood Pressure & Cholesterol

```python
# Box plots are perfect for comparing distributions across groups.
# They show: median (line), interquartile range (box), whiskers, and outliers (dots).
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=df, x="Heart Disease", y="BP",
            palette={"Presence": "#e74c3c", "Absence": "#2ecc71"},
            ax=axes[0])
axes[0].set_title("Blood Pressure by Heart Disease Status")

sns.boxplot(data=df, x="Heart Disease", y="Cholesterol",
            palette={"Presence": "#e74c3c", "Absence": "#2ecc71"},
            ax=axes[1])
axes[1].set_title("Cholesterol by Heart Disease Status")

plt.tight_layout()  # Prevents overlapping labels
plt.show()
```

### 4.3 Maximum Heart Rate

```python
# Violin plots combine box plots with KDE (kernel density estimation) —
# showing the distribution shape, not just summary statistics.
# The wider parts indicate where more data points are concentrated.
fig = px.violin(
    df,
    y="Max HR",
    x="Heart Disease",
    color="Heart Disease",
    color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"},
    box=True,  # Add a miniature box plot inside the violin
    points="outliers",  # Show outlier points
    title="Max Heart Rate by Heart Disease Status"
)
fig.show()
```

### 4.4 Categorical Features — Chest Pain Type, Sex, Thallium

```python
# For categorical features, we use count plots (bar charts) to see
# how each category relates to the target variable.

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Chest pain type
sns.countplot(data=df, x="Chest pain type", hue="Heart Disease",
              palette={"Presence": "#e74c3c", "Absence": "#2ecc71"},
              ax=axes[0])
axes[0].set_title("Chest Pain Type")
axes[0].set_xlabel("Type (1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic)")

# Sex
sns.countplot(data=df, x="Sex", hue="Heart Disease",
              palette={"Presence": "#e74c3c", "Absence": "#2ecc71"},
              ax=axes[1])
axes[1].set_title("Sex (0=Female, 1=Male)")

# Thallium test result
sns.countplot(data=df, x="Thallium", hue="Heart Disease",
              palette={"Presence": "#e74c3c", "Absence": "#2ecc71"},
              ax=axes[2])
axes[2].set_title("Thallium (3=normal, 6=fixed, 7=reversible)")

plt.tight_layout()
plt.show()
```

---
## 5. Bivariate Analysis — Relationships Between Features

Now we look at how pairs of features relate to each other and to the target.

### 5.1 Correlation Heatmap

```python
# Create a binary version of the target for correlation analysis.
# We need numbers (0/1) instead of text (Absence/Presence) for correlation.
df["Heart Disease Binary"] = (df["Heart Disease"] == "Presence").astype(int)

# Compute the correlation matrix — how each pair of numeric columns moves together.
corr_matrix = df.select_dtypes(include=[np.number]).corr()

# Create a mask for the upper triangle to avoid showing duplicate info.
# np.triu creates an upper-triangular matrix of True values.
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    mask=mask,           # Hide the upper triangle
    annot=True,          # Show correlation values in each cell
    fmt=".2f",           # 2 decimal places
    cmap="RdBu_r",       # Red = positive, Blue = negative correlation
    center=0,            # Center the colormap at 0
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5
)
plt.title("Correlation Heatmap — Heart Disease Dataset", fontsize=14)
plt.tight_layout()
plt.show()
```

```python
# Which features correlate most with Heart Disease?
# Sort by absolute correlation to find the strongest predictors.
target_corr = corr_matrix["Heart Disease Binary"].drop("Heart Disease Binary").abs().sort_values(ascending=False)
print("Features ranked by absolute correlation with Heart Disease:\n")
print(target_corr.round(3))
```

### 5.2 Scatter Plots — Continuous Feature Pairs

```python
# Scatter plot of Age vs Max Heart Rate, colored by Heart Disease.
# This reveals whether the relationship between two features differs
# for patients with and without heart disease.
fig = px.scatter(
    df,
    x="Age",
    y="Max HR",
    color="Heart Disease",
    color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"},
    opacity=0.5,  # Transparency helps when points overlap
    title="Age vs Maximum Heart Rate by Heart Disease",
    marginal_x="histogram",  # Add histogram on x-axis margin
    marginal_y="histogram",  # Add histogram on y-axis margin
)
fig.show()
```

```python
# Another insightful scatter: ST depression vs Max HR
fig = px.scatter(
    df,
    x="ST depression",
    y="Max HR",
    color="Heart Disease",
    color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"},
    opacity=0.4,
    title="ST Depression vs Max Heart Rate",
    marginal_x="box",
    marginal_y="box",
)
fig.show()
```

### 5.3 Pair Plot — All Numeric Features at Once

```python
# A pair plot creates scatter plots for every combination of selected features.
# We pick a subset of the most informative features to keep it readable.
# WARNING: With many features, pair plots become very large and slow.
selected_features = ["Age", "BP", "Cholesterol", "Max HR", "ST depression", "Heart Disease"]

sns.pairplot(
    df[selected_features],
    hue="Heart Disease",
    palette={"Presence": "#e74c3c", "Absence": "#2ecc71"},
    diag_kind="kde",  # KDE curves on diagonal instead of histograms
    plot_kws={"alpha": 0.4, "s": 10},  # Small transparent dots
    height=2.5
);
plt.suptitle("Pair Plot of Key Features", y=1.02, fontsize=14)
plt.show()
```

---
## 6. Advanced Visualizations

### 6.1 Parallel Coordinates — See All Features at Once

Each vertical axis is a feature. Each line is a patient.
Color shows heart disease status. Patterns (clusters of lines) reveal
which feature ranges are associated with disease presence.

```python
# Parallel coordinates are excellent for spotting multi-dimensional patterns.
fig = px.parallel_coordinates(
    df,
    dimensions=["Age", "BP", "Cholesterol", "Max HR", "ST depression", "Number of vessels fluro"],
    color="Heart Disease Binary",
    color_continuous_scale=["#2ecc71", "#e74c3c"],
    title="Parallel Coordinates — Heart Disease Features",
    height=500
)
fig.show()
```

### 6.2 Parallel Categories — Categorical Feature Flows

Shows how patients flow through different categorical feature combinations.

```python
# Parallel categories show flows between categorical variables.
# Width of each band = number of patients following that path.
fig = px.parallel_categories(
    df,
    dimensions=["Sex", "Chest pain type", "FBS over 120", "Exercise angina", "Heart Disease"],
    color="Heart Disease Binary",
    color_continuous_scale=["#2ecc71", "#e74c3c"],
    title="Patient Flow Through Categorical Features",
    height=500
)
fig.show()
```

### 6.3 Sunburst — Hierarchical View of Categorical Features

```python
# Sunburst shows part-to-whole relationships in a nested ring structure.
# We visualize: Heart Disease → Sex → Chest Pain Type
fig = px.sunburst(
    df,
    path=["Heart Disease", "Sex", "Chest pain type"],
    color="Heart Disease",
    color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"},
    title="Heart Disease by Sex and Chest Pain Type",
    height=500
)
fig.show()
```

---
## 7. Feature Distributions — All at Once

Let's create a grid of histograms for all numeric features, split by Heart Disease.

```python
# Select only the numeric columns (excluding the binary target we created).
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("Heart Disease Binary")

# Create a grid of subplots — one histogram per feature.
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
axes = axes.flatten()  # Convert 2D array of axes to 1D for easy iteration

for i, col in enumerate(numeric_cols):
    # KDE (Kernel Density Estimation) shows a smooth distribution curve.
    # It's like a smoothed histogram — easier to compare two groups.
    for label, color in [("Absence", "#2ecc71"), ("Presence", "#e74c3c")]:
        subset = df[df["Heart Disease"] == label][col]
        axes[i].hist(subset, bins=30, alpha=0.5, color=color, label=label, density=True)
    axes[i].set_title(col, fontsize=12)
    axes[i].legend(fontsize=8)

# Hide unused subplots (if the grid has more slots than features)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Distributions by Heart Disease Status", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
```

---
## 8. Outlier Detection with Box Plots

```python
# Box plots reveal outliers (dots beyond the whiskers).
# Outliers can be genuine extreme values or data entry errors.

continuous_features = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]

fig, axes = plt.subplots(1, len(continuous_features), figsize=(20, 5))

for i, col in enumerate(continuous_features):
    sns.boxplot(
        data=df, x="Heart Disease", y=col,
        palette={"Presence": "#e74c3c", "Absence": "#2ecc71"},
        ax=axes[i]
    )
    axes[i].set_title(col)

plt.suptitle("Box Plots — Outlier Detection", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

---
## 9. Interactive Exploration with Plotly

```python
# An interactive scatter matrix using plotly — hover over any point to see
# the patient's details across all selected dimensions.
fig = px.scatter_matrix(
    df,
    dimensions=["Age", "BP", "Max HR", "ST depression"],
    color="Heart Disease",
    color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"},
    opacity=0.3,
    title="Interactive Scatter Matrix",
    height=700,
    width=900
)
fig.update_traces(diagonal_visible=False)  # Hide diagonal (self vs self)
fig.show()
```

---
## 10. Key EDA Takeaways

Summarize your findings before moving to modeling:

1. **Target balance**: Is the dataset balanced or imbalanced?
2. **Strong predictors**: Which features show the biggest differences between Presence and Absence?
3. **Correlations**: Which features are correlated with each other? (Multicollinearity matters for some models.)
4. **Outliers**: Are there suspicious values that might need cleaning?
5. **Feature types**: Which features are truly continuous vs. categorical encoded as numbers?

## 11. Practice Exercises

1. Create a violin plot for `Cholesterol` grouped by both `Sex` and `Heart Disease`.
2. Make a faceted histogram of `Age` with one subplot per `Chest pain type`.
3. Build a `px.scatter_3d()` plot with Age, Max HR, and ST depression as the three axes.
4. Calculate and plot the heart disease prevalence rate by age group (e.g., 30–39, 40–49, etc.).

```python
# Clean up: drop the binary helper column we created
df = df.drop(columns=["Heart Disease Binary"])
```

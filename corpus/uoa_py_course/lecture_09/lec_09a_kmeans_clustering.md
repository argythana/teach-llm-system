<!-- source: lectures_07_13_pandas_plots_scikit/lecture_09_clustering_deploy_hf_app/reading_material/lec_09a_kmeans_clustering.ipynb @ 0cc874704aaa -->

# Lecture 09a. Clustering: `KMeans` implementation from `scikit-learn`

## A real problem KMeans solves

Imagine you run a chain of shops and you have **one row per customer** with their age, income, and how much they spend with you each month. You don't have any labels — no "good customer / bad customer" tag — just numbers. You want to send a personalised mailer to four or five distinct *kinds* of shopper, but you have no idea what those kinds are. **KMeans answers exactly this question**: given the rows, find the K natural clusters that minimise within-cluster variance, and tell you which row belongs to which cluster. This whole lecture is built around that workflow on a real mall-customers dataset.

## Where this technique shows up in industry

- **Retail / marketing** — *customer segmentation* for targeted campaigns. Group millions of shoppers into a handful of personas using RFM-style features (recency / frequency / monetary value) so each persona gets the right offer. The exercise we run on `mall_customers.csv` is a tiny version of this.
- **Image processing** — *colour quantization*. Reduce a 24-bit image to a 16- or 256-colour palette by clustering pixel RGB values. This is how GIF compression and many old-school dithering algorithms work, and it is still useful today for thumbnails and low-bandwidth previews.
- **Operations & security** — *anomaly detection*. Fit KMeans on "normal" system telemetry, then flag any new observation whose distance to the nearest centroid is suspiciously large. A common building block in fraud-detection and infrastructure-monitoring stacks (often paired with an LSTM or an autoencoder downstream).


## How this lecture is built

The nine sections below form **one continuous story**, climbing in three deliberate stages:

- **Intro (§1–§3) — get to know the data.** Set up the environment (§1), load the mall-customers dataset (§2), then describe and explore it visually (§3). By the end of §3 you can describe the four features and can already eyeball ~5 natural groupings in the scatter plots.
- **Development (§4–§7) — the KMeans workflow, increasing complexity step by step.** Apply the estimator API to two features with a guessed `K = 5` (§4), then pick `K` from the *data* using the elbow + silhouette metrics (§5). Add two more features for a four-dimensional fit at `K = 6` (§6), then turn the integer cluster IDs into human-readable personas (§7). Each step builds directly on the previous one's output.
- **Recap (§8–§9) — pointers and career framing.** Close with further-reading links (§8) and a final beat on where KMeans actually shows up on the job (§9), tying back to the three industry uses named above.

The notebook is dense (~92 code cells, ~3 h fresh walk-through). The marked **⏸ Pause points** and the one **⏭ Skip if running long** callout let the teacher adapt the in-class pace; the three optional notebooks (`lec_09d` parameters, `lec_09e` algorithm-intuition animations, `lec_09f` other clustering algorithms) sit downstream for students who want career-track depth, and are self-contained — they load their own data, so they can be read in any order after this notebook.

## Before you start

This lecture assumes you can:

- Load a CSV into a `pandas` DataFrame and slice it (**Lecture 07**).
- Make a 2D and a 3D `plotly.express` scatter plot (**Lecture 08** — `lec_08a` / `lec_08c`).
- Stand up a local `gr.Interface(...).launch()` Gradio demo (**Lecture 08g**).
- Read and edit a `requirements.txt` and use `pip install -r` (**Lecture 03**).

If any of those are rusty, do those first — this lecture builds on all of them.

## Preliminary Definitions

* **Algorithm**: a finite, step-by-step procedure that transforms inputs into outputs to solve a problem.
* **Cluster**: a group of observations that are more similar to each other than to observations in other groups.
* **Unsupervised learning**: learning patterns from data **without** labelled targets (no `y`); the algorithm discovers structure on its own.

## What is KMeans

Introduction to `KMeans` by [my favourite tutor.](https://www.youtube.com/watch?v=lQ39ZRFfYbI)

`KMeans` is an "unsupervised learning" algorithm → We don't know the labels. We assign intuitive labels according to the data.   
We don't have a "y" target variable that we want to predict.  

The goal is to create K groups (clusters) of homogeneous observations.   
The number K of groups is unknown. The labels of the groups are unknown.  
A more accurate description could be "unlabelled learning".

Using what the data suggest, we choose the number of groups, and then we think of appropriate and intuitive labels to describe each cluster (group).     

Centroids: the centre of the cluster. Maybe an actual observation in the data or not.  

Measuring success:   
a) minimise the total distance of all observations from the centroid. This metric is called "inertia" or "**Within Cluster Sum of Squares (WCSS)**",    
b) maximise cluster "cohesion", how similar are the observations in a cluster compared to other clusters (Silhouette Score).   
c) for all available metrics on scikit-learn please read the relevant section.  

Important note:   
The number of groups (clusters) in social sciences is in many cases a matter of policy, a strategic decision based on possible insights of the data.   
It depends on how much we want to "zoom out" or "zoom in" the data — argythana(tm).      
The number of clusters in positive science is in most cases predetermined. E.g. Fraud, Measurement error.

We may examine and report about the data for different numbers of clusters depending on our needs.   
In most cases, we start from 2 clusters and increase K up to a reasonable point.    
Often, there is no "correct" number of clusters.  

Actually, "unsupervised" learning ["is somewhat supervised"](https://www.youtube.com/watch?v=JbP9EPPvVXg) to an extent.   
I would add that clustering is similar to classification of observations to "unknown classes".
These new "classes" can in turn be labelled according to users' needs.  

Clustering can be thought of as trying to "classify" to unknown categories (classes), when we don't know the existing classes of the observations.  

## When KMeans is a good fit (shape and data size) and what to watch out for

- KMeans works best when the clusters are roughly **spherical (round "blobs") of similar size and density**, on **numeric features scaled to comparable ranges** (e.g. standardised). It struggles when clusters are elongated, curved, ring-shaped, or have very different sizes — in those cases consider DBSCAN, spectral clustering, or Gaussian Mixture Models. More reading on this in the dedicated notebook.

- KMeans is also **sensitive to outliers**, because a single distant point can pull a centroid noticeably. 
- Categorical features must be encoded numerically (as we do later with `Genre` → `Gender`).

For more, examine the [comparison of various clustering algorithms](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html).

## Is KMeans compute-intensive?
 
For typical datasets, **no** — it is one of the fastest clustering algorithms. Each iteration costs roughly `O(n × k × d)` (number of points × number of clusters × number of features), and it usually converges in a small number of iterations. A laptop can comfortably cluster hundreds of thousands of points with a handful of clusters in seconds. For very large data (millions of rows), scikit-learn provides `MiniBatchKMeans`, which trades a tiny bit of accuracy for a big speed-up by fitting on small random batches.

```python
# Uncomment the line below to read the documentation about KMeans.
# KMeans?
```

## Lecture resources

- [Tutorial on Kmeans from kaggle](https://www.kaggle.com/satishgunjal/tutorial-k-means-clustering), improvements, additions by Thanasis Argyriou
- [The mall customers dataset, source: Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [Kmeans implementation from scikit learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [What is scikit learn](https://scikit-learn.org/stable/getting_started.html)

## 1. Import the necessary modules.   

Make sure you activate the virtual environment and install the necessary packages first.  

The new package that we install in this lecture is `scikit-learn`.

```python
# python -m pip install -U scikit-learn  # -U means --upgrade if already installed
```

```python
# data management libraries
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# interactive visualizarion libaries
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# scikit learn clustering library
from sklearn.cluster import KMeans
# import the silhouette metric
from sklearn.metrics import silhouette_score
```

## 2. Load the data using the correct path

```python
# If your notebook is in the same folder as the data file just use the files name.
df = pd.read_csv("mall_customers.csv")
```

```python
# If you data is inside a folder called data, two folders above your current working folder:
# You get extra points for that in the final assignment

#df = pd.read_csv("../../data/mall_customers.csv")

# Or use an absolute path, that is too ugly, and will NOT work for you because you a different user. You are not tharg.
#df = pd.read_csv("C:\\Users\\tharg\\uoa_py_course\\data\\mall_customers.csv")
```

### Understand the data features.   
See which are the variables, what is their type, what are the values that the variables take.  
At this step, you should think about possible relations that you ought to examine.   
Which do you think might be more important?

```python
# show first lines of dataset
df.head(3)
```

```python
# show column names of dataset.
# 5 columns with suboptimal names.
df.columns
```

```python
# show data types of dataset.
# four numeric, one categorical variable type.
df.dtypes
```

```python
# simple commnand to show column names and data types.
df.info()
```

### Short summary notes about the variables.

CustomerID is not useful, unless you know "who is who".  

Spending score is what makes a huge difference. How could we group the customers, according to their spending score, gender, income, age?   

The second most important variable, in this case, should be annual income.  

Gender might, or might might not make a difference.  

Age is a tricky variable. Depending in the issue in question, it might be in linear correlation with the other variables, or it might grow and then gradually fall in importance. e.g. The possibility of deseases and age should be correlated and should grow as age grows.   

On the other hand, income could grow as age grows but then, after retirement, as age grows income might fall. The same line of thought might apply to age and spending score.

## 3. Descriptive statistics and Exploratory Data Analysis (EDA)

### 3.1. Descriptive Statistics

```python
# get rid of customer ID column, useless in this case.
df = df.drop("CustomerID", axis=1)  # axis=1 means drop a column, axis=0 would mean drop a row.
```

```python
# very basic Descriptive Statistics
df.describe()  # show only numeric variables
```

```python
# df.describe(include='all')
```

```python
# Descriptive stats for categorical variables.
df.describe(include=object)

# alternative way to desribe one single column.
#df.Genre.describe()
```

```python
# count number of observations in Genre column.
df.Genre.value_counts()
```

```python
# number of observations in Genre column, in percentage.
df["Genre"].value_counts(normalize=True)
```

### Short summary of descriptive statistics about the data.

200 observations.  
Younger is 18, older is 70, mean age is 38.85 (whatever that age means).     
Highest income is 137k, lowest is 15k, mean income is 60.5k.  
The gender distribution is slightly skewed towards "Female", with 112 observations "Female" (56%) and 88 "Male".

### 3.2. Exploratory Data Analysis   
Plot the variables one by one, then in pairs, or by three and see what story the graphs may tell.

```python
# sns.histplot?
```

```python
# Basic univariate plot
sns.histplot(x=df["Age"], kde=True);
```

```python
# same as above, different syntax
sns.displot(x=df["Annual_Income_(k$)"], kde=True);
```

We can observe 5 or 7 clusters from this plot.

```python
# Each visualization library has slightly different "syntax" and "names".
# For the plot below we use `plotly express` library and the `scatter()` function.
# Notice how we set the title of the plot using the `title` argument of the `scatter()` function.
# This is a bit different from how we set the title in matplotlib or seaborn.
px.scatter(
    data_frame=df,
    x="Annual_Income_(k$)",
    y="Spending_Score",
    height=400, width=400,
    title="Income versus spending score, plotly plot"
    )
```

```python
# plot categorical variable with numerical variable.
sns.catplot(
    data=df,
    x="Genre",
    y="Spending_Score",
    # height=400, width=400,
    );
```

```python
sns.boxplot(
    data=df,
    x="Genre",
    y="Spending_Score"
    ).set(title="Women have slightly higher spending score in this dataset.");
```

```python
sns.boxplot(
    data=df,
    x="Genre",
    y="Annual_Income_(k$)"
    ).set(
    title="Women have higher spending score despite slightly lower income.");
```

```python
# not a big difference it seems, but plots are not good enough for small differences.
px.scatter(
    data_frame=df,
    x="Annual_Income_(k$)",
    y="Spending_Score",
    facet_col="Genre",
    color="Genre",
    height=400, width=800
    )
```

#### Simple descriptive stats on the mean and median of the two genders shows preliminary evidence of a very small difference.   

Plots are not good enough for small differences.

```python
# show mean income and spending score by gender
df.groupby('Genre')[['Annual_Income_(k$)', 'Spending_Score']].mean()
```

```python
# show median of income and spending score by gender
df.groupby('Genre')[['Annual_Income_(k$)', 'Spending_Score']].median()
```

```python
# plot Age and Score and add some color to your gender life.
px.scatter(
    data_frame=df,
    x="Age",
    y="Spending_Score",
    color="Genre",
    height=500, width=700
    )
```

No one above 40 years old has spending score > 60.  
Very few customers below 30 years old have low spending score < 40.  

**The highest spending scores appear to be also a matter of age and not only a matter of income.**

This is the first tutorial that shows this conclusion (after reading dozens of tutorials on kaggle, here is some original data insight).

```python
px.scatter(df,
        x="Age",
        y="Annual_Income_(k$)",
        color="Genre",
        height=400, width=600,
        title="Age versus Income, by Gender",
        ).update_layout(margin=dict(t=35, l=15, r=15, b=15))
```

The plots above confirm that there are no customers aged 20 to 27 with income above 90.    
Still, there are many in this age reange with the highest spending scores.

```python
# a pairplot is a quick way to visualize relationships between all pairs of variables in a dataset.
sns.pairplot(
    data=df,
    hue='Genre',
    palette=['Blue', 'Red']
    );
```

The plot above shows a bit higher spending score for women.  
The higher distribution peaks for female are there because there are more "female" observations.    
The distribution of spending score for women is a bit shifted to the right (bottom right plot).    
But, that difference is again very small and might be true only in this small dataset (sample bias).

```python
# Boxplots are a good way to compare distributions of a numerical variable across different categories of a categorical variable. 
# They show the median, quartiles, and potential outliers in the data. In this case, we can use boxplots to compare the spending scores
# We use make_subplots which is another function from plotly library that allows us to create multiple subplots in one figure.
# We specify that we want 1 row and 2 columns of subplots, and we set horizontal_spacing to False to remove the space between the two subplots.

fig = make_subplots(rows=1, cols=2, horizontal_spacing=False, shared_yaxes=True)

fig.append_trace(go.Box(y=df[df['Genre']=='Male']['Spending_Score'],
                        name='Male', boxpoints='all'), row=1, col=1)

fig.append_trace(go.Box(y=df[df['Genre']=='Female']['Spending_Score'],
                        name='Female', boxpoints='all'), row=1, col=2)

fig.update_layout(title='Spending score for Men vs Women:',
                 plot_bgcolor='#fff',
                 yaxis=dict(showticklabels=False),
                 height=400, width=600,
                 ).update_layout(margin=dict(t=60, l=5, r=0, b=15))

fig.add_annotation(x=0.5, y=1.1, text='<b>Women have slightly higher spending score</b>',
                   xref='paper', yref='paper', showarrow=False)
fig.show()
```

### Plot 4 dimensions (4 features) on the 3D space.   
Try different views, angles, perspectives, zooms to see how the data points look like.   

There are at least five clusters and the middle one appears more densely populated.

```python
three_dim_fig = px.scatter_3d(
    df,
    x="Annual_Income_(k$)",
    y="Spending_Score",
    z="Age",
    color="Genre",
    height=600, width=900,
    color_discrete_map={"Male": "blue", "Female":"red"},
).update_layout(margin=dict(t=40, l=40, r=40, b=40))#.update_scenes(xaxis_autorange="reversed")

three_dim_fig.update_traces(marker_size = 4)

three_dim_fig.show()

# save to file, use everywhere, since it can be opened in a browser
# three_dim_fig.write_html("three_dim_customers.html")
```

#### This plot shows that ALL the customers with low annual income and high spending score belong to a relatively young age, < 35 years old

### ⏸ Pause point — questions on the data before we run KMeans

This is a good moment to check that everyone has the dataset loaded, the EDA reproduced, and at least one plot rendering. The next section introduces the KMeans constructor — make sure no one is still stuck on `pd.read_csv` or the pairplot before stepping into the algorithm.

## 4. Apply KMeans: simple example with two features and five clusters

We start with two variables (`Annual_Income_(k$)` and `Spending_Score`) and ask for five clusters.

### A quick note on `.fit()`, `.predict()` and `.fit_predict()`

In `scikit-learn`, almost every model — supervised or unsupervised — follows the **same workflow**:   

First **create** a model, then **train** it on data, and finally you **use it** to make predictions on data. The three methods below are the building blocks of that workflow:

- **`model.fit(X)`** &nbsp;→&nbsp; **train (learn from the data).**
  This is where the actual algorithm runs. For `KMeans`, calling `model.fit(X)` runs Lloyd's algorithm on the matrix `X`: it places the centroids, assigns each point to the nearest centroid, moves the centroids to the mean of their assigned points, and repeats until the centroids stop moving. After `fit`, the trained information is **stored on the model object itself** in attributes whose names end with an underscore, e.g. `model.cluster_centers_` (the final centroid coordinates) and `model.labels_` (the cluster each training point ended up in). `fit` does not return the predictions — it returns the fitted model. Think of it as: *"learn the structure of `X` and remember it."*

- **`model.predict(X_new)`** &nbsp;→&nbsp; **use the trained model on (possibly new) data.**
  Once the model has been fitted, `predict` takes a matrix of observations and returns a 1D array of cluster labels (integers `0, 1, 2, ...`) — one label per row. The model itself is **not changed** by calling `predict`; the centroids learned during `fit` are reused to assign each new point to its nearest centroid. You can call `predict` on the same `X` you trained on (to recover the training labels) or on completely new customers you have never seen before. Think of it as: *"given what you already learned, where does each of these points belong?"*

- **`model.fit_predict(X)`** &nbsp;→&nbsp; **shortcut: do both in one call.**
  This is exactly equivalent to calling `model.fit(X)` and then `model.predict(X)` on the **same** data — it trains the model on `X` and immediately returns the cluster labels for those same rows. It is the convenient one-liner you will see most often in clustering examples, because in unsupervised learning you usually want the labels of the very data you trained on. Use `fit` + `predict` separately when you plan to apply the trained model to **other** data later (e.g. new customers).

**Rule of thumb for beginners:**

| You want to...                                        | Use                |
|-------------------------------------------------------|--------------------|
| Train the model and keep it for later                 | `model.fit(X)`     |
| Get cluster labels for new points using a trained model | `model.predict(X_new)` |
| Train and get the training labels in one step         | `model.fit_predict(X)` |

A subtle but important point: `KMeans` initialises the centroids **randomly**, so two runs on the same data can give slightly different cluster numberings. To get reproducible results we always pass `random_state=...` when we create the model.

```python
# Create a matrix of two variables.
# We will train our clustering algorithm on these two features for demonstration purposes.
X = df[['Annual_Income_(k$)', 'Spending_Score']]
```

```python
# Show first lines of the data
X.head(3)
```

```python
# Assign to a variable the Kmeans algo to initialize it, and customise it with 5 clusters
# set random_state to always generate Xthe same results
model = KMeans(n_clusters=5, random_state=42)
```

```python
# show what the above variable name haXs created
model
```

```python
# KMeans?
```

```python
# fit the 5 clusters customized algorithm to the data.
fitted_kmeans = model.fit(X)
```

```python
# fitted_kmeans?
```

```python
# create y_pred = assign each observation of X to a cluster
y_pred = fitted_kmeans.predict(X)
```

```python
# all the commands above could be in one line:
y_pred = KMeans(n_clusters=5).fit_predict(X)
```

```python
# show the clusters in which each observation has been assigned to
y_pred
```

#### Assign ("predict") a cluster for one or more new observations.

```python
# predict the cluster of a new observation that has annual income 20 and spending score 50.
new_observation = [[20, 50]]
```

```python
#prediction
new_observation_pred = fitted_kmeans.predict(new_observation)
# this belongs in the cluster with label 3
new_observation_pred
```

```python
# Convert "new observation" to dataframe so that it has labels to remove the warning.
new_observation_as_a_df = pd.DataFrame(
    new_observation, columns=['Annual_Income_(k$)', 'Spending_Score']
    )

fitted_kmeans.predict(new_observation_as_a_df)
```

#### Show observations in clusters in a dataframe table.

```python
# create a new column, add the column to the dataframe to show the cluster of each observation.
df["cluster"] = pd.DataFrame(y_pred, columns=["cluster"])

# set cluster column values as category data type
df["cluster"] = df["cluster"].astype('category')

#show first rows of observations
df.head(5)
```

```python
# count number of observations in each cluster, sorted by number of observations
df['cluster'].value_counts()
```

```python
# same as above, different syntax, output sorted by cluster number
# The FutureWarning means this will not work soon. Important to know this, that is why I leave the code as is.
df.groupby('cluster').size()
```

```python
# Show average of variables my cluster
# This shows some basic descriptive stats about the clusters.
# e.g. cluster 3 has the highest spending score (82) with average income=82k and age=32
# e.g. cluster 0 has almost highest spending score (79) with average income only 26k and age=25
# This IMPORTANT finding confirms the previous insight about the importance of age.
df.iloc[:, 1:-1].mean()
```

```python
df.iloc[:, 1:].groupby("cluster", observed=True).mean()
```

```python
# fitted_kmeans?
```

```python
fitted_kmeans.cluster_centers_
```

```python
# Simple way to plot the clusters with their centroids using seaborn and matplotlib.
sns.scatterplot(
    data=df,
    x="Annual_Income_(k$)",
    y="Spending_Score",
    hue="cluster",
    palette="deep"
    )

sns.scatterplot(
    x=fitted_kmeans.cluster_centers_[:, 0],  # x coordinate of the centroids
    y=fitted_kmeans.cluster_centers_[:, 1],  # y coordinate of the centroids
    s=200,
    c=sns.color_palette("deep", 5),
    label='Centroids',
    marker='*')

# Put the legend outside of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0);
```

```python
# plot the five clusters in 3D space in each cluster, sorted by number of observations
df['cluster'].value_counts()
three_dim_clusters_fig = px.scatter_3d(
    df,
    x="Annual_Income_(k$)",
    y="Spending_Score",
    z="Age",
    color="cluster",
    height=600, width=600,
    color_discrete_map={0: "blue", 1:"purple", 2:"green", 3:"yellow", 4:"red"},
).update_layout(margin=dict(t=40, l=40, r=40, b=40))

three_dim_clusters_fig.update_traces(marker_size = 3)

three_dim_clusters_fig.show()

# save to file, use everywhere with a browser
#three_dim_clusters_fig.write_html("three_dim_customers.html")
```

## Stepping up — pick K from the data, not by guessing

Until now `K = 5` was a guess. Two metrics — the **elbow method** (`inertia_`) and the **silhouette score** — let us pick `K` from the *data* instead of choosing by intuition.

The elbow plot tells you where adding another cluster stops buying you much; the silhouette score tells you how cleanly separated the clusters actually are. When they disagree, the maintainer's call usually goes to whichever `K` produces clusters the *business* can act on.

## 5. Model evaluation metrics for clustering (choosing `k`)

In this section we use two metrics — the **elbow method (WCSS)** and the **silhouette score** — to get a suggestion for a good number of clusters `k`. This should be done first, but for presentation and teaching purposes we showed how KMeans is applied in scikit-learn first.

We will now use all four features (`Genre`, `Age`, `Annual_Income_(k$)`, `Spending_Score`) instead of just two. Because `Genre` is categorical, we first encode it as a numeric "dummy" column.

```python
df.columns
```

```python
# Create the matrix X of the features to use in the algorithm
X = df[["Genre", "Age", "Annual_Income_(k$)", "Spending_Score"]]

# show first rows
X.head(2)
```

### 5.1. Encoding the categorical feature (`Genre` → `Gender`)

**What values does `KMeans` actually need?**

`KMeans` uses **Euclidean distance** to assign points to centroids, so every feature must be **numeric**. It does not care about *types* — only that arithmetic works.

- `pd.get_dummies(..., drop_first=True)` returns **`True`/`False`** values (a boolean Series).
- Booleans behave as `1` (True) and `0` (False) in numeric operations, so KMeans runs without complaint and gives identical results to the integer-encoded version.
- It is still cleaner to **cast explicitly to `int`** so the column reads as `0/1`. This avoids surprises if the column is later combined with other numeric features, scaled, or exported.

Below we encode `Genre` so that **`Male = 1`, `Female = 0`** (`drop_first=True` drops the alphabetically first category, `Female`).

```python
# encode the categorical Genre column as 0/1 for use in KMeans
# pd.get_dummies with drop_first=True drops "Female" alphabetically and keeps "Male"
# .astype(int) converts the True/False output to clean 0/1 integers (Male=1, Female=0)
X["Gender"] = pd.get_dummies(X.Genre, drop_first=True, prefix="Genre").astype(int)
X.head(3)
```

```python
# drop the Genre column from X. We now use the Gender column.
X = X.drop("Genre", axis=1)
# show first 3 rows
X.head(3)
```

### 5.2. Elbow method (WCSS)

**What is WCSS?** WCSS stands for *Within-Cluster Sum of Squares*. For each cluster, we take every point that belongs to it, measure the squared distance from that point to the cluster's centroid (centre), and add all of those squared distances together. WCSS is the sum across **all** clusters.

A **lower WCSS** means points sit closer to their centroid, i.e. tighter clusters. WCSS always decreases as we add more clusters (with `k = n` clusters, every point would be its own centroid and WCSS would be 0), so we cannot simply minimise it. Instead, we look for the **"elbow"** in the curve: the value of `k` after which adding another cluster no longer produces a meaningful drop in WCSS.

In scikit-learn, WCSS is exposed as the `inertia_` attribute of a fitted `KMeans` model.

Apply KMeans with four features and 1 to 15 clusters to demonstrate the elbow method.   
Look at the "elbow" — the point at which you get not much more valuable info by having more clusters.

```python
# empty list to contain the score of each cluster.
clustering_score = []

# Loop 15 times.
# Create and fit 15 models.
# Get the inertia score for 15 models, from 1 cluster to 15.
for clusters_number in range(1, 15):
    # initiate and customize algo
    model = KMeans(n_clusters=clusters_number, random_state=42)
    # fit algo to the data
    model.fit(X)
    # inertia_ = Sum of squared distances of samples to their closest cluster center.
    # this is the Within Clusters Sum of Squares metric, WCSS
    clustering_score.append(model.inertia_)
```

```python
# show the list with the clusters' score from 1 to 10
clustering_score
```

```python
# WCSS score for 1 cluster
clustering_score[0]  # first item in the list
```

```python
# WCSS score for 2 clusters
clustering_score[1]  # second item in the list
```

```python
# creata pandas dataframe from list of clustering WCSS score
wcss_df = pd.DataFrame({'clusters_score':clustering_score})
```

```python
# creata new column of dataframe from range of clusters
wcss_df["clusters_number"] = pd.Series(range(1,11))
```

```python
#show clusters score for number of clusters
wcss_df
```

We see that from 4 to five clusters there is a large reduction in WCSS, but from 5 to 6 there is a reduction from 58348 to 51132, much smaller.   
The reduction in WCSS from 6 to 7 clusters is similar.  

There is a much smaller reduction from 7 to 8 clusters.

```python
plt.figure(figsize=(7,5))

#plot the simple range from 1 to 10 on x and the clustering score on y.
plt.plot(range(1, 15), clustering_score);
```

```python
# plot WCSS
# create a figure and set size
plt.figure(figsize=(7,5))

#plot the simple range from 1 to 10 on x and the clustering score on y.
plt.plot(range(1, 15), clustering_score)

#plot a red o on number 5 on x axis, and the WCSS score 
plt.scatter(5, clustering_score[4], s = 30, c = 'red', marker='o')

# add title and labels
plt.title('The Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('Clustering Score, WCSS');
```

We also see that the slope of the curve from 5 to 6 clusters is reduced but very slightly. This suggests five clusters are a good choice.
But using four features instead of two suggest that 6 cluster might also be a good choice.

### 5.3. Silhouette score

**What is the silhouette score?**   
It measures how well each point fits inside its own cluster compared to the next-nearest cluster. For every point we compute two numbers:

- `a` = the mean distance from the point to all **other points in its own cluster** (how *cohesive* the cluster is).
- `b` = the mean distance from the point to all points in the **nearest neighbouring cluster** (how *separated* the clusters are).

The point's silhouette is then `(b - a) / max(a, b)`, a value between **-1 and +1**. Close to **+1** means the point is well inside its cluster; around **0** means it sits on the border between two clusters; **negative** suggests it may have been assigned to the wrong cluster. The overall silhouette score reported below is the **average** of these per-point values.

Unlike WCSS, **higher silhouette is better**, so we can pick the `k` that maximises it directly.

scikit-learn silhouette score [function and examples.](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

For a more advanced reading on silhouette score you may read this [tutorial.](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py)   

For more advanced clustering performance evaluation methods, you may read this [guide.](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

```python
# Create empty list to store score
silhouette_scores_list = []

# loop from 2 to 15 clusters:
for clusters_number in range(2, 15):  # Silhouette starts from 2 clusters.
    # initiate and customize algo
    model = KMeans(n_clusters=clusters_number, random_state=42)
    # fit algo to the data
    clusters = model.fit_predict(X)
    # calculate silhouette score for each number of clusters
    silhouette_metric_score = silhouette_score(X, clusters)
    # append each score to the list
    silhouette_scores_list.append(silhouette_metric_score)
    print(
        f"For clusters={clusters_number}, the average silhouette_score is: {silhouette_metric_score}")
```

```python
# The percentage difference for silhouette score for 6 versus five clusters
((0.45-0.376)/0.376) * 100
```

We get a higher silhouette score for 6 clusters.   
Around 20% higher versus 5 clusters.

### Elbow vs silhouette — how to use them together

The two metrics often disagree slightly: the **elbow** (within-cluster sum of squares, `inertia_`) rewards *compact* clusters, so it keeps going down as `K` grows and just bends at the elbow. The **silhouette score** rewards *well-separated* clusters, so it has a real maximum at the `K` that produces the cleanest split.

In practice you read both together: the elbow tells you when you're *paying for nothing* (each new cluster shaves only a tiny bit off WCSS), and the silhouette tells you whether the clusters are *actually well-separated*. When they disagree by ±1, prefer the `K` the business can act on — six personas are easier to brief a marketing team on than seven.

You'll see this exact tension on the four-feature run in §6 below, where the silhouette numbers favour `K = 6` (which we use) but the elbow alone could have argued for `K = 5`.

### 5.4. Further reading on clustering evaluation metrics

- [Scikit-learn model evaluation and metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#metrics-and-scoring-quantifying-the-quality-of-predictions)   
- [Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)   

If there are no labels, use:   

- [Silhouette Coefficient](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)  
- [Calinski-Harabasz index](https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index)

## Stepping up again — from 2 features to 4

We now bring `Age` and `Gender` (numerically encoded, `Male = 1`, `Female = 0`) into the feature matrix and re-run KMeans with `K = 6` — the value the silhouette score preferred. Adding features changes the geometry, so the cluster shapes will differ from the 2-feature version above.

### ⏭ Skip if running long

If the class is tight, **§6 (4-feature KMeans) and §7 (profile + label) can be moved to at-home review**. They reproduce the same workflow as §4 with two extra columns, and the profiling step in §7 is mechanical (per-cluster `describe()` + label assignment) — it benefits from quiet thought rather than a rushed in-class demo. The in-class flow then jumps from here to §8/§9 (further reading + career frame) for the closing discussion.

If the class is on time, continue below.

## 6. Apply KMeans with four features and six clusters

```python
# our data until now, grouped in 5 clusters
df.head(3)
```

```python
#show the features matrix
X.head(3)
# remember we have encoded "Genre" as "Gender".
```

```python
# Create KMeans model with 6 clusters.
model = KMeans(n_clusters=6, random_state=42)
# fit algo to the data
six_clusters_fit = model.fit(X)
six_clusters_kmeans = model.fit_predict(X)
```

```python
# Show the clusters in which each observation has been assigned to
six_clusters_kmeans
```

```python
# create a new column, add it to the dataframe to show cluster sof observations.
df["six_clusters"] = pd.DataFrame(
    six_clusters_kmeans, columns=["six_clusters"])

# set six clusters column values as category data type
df["six_clusters"] = df["six_clusters"].astype('category')
```

```python
# show first 4 rows
df.head(4)
```

```python
# plot the six clusters in the 3D space
three_dim_six_clusters = px.scatter_3d(
    df, 
    x="Annual_Income_(k$)", 
    y="Spending_Score",
    z="Age",
    color="six_clusters",
    height=700, width=1000,
    color_discrete_map={0: "blue", 1:"purple", 2:"green", 3:"yellow", 4:"red", 5:"brown"},
    size='Annual_Income_(k$)',
).update_layout(margin=dict(t=50, l=60, r=60, b=40))

# three_dim_six_clusters.update_traces(marker_size=3)  # uncomment to make the points smaller
three_dim_six_clusters.show()

# save to file, use everywhere with a browser
#three_dim_clusters.write_html("three_dim_customers.html")
```

Examining the data, we see that the customers with spending score 40 to 60 have been devided to two clusters, based on their age.    
The new cluster mainly contains customers above 40+ with high spending score. It is designated as cluster 1 in this case, and this the larger cluster.  

**Important reminder: After all, the number of clusters is also a matter of policy decisions.**

## The last step — give the clusters human-readable names

Cluster numbers like `0, 1, 2, 3, 4, 5` are useless to a marketing team — they need *names*. In this section we inspect each cluster's per-feature averages and assign a one- or two-word human-readable label.

This is the step that turns a KMeans output into something a non-technical stakeholder can read in a slide deck.

## 7. Define, describe and label clusters after examining them one by one.

### A quick revisit — `predict` in an unsupervised model

In §4 you ran `fitted_kmeans.predict(...)` on three new customer points and got back integer cluster IDs. In §6 you ran `fit_predict(X)` on the full four-feature matrix and got back IDs again. Both times *predict* returned a **label per row** — but there was no `y` target anywhere in the workflow.

The reason: **scikit-learn uses one consistent API across supervised and unsupervised models**. For an unsupervised clusterer, `predict(X_new)` does not predict a target value — it assigns each new point to the nearest *already-trained* centroid. The name is API-level, not semantic.

This section pushes the idea one more step: those integer IDs are not enough on their own. Marketing teams want *names*. The work below converts `{0, 1, 2, 3, 4, 5}` into `{"young_yolos", "save_or_spend_elsewhere", ...}`. That conversion — IDs to names — is the move that turns a KMeans output into a deliverable.

```python
# plot one of the six clusters, cluster "zero" in the 3D space to examine it
plot_cluster_zero = px.scatter_3d(
    df[df["six_clusters"] == 0],  # Just change the dataframe to filter out all other clusters
    x="Annual_Income_(k$)",
    y="Spending_Score",
    z="Age",
    color="six_clusters", height=800, width=800,
    color_discrete_map={
        0: "blue", 1:"purple", 2:"green", 3:"black", 4:"red", 5:"brown"},
    # size='Annual_Income_(k$)'  # Not needed because I don't compare it with aothe clusters.
).update_layout(margin=dict(t=50, l=60, r=60, b=40))

plot_cluster_zero.update_traces(marker_size=4)

plot_cluster_zero.show()
```

```python
# show basic descriptive stats for cluster labeled as 0.
df[df["six_clusters"] == 0].describe()
```

**How would you describe, define this "cluster 0" in a few words? How would you "label" it in one or two words to remember it?**     
How is this cluster different from others?    
I define it as: "45ers and above, with average income and average spending relative to their income".  
And label them as: "wise_constrained", implyting that they are wise because they spend according to their budget constraint, and are also wise "age-wise".

```python
# plot out of the six clusters, cluster "one" in the 3D space to examine it
plot_cluster_one = px.scatter_3d(
    df[df["six_clusters"] == 1],  # Just change the dataframe to filter out all other clusters
    x="Annual_Income_(k$)",
    y="Spending_Score",
    z="Age",
    color="six_clusters",
    height=800, width=800,
    color_discrete_map={
        0: "blue", 1:"purple", 2:"green", 3:"black", 4:"red", 5:"brown"},
    # size='Annual_Income_(k$)'  # Not needed because I don't compare it with another cluster.
).update_layout(margin=dict(t=50, l=60, r=60, b=40))

plot_cluster_one.update_traces(marker_size=4)

plot_cluster_one.show()
```

```python
# statistically describe the observations that belong to cluster 1.
df[df["six_clusters"] == 1].describe()
```

**How would you describe, define this "cluster one" in a few words? How would you "label" it in one or two words to remember it?**     
How is this cluster different from others?    
Let's see... hmmm ... Age 27 to 40, high to very high income, very high spending score.   
I assume that perhaps they are postgraduates from BIS UoA that after ther masters started finally earning a nice salary and now they want to enjoy it.  

I define this cluster as: "in their thirties, with a lot of cash and finally can spend a lot".  
And label them as: "start_earning_living_it", implyting that they enjoy spending now that they started earning it after working and studying hard for years.

```python
# plot "cluster two" out of the six clusters in the 3D space to examine it
plot_cluster_two = px.scatter_3d(
    df[df["six_clusters"] == 2],  # Just change the dataframe to filter out all other clusters
    x="Annual_Income_(k$)",
    y="Spending_Score",
    z="Age",
    color="six_clusters",
    height=800, width=800,
    color_discrete_map={
        0: "blue", 1:"purple", 2:"green", 3:"black", 4:"red", 5:"brown"},
    # size='Annual_Income_(k$)'  # Not needed because I don't compare it with another cluster.
).update_layout(margin=dict(t=50, l=60, r=60, b=40))

plot_cluster_two.show()
```

```python
# statistically describe the observations that belong to cluster 2.
df[df["six_clusters"] == 2].describe()
```

**How would you describe, define this "cluster two" in a few words? How would you "label" it in one or two words to remember it?**     
How is this cluster different from others?    
Let's see... hmmm ... all ages, very low income, very low spending score.   
 
I would label them as: "have_not_spend_not", implying that they don't spend a lot because they don't have a lot.

```python
# statistically describe the observations that belong to cluster 3.
df[df["six_clusters"] == 3].describe()
```

**How would you describe, define this "cluster three" in a few words? How would you "label" it in one or two words to remember it?**     
How is this cluster different from others?    
Let's see... hmmm ... all ages, high to very income, low to very low spending score.   
 
I would label them as: "savers_or_spend_elsewhere", implying that they don't spend a lot because they save it, or they don't spend it at the Mall.

```python
# statistically describe the observations that belong to cluster 4.
df[df["six_clusters"] == 4].describe()
```

**How would you describe, define this "cluster four" in a few words? How would you "label" it in one or two words to remember it?**     
How is this cluster different from others?     
I would label them as: "young_cautious", implying that they don't spend a lot because of they are budget constrained, and tehy are young.  
This cluster differs to cluster zero mostly because of their age.

```python
# statistically describe the observations that belong to cluster 5.
# Last of six clusters, index in python starts from zero.
df[df["six_clusters"] == 5].describe()
```

**How would you describe, define this "cluster five" in a few words? How would you "label" it in one or two words to remember it?**     
How is this cluster different from others?     
Yound age, max age is 35, low income, high to super high spending score
I would label them as: "young_yolos", implying that they don't spend it all now that they are young, despite not having a lot.

```python
# Replace the cluster number with my labels to help me remember what the cluster groups might suggest.
my_funny_intuitite_cluster_labes = {0: "wise_constrained",
                                    1: "start_earning_living_it",
                                    2: "have_not_spend_not",
                                    3: "save_or_spend_elsewhere",
                                    4: "young_cautious",
                                    5: "young_yolos",
}
```

```python
# Create new column assigning my labels according to cluster number.
df["six_clusters_labels"] = df["six_clusters"].map(my_funny_intuitite_cluster_labes)
```

```python
# show first rows of data
df.head(4)
```

**Notice the new cluster in the middle. It is only a matter of age.**

```python
# 2D plot the six clusters, use as legend the new labels.

sns.scatterplot(
    x="Annual_Income_(k$)",
    y="Spending_Score",
    hue="six_clusters_labels",
    data=df, palette="deep")

# also add cluster numbers next to text in the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, title="Clusters");
```

```python
# 3D plot the six clusters with descriptive legend labels.
cluster_legend_labels = {
    "wise_constrained": "wise_constrained: 40+, average income, moderate spending",
    "start_earning_living_it": "start_earning_living_it: age 27-40, high income, very high spending",
    "have_not_spend_not": "have_not_spend_not: low income, low spending",
    "save_or_spend_elsewhere": "save_or_spend_elsewhere: high income, low spending",
    "young_cautious": "young_cautious: under 35, low income, low spending",
    "young_yolos": "young_yolos: under 35, low income, very high spending",
}

# Plot using long labels in legend so each cluster description is visible.
three_dim_six_clusters_labels = px.scatter_3d(
    df.assign(
        six_clusters_labels_long=df["six_clusters_labels"].map(cluster_legend_labels)
    ),
    x="Annual_Income_(k$)",
    y="Spending_Score",
    z="Age",
    color="six_clusters_labels_long",
    height=800,
    width=1400,
    color_discrete_map={
        cluster_legend_labels["wise_constrained"]: "blue",
        cluster_legend_labels["start_earning_living_it"]: "purple",
        cluster_legend_labels["have_not_spend_not"]: "green",
        cluster_legend_labels["save_or_spend_elsewhere"]: "yellow",
        cluster_legend_labels["young_cautious"]: "red",
        cluster_legend_labels["young_yolos"]: "brown",
    },
    size="Annual_Income_(k$)",
).update_layout(
    margin=dict(t=50, l=60, r=420, b=40),
    legend=dict(
        title="Cluster label and profile",
        x=1.02,
        y=1,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="lightgray",
        borderwidth=1,
    ),
)

three_dim_six_clusters_labels.update_traces(marker_size=4)

three_dim_six_clusters_labels.show()
```

```python
# 3D plot three out of the six clusters to compare them.
three_dim_six_clusters_labels = px.scatter_3d(
    df[(df["six_clusters_labels"] == "wise_constrained") |
       (df["six_clusters_labels"] == "start_earning_living_it") |
       (df["six_clusters_labels"] == "have_not_spend_not")],
    x="Annual_Income_(k$)",
    y="Spending_Score",
    z="Age",
    color="six_clusters_labels", 
    height=800, width=800,
    color_discrete_map={
        "wise_constrained": "blue",
        "start_earning_living_it":"purple",
        "have_not_spend_not":"green"},
    size='Annual_Income_(k$)',
).update_layout(margin=dict(t=50, l=60, r=60, b=40))

three_dim_six_clusters_labels.show()
```

```python
# Show clusters by size
df.groupby('six_clusters', observed=False).size()
```

```python
# group cluster labeled as 1, by gender and show size
df[df["six_clusters"] == 1].groupby("Genre").size()
```

### ⏸ Pause point — last questions on the workflow

Before we close with the further-reading list (§8) and the career-frame discussion (§9), last questions on:

- the estimator-API pattern (`fit` / `predict` / `fit_predict`, `cluster_centers_`, `labels_`, `inertia_`)?
- picking `K` from the elbow + silhouette?
- the per-cluster profile → human-readable label step?

## 8. Extra reading about clustering and unsupervised learning.

[Unsupervised learning algorithms in scikit learn.](https://scikit-learn.org/stable/unsupervised_learning.html)    
Clustering is a small subset of unsupervised learning.     
[Clustering algorithms in scikit learn.](https://scikit-learn.org/stable/modules/clustering.html#)

## 9. Where does this end up on the job?

KMeans plays at least three different roles in industry — knowing which one a project needs is half the job.

**1. As the deliverable.** Customer segmentation in retail / banking / telecoms is often the project itself. Modern production systems continuously ingest activity events and **periodically recompute KMeans clusters** over RFM features (recency / frequency / monetary value); the resulting personas feed campaign-management tools and loyalty-tier rules directly. The `lec_09a` → `lec_09c` flow you just worked through is a small version of this pattern, and it is a common deliverable in CRM-analytics roles.

**2. As feature engineering.** When clustering isn't the headline, the cluster ID gets passed downstream as one input feature for a classifier or regressor — *cluster-id-as-feature*. Cheap way to coarsen a complex feature space without writing rule-based segments by hand.

**3. As infrastructure.** Modern vector databases (**FAISS**, Pinecone, Milvus) use k-means under the hood to build their *Inverted File* (IVF) indexes — clustering embeddings into buckets so that approximate nearest-neighbour search only has to scan a few buckets per query instead of the whole corpus. Meta has reported running FAISS on 1.5 trillion vectors. Most retrieval-augmented-generation (RAG) systems that do anything other than a brute-force scan are sitting on top of a k-means clustering somewhere underneath.

For when the data shape doesn't fit KMeans's spherical-cluster assumption, see the alternatives in `lec_09f` (DBSCAN, hierarchical / agglomerative, Gaussian mixtures). The day-one skill the job market pays for — in all three roles above — is the same: **pick K, profile the clusters, deploy or hand off the result downstream.**

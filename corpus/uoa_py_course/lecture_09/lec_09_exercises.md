<!-- source: lectures_07_13_pandas_plots_scikit/lecture_09_clustering_deploy_hf_app/practice_exercises/lec_09_exercises.ipynb @ 0cc874704aaa -->

# Lecture 09 — Practice Exercises

All exercises from Lecture 09 collected in one place. Work through them after studying the corresponding lecture material.

Each exercise is tagged with **the learning goal it covers** (see `reading_material/goals_09.md`) and **its difficulty tier**:

- `[G1]`–`[G8]`: required goals (every required exercise must be completed)
- `[O1]`–`[O3]`: optional / career-track goals (the stretch exercises only)
- *trivial* ≈ 15 min, *realistic* ≈ 45 min, *stretch* ≈ 90 min

| Section | Source notebook | Topic |
|---|---|---|
| 0 | `lec_09a` | Concept warm-up: unsupervised vs supervised |
| 1 | `lec_09a` | KMeans on the Mall Customers dataset |
| 2 | `lec_09a` | Choosing `K` (elbow + silhouette) and profiling clusters |
| 3 | `lec_09b` | KMeans assumptions and where it breaks |
| 4 | `lec_09c` | Wrap a trained KMeans in a Gradio app |
| 5 | `lec_09d` *(optional)* | Tuning `init` and `n_init` (stretch) |

Solutions for every required exercise (and the stretch one) are in `lec_09_exercises_solutions.ipynb`.

---
## Setup — Load libraries and data

Run this cell first to have everything ready for the exercises below. The `mall_customers.csv` file lives in the `reading_material/` folder next door, so we load it from there with a relative path.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons, make_blobs
```

```python
DATA_PATH = '../reading_material/mall_customers.csv'
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["CustomerID"])
print(f"Mall customers dataset: {df.shape[0]} rows, {df.shape[1]} columns")
df.head(3)
```

---
### Exercise 0.1 [G1, *trivial*] — Unsupervised vs supervised

In a markdown cell or comment block below, answer in **one sentence**:

> *In one sentence, what is the difference between supervised and unsupervised learning, and name one task each one is typically used for.*

This is a concept-only exercise — no code required. Aim for one clear sentence; if you need three sentences, your distinction isn't tight enough yet.

```python
# Exercise 0.1 — Your one-sentence answer here, as a comment or in a markdown cell.

# Supervised learning ...
# Unsupervised learning ...
```

---
## 1. KMeans on the Mall Customers dataset

*From `lec_09a_kmeans_clustering.ipynb`*

In the lecture we walked through the full KMeans workflow on this dataset: load → EDA → fit → predict → evaluate → label clusters. Now you do it yourself.

### Exercise 1.1 [G2 + G3, *realistic*] — KMeans with two features and `K = 5`

Fit a `KMeans` model on the two features `Annual_Income_(k$)` and `Spending_Score`, asking for **5 clusters**.

**Steps:**
1. Build the feature matrix `X` with these two columns.
2. Create `model = KMeans(n_clusters=5, random_state=42, n_init=10)`.
3. Use `fit_predict(X)` to get cluster labels in **one step**.
4. Add the labels as a new column on `df` named `cluster`.
5. Inspect `model.cluster_centers_`, `model.labels_`, and `model.inertia_`. Print all three.
6. Make a 2D scatter plot coloured by `cluster`, with the centroids overlaid.

```python
# Exercise 1.1 — Your code here
```

### Exercise 1.2 [G6, *trivial*] — Predict the cluster of new customers

Imagine three new customers walk into the mall:

| Customer | Annual income (k$) | Spending score |
|---|---|---|
| A | 20  | 80 |
| B | 75  | 50 |
| C | 110 | 15 |

Using the model you trained in Exercise 1.1, predict which cluster each one belongs to.

**Steps:**
1. Build a small `pandas.DataFrame` with the three new customers (use the same column names as the training data).
2. Call `model.predict()` on the new DataFrame.
3. Print the cluster id for each customer.

**Question to answer in a markdown cell below:** Why is this called *predict* even though there is no `y` target?

```python
# Exercise 1.2 — Your code here
```

---
## 2. Choosing `K` and profiling clusters

*From `lec_09a_kmeans_clustering.ipynb`*

`K` is a hyperparameter you must pick yourself. We use two metrics in the lecture: **WCSS** (within-cluster sum of squares, the "elbow") and the **silhouette score**.

### Exercise 2.1 [G4, *realistic*] — Elbow + silhouette on the same data

Compute both metrics across a range of `K` values on the **two-feature** matrix (`Annual_Income_(k$)`, `Spending_Score`) and plot them side-by-side.

**Steps:**
1. Loop `k` from 1 to 10. For each `k`, fit a fresh `KMeans` and append `model.inertia_` to a list.
2. Loop `k` from 2 to 10 (silhouette is undefined for K=1). For each `k`, fit, predict, and append `silhouette_score(X, labels)` to a second list.
3. Plot the two lists side-by-side (`fig, axes = plt.subplots(1, 2, ...)`): elbow on the left, silhouette on the right. Mark your chosen `K` on both with a vertical dashed line.

**Question to answer in a markdown cell below:** Do the elbow method and the silhouette score agree on the best `K`? When they disagree, which one should you trust and why?

```python
# Exercise 2.1 — Your code here
```

### Exercise 2.2 [G5, *realistic*] — Apply your chosen `K` with **four** features and profile the clusters

Now use **all four** features: `Genre`, `Age`, `Annual_Income_(k$)`, `Spending_Score`.

**Steps:**
1. Encode `Genre` as a numeric `Gender` column (`Male` → 1, `Female` → 0) so KMeans can use it.
2. Build `X` with the four numeric columns.
3. Fit `KMeans` with the `K` you chose in 2.1 (use `random_state=42`, `n_init=10`).
4. Print the per-cluster mean of every numeric feature (`df.groupby('cluster').mean()`).
5. Give each cluster a one- or two-word **human-readable label** (e.g. "young high-spenders") in a small dictionary `cluster_labels = {0: "...", 1: "...", ...}`.
6. Add a `cluster_label` column to `df` and re-plot the 2D scatter using the labels in the legend.

```python
# Exercise 2.2 — Your code here
```

---
## 3. KMeans assumptions and where it breaks

*From `lec_09b_kmeans_assumptions_caveats.ipynb`*

KMeans implicitly assumes that clusters are **roughly spherical**, **of similar size**, and **of similar variance**. When those assumptions are violated, the algorithm produces clusters that look obviously wrong to the human eye.

### Exercise 3.1 [G7, *realistic*] — Two failure modes: non-convex shapes and anisotropic blobs

Reproduce **both** classical KMeans failure modes on synthetic data.

**Part A — Non-convex shapes (`make_moons`):**

1. `X_moons, y_moons = make_moons(n_samples=400, noise=0.06, random_state=42)`.
2. Plot `X_moons` coloured by the true label `y_moons` (you should see two interlocking moons).
3. Fit `KMeans(n_clusters=2, random_state=42, n_init=10)` on `X_moons`, predict labels, and plot side-by-side with the true labels.

**Part B — Anisotropic (stretched) blobs:**

1. `X_blobs, y_blobs = make_blobs(n_samples=600, centers=3, cluster_std=0.6, random_state=42)`.
2. Apply an anisotropic linear transformation: `X_aniso = X_blobs @ np.array([[0.6, -0.6], [-0.4, 0.8]])`.
3. Fit `KMeans(n_clusters=3, random_state=42, n_init=10)` on `X_aniso` and plot predicted vs true labels side-by-side, with centroids overlaid on the predicted plot.

**Question to answer in a markdown cell below (one sentence per failure mode):** What property of KMeans causes it to fail on each of the two cases above?

```python
# Exercise 3.1 — Your code here
```

---
## 4. Wrap a trained KMeans in a Gradio app

*From `lec_09c_gradio_app_huggingface_deploy.ipynb`*

### Exercise 4.1 [G8, *realistic*] — Wrap a trained KMeans in a Gradio app (local)

Re-use the four-feature model from Exercise 2.2 and wrap it in a small Gradio interface that runs locally (no Hugging Face deploy required for this exercise — that's the lecture-09c walk-through).

**Steps:**
1. Re-train (or pickle and reload) the four-feature `KMeans` model.
2. Write a function `predict_segment(genre, age, income, spending) -> str` that:
   - encodes `genre` as `0/1` exactly as in Exercise 2.2,
   - builds a single-row DataFrame with the four features in the same column order,
   - calls `model.predict(...)` and returns the cluster id (and, optionally, the human-readable label from your `cluster_labels` dict).
3. Build a `gr.Interface` (or `gr.Blocks`) with four input components — `gr.Radio` for genre, `gr.Slider` for age / income / spending — wired to your `predict_segment` function.
4. Call `.launch()` (the local URL `http://127.0.0.1:7860/` should open).

**Reminder:** the deploy-to-Hugging-Face step is in `lec_09c`. This exercise stops at "runs locally" — that is enough to demonstrate G8 mechanically.

```python
# Exercise 4.1 — Your code here
```

---
## 5. Optional / career-track stretch exercise

*From `lec_09d_kmeans_other_parameters.ipynb` — optional bucket*

### Exercise 5.1 [O1, *stretch — optional*] — `init='random'` vs `init='k-means++'`

KMeans is sensitive to where the centroids start. By default scikit-learn uses the smart initialisation `'k-means++'`, but you can ask for `'random'`.

**Steps:**
1. Use the **two-feature** mall-customers matrix (`Annual_Income_(k$)`, `Spending_Score`).
2. Fit two models with `K = 5`:
   - `KMeans(n_clusters=5, init="random", n_init=1, random_state=0)`
   - `KMeans(n_clusters=5, init="k-means++", n_init=1, random_state=0)`
3. Print `inertia_` for both. Which one is lower?
4. Repeat the experiment with `n_init=10`. Does the gap shrink?

**What this teaches:** why `n_init` exists and why `'k-means++'` is the default. Read `lec_09d_kmeans_other_parameters.ipynb` for the full discussion of the other constructor parameters.

```python
# Exercise 5.1 — Your code here
```

### Exercise 5.2 [O2, *stretch — optional*] — Lloyd's algorithm in two phases

This is a **written-answer** exercise — no coding required.

**Steps:**
1. Open `reading_material/lec_09e_kmeans_step_by_step_animations.ipynb` and watch the centroid-convergence animation (`kmeans_animation.gif`).
2. In a markdown cell below, describe **Lloyd's algorithm in two phases** in your own words, in **three sentences**:
   - one sentence describing the **assignment** phase (what happens to each data point);
   - one sentence describing the **update** phase (what happens to each centroid);
   - one sentence describing the **stopping condition** (when the loop exits).

**What this teaches:** intuition for why KMeans converges, and why running it a few times with different `random_state` values is sometimes necessary (the loop can land in different local minima depending on where the centroids started).

```python
# Exercise 5.2 — Written answer here (in a markdown cell below this one, or as a comment).

# Phase 1 (assignment):  ...
# Phase 2 (update):      ...
# Stopping condition:    ...
```

### Exercise 5.3 [O3, *stretch — optional*] — When to prefer DBSCAN, hierarchical, GMM over KMeans

This is a **written-answer** exercise — no coding required.

**Steps:**
1. Open `reading_material/lec_09f_other_clustering_algos.ipynb` and read the three short sections (DBSCAN, hierarchical / agglomerative, Gaussian Mixture Models).
2. In a markdown cell below, write **one sentence per algorithm** answering the question: *when would I prefer this algorithm over KMeans, on what kind of data?*

Aim for one specific cluster shape or dataset property per algorithm — not generic phrases like "when the data is complex". The model answer for DBSCAN, for example, is one sentence about *density-based* clustering and *non-convex* shapes.

**What this teaches:** a working data scientist needs to recognise — within seconds of looking at a scatter plot — whether KMeans is the right tool. The fastest way to build that reflex is to memorise the *one* data shape each major alternative handles better.

```python
# Exercise 5.3 — Written answer here (in a markdown cell below this one, or as a comment).

# DBSCAN — prefer when:                  ...
# Hierarchical / agglomerative — prefer when:  ...
# Gaussian Mixture Models — prefer when:       ...
```

---
## Summary

After completing the **7 required** exercises (0.1, 1.1–1.2, 2.1–2.2, 3.1, 4.1) you will have practised every required learning goal G1–G8. The **three stretch exercises** in section 5 (5.1–5.3) cover the three optional goals O1–O3 — they are bonus work for students who want the career-track depth.

| Section | Required exercises | Skills practised | Goal(s) |
|---|---|---|---|
| 0 | 0.1 | One-sentence distinction between supervised and unsupervised | G1 |
| 1 | 1.1, 1.2 | Fitting `KMeans`, reading `cluster_centers_` / `labels_` / `inertia_`, `predict()` on new points | G2, G3, G6 |
| 2 | 2.1, 2.2 | Picking `K` with elbow + silhouette side-by-side; profiling and labelling clusters at the chosen `K` | G4, G5 |
| 3 | 3.1 | Recognising both classical KMeans failure modes (non-convex, anisotropic) in one combined exercise | G7 |
| 4 | 4.1 | Wrapping a trained model in a Gradio app | G8 |
| 5 *(optional)* | 5.1 | Tuning `init` and `n_init` | O1 |
| 5 *(optional)* | 5.2 | Lloyd's algorithm in two phases (written answer after watching `lec_09e`) | O2 |
| 5 *(optional)* | 5.3 | When to prefer DBSCAN / hierarchical / GMM (written answer after reading `lec_09f`) | O3 |

**Key scikit-learn API patterns reinforced here:**

- `model.fit(X)` &nbsp;→&nbsp; train and store `cluster_centers_`, `labels_`, `inertia_`
- `model.predict(X_new)` &nbsp;→&nbsp; assign labels to **new** points
- `model.fit_predict(X)` &nbsp;→&nbsp; one-line shortcut for unsupervised learning
- `silhouette_score(X, labels)` &nbsp;→&nbsp; quality of a clustering

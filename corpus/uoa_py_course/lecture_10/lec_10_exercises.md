<!-- source: lectures_07_13_pandas_plots_scikit/lecture_10_knn_train_test_split/practice_exercises/lec_10_exercises.ipynb @ 0cc874704aaa -->

# Lecture 10 — Practice Exercises

All exercises from Lecture 10 collected in one place. Work through them after studying the corresponding lecture material in `../reading_material/`.

Each exercise is tagged with:
- The **goal** it covers (G3–G9 for required, O1–O2 for optional/stretch — see `../reading_material/goals_10.md`).
- A **difficulty tier**: *trivial* (≤15 min), *realistic* (≤45 min), *stretch* (≤90 min).
- The **source notebook** for the technique you'll use.

Stretch exercises are tied to optional goals and depend on the optional notebooks (`lec_10d`, `lec_10e`). You can skip them and still cover every required goal.

Per the course README, practice exercises are **optional** and **graded only positively** — they are bonus work, not a requirement.

## Setup — Load libraries

Run this cell once to have everything ready for the exercises below. All datasets come from scikit-learn (`load_wine`, `load_breast_cancer`, `load_iris`) so no external CSV files are needed.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_wine, load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_STATE = 42
```

---
## 0. Before the algorithm — the train/test split (conceptual exercises)

*Source: `../reading_material/lec_10a_knn_classification.ipynb` §B1.3*

The two exercises below are written-answer warm-ups on the concept of the train/test split. They do not require running any code; answer them in markdown cells (or as Python comments) below each prompt. They map to the new conceptual goals G1 and G2 in `goals_10.md`.

### Exercise 0.1 [G1, *trivial*] — Why a held-out test set?

In **two to four sentences**, explain to a classmate who has only done unsupervised learning (clustering) why supervised classification needs a train/test split. Your answer should mention:

- What goes wrong if you score the model on the same data you fit it on.
- One concrete number from this lecture that gives the wrong impression when you skip the split.
- Why KNN's "lazy learning" property does not exempt it from this requirement.

```python
# Exercise 0.1 — your answer here (as a markdown cell below this one, or as a multi-line comment).

# Why does supervised classification need a train/test split?
#  ...
```

### Exercise 0.2 [G2, *trivial*] — Two-way vs three-way split

In one short paragraph each, describe:

- A scenario where a **two-way** train/test split (e.g., 70/30) is the right choice. Name one concrete advantage of stopping at two splits.
- A scenario where a **three-way** train / validation / test split is the right choice. Name one concrete problem the validation set fixes that the two-way split cannot.

Aim for ~3-4 sentences per scenario. No code required.

```python
# Exercise 0.2 — your answer here.

# Scenario A — two-way split is enough when ...
#  ...

# Scenario B — three-way split is needed when ...
#  ...
```

---
## 1. The basic KNN pipeline on a new dataset

*Source: `../reading_material/lec_10a_knn_classification.ipynb`*

In the lecture we walked through the full KNN pipeline on iris. These two exercises ask you to repeat the workflow on a **different** dataset (the wine dataset — 13 features, 3 classes, 178 observations).

### Exercise 1.1 [G3, *trivial*] — Four-line KNN pipeline

Load the wine dataset using `load_wine(as_frame=True)`. Build the simplest possible KNN classifier on the **first two features** of the wine data:

1. Split into train / test (use `random_state=RANDOM_STATE`, `test_size=0.3`, `stratify=y`).
2. Fit a `KNeighborsClassifier()` with default parameters on the training set.
3. Predict the class of the test set.
4. Print the test-set accuracy as a percentage.

You should be able to do this in roughly four short cells.

```python
# Exercise 1.1 — your code here.
# Load the wine dataset and grab the first two features.

# wine = load_wine(as_frame=True)
# X = wine.data.iloc[:, :2]
# y = wine.target
```

```python
# Split into train and test sets.
```

```python
# Fit a default KNeighborsClassifier on the training set, predict on the test set.
```

```python
# Print the test-set accuracy as a percentage.
```

### Exercise 1.2 [G3 + G4, *realistic*] — Full pipeline + confusion matrix on breast cancer

Now use the **whole** breast-cancer dataset (`load_breast_cancer(as_frame=True)` — 30 features, 2 classes, 569 observations).

1. Split into train / test (same settings as 1.1).
2. **Scale the features** with `StandardScaler` (`.fit_transform` on train, `.transform` on test).
3. Fit a `KNeighborsClassifier(n_neighbors=7)` on the scaled training set.
4. Predict on the scaled test set.
5. Print:
   - The test-set accuracy.
   - The confusion matrix.
   - The classification report (`classification_report(y_test, y_pred)`).

The breast-cancer features have wildly different scales — this is exactly the situation where `lec_10b` warned you that scaling matters. Predict the accuracy you would get *without* scaling, then change one line and check.

```python
# Exercise 1.2 — your code here.
# Load breast cancer, split, scale, fit, evaluate.
```

---
## 2. Tuning K with the train-vs-test curve

*Source: `../reading_material/lec_10a_knn_classification.ipynb` §B4*

### Exercise 2.1 [G5, *realistic*] — K-sweep on the wine dataset

Using the wine dataset with **all 13 features** (scaled — see Exercise 1.2 for why), plot **training accuracy and test accuracy on the same axes** as K varies from 1 to 30.

Pick the K that gives the best **test** accuracy. Annotate your plot with that K (e.g., a vertical line or a text marker).

Hint: this is exactly the pattern from `lec_10a` cells 88–93, applied to a new dataset.

```python
# Exercise 2.1 — your K-sweep code here.
```

---
## 3. Failure modes (the lec_10b material)

*Source: `../reading_material/lec_10b_knn_assumptions_caveats.ipynb`*

### Exercise 3.1 [G6, *realistic*] — Scaled vs unscaled KNN

Build a synthetic 2-feature dataset with `make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=RANDOM_STATE)`. Then **multiply the first feature by 100** so it dominates the distance metric.

Fit a `KNeighborsClassifier(n_neighbors=5)` and report the test accuracy in two scenarios:

1. Without scaling.
2. With `StandardScaler` applied (fit on train, transform on test).

Print both accuracies side by side. The unscaled version should be visibly worse — confirm that and write a one-sentence explanation (in a markdown cell) of *why*.

```python
# Exercise 3.1 — your scaled-vs-unscaled comparison here.
```

### Exercise 3.2 [G7, *realistic*] — Name the failure mode

For each scenario below, **name which KNN failure mode** (high dimensionality / class imbalance / noisy or irrelevant features / feature scale sensitivity) you would suspect, and **what one plot** you would draw to confirm.

Answer in markdown cells below each scenario — no code required.

**Scenario A.** You fit KNN on a customer-churn dataset with 200 features and accuracy is 51% (random would be 50%). Your colleague's logistic regression on the same data hits 78%.

**Scenario B.** You fit KNN on a fraud-detection dataset with 99.5% non-fraud and 0.5% fraud. The model predicts "non-fraud" for every test observation and accuracy looks like 99.5%. Your boss is suspicious.

**Scenario C.** You fit KNN on a housing dataset with three features: rooms (range 1–8), square metres (range 30–500), and listing price in cents (range 50000–1500000). KNN's predictions are nearly random.

**Scenario D.** You fit KNN on a 5-feature dataset where features 1–2 are informative and features 3–5 are pure noise added by your data team to "increase coverage". Accuracy drops vs. a 2-feature baseline.

**Your answers for 3.2:**

- *Scenario A*: failure mode = ... ; diagnostic plot = ...
- *Scenario B*: failure mode = ... ; diagnostic plot = ...
- *Scenario C*: failure mode = ... ; diagnostic plot = ...
- *Scenario D*: failure mode = ... ; diagnostic plot = ...

---
## 4. Decision boundaries

*Source: `../reading_material/lec_10c_knn_decision_boundaries.ipynb`*

### Exercise 4.1 [G8, *realistic*] — Boundary with distance weighting on a chosen feature pair

Re-plot the iris decision boundary, but this time:

- Use the **petal** features (`X = iris.data[:, 2:4]`) instead of the sepal features.
- Use `weights="distance"` instead of `weights="uniform"`.
- Try K = 5 and K = 15 on the same figure (two side-by-side subplots).

Look at the resulting plot and answer in a markdown cell: **how does the boundary differ from the sepal-features version in `lec_10c`?**

```python
# Exercise 4.1 — your boundary plot here.
# Tip: the plot_knn_boundary helper from lec_10c.ipynb is what you want;
# you can copy it into a cell here, or re-implement the mesh-grid pattern from scratch.
```

---
## 5. Predicting on new observations

*Source: `../reading_material/lec_10a_knn_classification.ipynb` §B2.4*

### Exercise 5.1 [G9, *trivial*] — Predict three new flowers

Using a `KNeighborsClassifier` fit on the **full** iris dataset (no train/test split — we want maximum data behind the prediction since we're predicting on flowers nobody has measured yet):

1. Construct three new observations as a `pandas.DataFrame` with the four iris feature columns. Use values you make up that span the realistic ranges (sepal length 4–8 cm, petal length 1–7 cm, etc.).
2. Call `.predict()` on the DataFrame to get the predicted species for each.
3. Also call `.predict_proba()` to see how confident the model is.
4. In one sentence: how is predicting on these new observations different from predicting on `X_test`?

```python
# Exercise 5.1 — your three-observation prediction here.
```

---
## Stretch exercises

*These map to the **optional / career-track** goals (O1, O2, O3). They use material from `lec_10d` (optional) and `lec_10e` (optional). Skip these and you still cover every required goal.*

### Exercise S.1 [O1, *stretch*] — Tune `metric` and `p`

*Source: `../reading_material/lec_10d_knn_other_parameters.ipynb`*

Using the wine dataset (scaled, all 13 features, K=7, `weights="uniform"`), measure the test-set accuracy with each of these distance settings:

- `metric="minkowski", p=1` (Manhattan)
- `metric="minkowski", p=2` (Euclidean — the default)
- `metric="minkowski", p=3` (in between)
- `metric="chebyshev"` (max of absolute differences across features)

Plot the four accuracies as a bar chart, and answer in markdown: **which won on this dataset, and by how much?**

```python
# Exercise S.1 — your metric-tuning comparison here.
```

### Exercise S.2 [O2, *stretch*] — Find a dataset where logistic regression beats KNN

*Source: `../reading_material/lec_10e_knn_vs_other_classifiers.ipynb`*

Construct a 2-class, 2-feature synthetic dataset where the true decision boundary is roughly **linear**: use `make_classification(n_samples=400, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, class_sep=1.5, random_state=RANDOM_STATE)`.

Fit both a `KNeighborsClassifier(n_neighbors=5)` and a `LogisticRegression()` on the same train/test split (scale features for both). Report the test accuracies and explain in two sentences **why the linear model has the edge here**.

(For full credit: produce the dataset's decision-boundary plot for both models on the same figure — two subplots — and discuss the *shape* of the boundary, not just the number.)

```python
# Exercise S.2 — your logistic-vs-KNN comparison here.
```

### Exercise S.3 [O3, *stretch*] — KNN regression with honest evaluation

*Source: `../reading_material/lec_10f_knn_regression_teaser.ipynb`*

In `lec_10f` we plotted KNN regression curves on a noisy sine wave and reported **training** MSE — which under-estimates the true generalisation error (small K looked great on training; the eye told you it was overfitting).

Build the honest version of that experiment:

1. Re-generate the noisy-sine dataset from `lec_10f` (`X_reg` and `y_reg` with `random_state=42`).
2. **Split** into train / test with `test_size=0.25` and a fixed `random_state`.
3. Sweep K from 1 to 30. For each K, fit a `KNeighborsRegressor` and record **both** training MSE and **test** MSE.
4. Run the sweep **twice** — once with `weights='uniform'`, once with `weights='distance'`. Four arrays of MSE total.
5. Plot all four curves on one figure (one line per `(set, weights)` combo). Add a vertical line at the K that minimises **test** MSE.
6. In a markdown cell, answer: (a) which `(K, weights)` minimised test MSE; (b) what does the training-MSE curve do near K=1 and why is it misleading; (c) name one failure mode from `lec_10b` that would matter if the dataset were 100 features instead of 1.

```python
# Exercise S.3 — your code here.
# Hint: rng = np.random.RandomState(42); X_reg = np.sort(5 * rng.rand(80, 1), axis=0); y_reg = ...
```

---

You're done. If you finished every required exercise, you have:

- Explained the train/test split concept and the train/validation/test variant (Exercises 0.1, 0.2).
- Built KNN pipelines on three datasets (iris, wine, breast cancer, plus synthetic ones).
- Read confusion matrices and classification reports.
- Tuned K with a train-vs-test curve.
- Demonstrated for yourself why scaling matters.
- Named four failure modes and the diagnostic plots that surface them.
- Plotted decision boundaries with two weighting rules.
- Predicted the class of made-up observations.
- If you also tackled the stretch tier: tuned `metric` / `p` (S.1), pitted logistic regression against KNN on a linear-boundary dataset (S.2), and built an honest train-vs-test sweep for KNN regression (S.3).

Check your work against `lec_10_exercises_solutions.ipynb`.

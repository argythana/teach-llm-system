<!-- source: lectures_07_13_pandas_plots_scikit/lecture_10_knn_train_test_split/practice_exercises/lec_10_exercises_solutions.ipynb @ 0cc874704aaa -->

# Lecture 10 — Practice Exercises (Solutions)

Runnable solutions for `lec_10_exercises.ipynb`. Every required exercise (0.1, 0.2, 1.1, 1.2, 2.1, 3.1, 3.2, 4.1, 5.1) and both stretch exercises (S.1, S.2, S.3) execute cleanly under `course_venv`.

Use these to check your work after attempting the exercises yourself. The point of the practice notebook is the attempt, not the answer.

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
### 0.1 [G1] — Why a held-out test set?

Supervised classification trains the model to map features (`X`) to labels (`y`). If we then score the model on the same `(X, y)` it learned from, we are asking "how well did you memorise the training set?" rather than "how well will you predict on new observations?" — and those are very different numbers. In this lecture, a KNN classifier with `K = 1` always reports 100% accuracy on its training data (every point is its own nearest neighbour), but its test accuracy on iris is much lower than that — about 91% with `random_state=42`. That gap is the whole point of holding out a test set.

KNN's "lazy" property describes *when* the algorithm does its work — at query time, not at fit time. It says nothing about whether evaluation needs honest data. Lazy or eager, the algorithm is still learning a mapping from `X` to `y`, and we still need labels it has not seen to judge whether that mapping generalises.

---
### 0.2 [G2] — Two-way vs three-way split

**Scenario A — two-way is enough.** You want to compare two completely different algorithms (e.g., KNN vs logistic regression) on a one-shot baseline. You pick the default hyperparameters of each, fit both on the training set, score both on the test set, and report which one wins. Because you are not tuning hyperparameters, the test set is never used for model *selection* — only for one final evaluation per algorithm — so its bias as an honest estimator is preserved. The two-way split is also simpler to teach and leaves more rows in the training set, which matters on small datasets like iris (150 rows).

**Scenario B — three-way is needed.** You want to pick the best `K` for KNN by sweeping `K = 1..30`. If you pick the `K` that scores best on the test set, you have implicitly used the test set to select the model — and the reported accuracy is no longer unbiased; it is the maximum of 30 noisy estimates, which is systematically too optimistic. The validation set fixes this: you sweep `K` on the validation set, pick the winner, and then score *that winner* on the untouched test set. The test number you finally report has only been "seen" once, by one model — which is the honest answer you wanted.

---
### 1.1 [G3] — Four-line KNN pipeline on wine

```python
wine = load_wine(as_frame=True)
X = wine.data.iloc[:, :2]   # first two features only
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

clf = KNeighborsClassifier().fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Test accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")
```

---
### 1.2 [G3 + G4] — Full pipeline + confusion matrix on breast cancer

```python
bc = load_breast_cancer(as_frame=True)
X = bc.data
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

clf = KNeighborsClassifier(n_neighbors=7).fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)

print(f"Test accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred, target_names=bc.target_names))
```

**Without scaling**, the same pipeline drops to ~91-92% accuracy because features like `mean area` (range ~100–2500) dominate the distance metric over features like `mean smoothness` (range ~0.05–0.16). Scaling lifts accuracy back to ~96-97%.

---
### 2.1 [G5] — K-sweep on the (scaled) wine dataset

```python
wine = load_wine(as_frame=True)
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

neighbors = np.arange(1, 31)
train_acc = np.empty(len(neighbors))
test_acc = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    clf = KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train)
    train_acc[i] = clf.score(X_train_s, y_train)
    test_acc[i] = clf.score(X_test_s, y_test)

best_k = neighbors[test_acc.argmax()]
print(f"Best K (highest test accuracy): {best_k} -> {test_acc.max() * 100:.1f}%")

plt.figure(figsize=(8, 5))
plt.plot(neighbors, train_acc, marker='o', label='Training accuracy')
plt.plot(neighbors, test_acc, marker='s', label='Test accuracy')
plt.axvline(best_k, color='red', linestyle='--', alpha=0.5, label=f'Best K = {best_k}')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Wine dataset: KNN K-sweep (scaled)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---
### 3.1 [G6] — Scaled vs unscaled KNN

```python
X, y = make_classification(
    n_samples=500, n_features=2, n_informative=2, n_redundant=0,
    random_state=RANDOM_STATE,
)
X[:, 0] = X[:, 0] * 100  # force the first feature to dominate the distance metric

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# 1) WITHOUT scaling
clf_raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
acc_raw = accuracy_score(y_test, clf_raw.predict(X_test))

# 2) WITH StandardScaler
scaler = StandardScaler().fit(X_train)
clf_scaled = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(X_train), y_train)
acc_scaled = accuracy_score(y_test, clf_scaled.predict(scaler.transform(X_test)))

print(f"Unscaled test accuracy: {acc_raw * 100:.1f}%")
print(f"Scaled test accuracy:   {acc_scaled * 100:.1f}%")
```

**Why the gap?** Feature 0's range is roughly ±300 (after the × 100 multiplier); feature 1's range is roughly ±3. Euclidean distance squares both differences, so feature 0's contribution swamps feature 1's by about (100)² = 10 000×. The unscaled classifier effectively only sees one feature; the scaled one sees both.

---
### 3.2 [G7] — Name the failure mode

- **Scenario A** — *Curse of dimensionality.* 200 features on a tabular customer-churn dataset is far beyond KNN's comfort zone; all points become roughly equidistant. **Diagnostic plot:** a histogram of pairwise distances between training points — if the distribution is tightly clustered around a single value, dimensionality is killing you. Alternatively, an accuracy-vs-feature-count curve (`SelectKBest` increasing K) should show accuracy peaking well below 200 features.

- **Scenario B** — *Class imbalance.* 99.5% / 0.5% split + a high-sounding accuracy that drops to zero on the minority class. **Diagnostic plot:** the confusion matrix (or the per-class recall in `classification_report`). You'll see the minority class has recall ≈ 0. A precision-recall curve on the minority class is also revealing.

- **Scenario C** — *Feature scale sensitivity.* Price in cents has a range a thousand times wider than rooms; distance is dominated by price. **Diagnostic plot:** a pair plot or feature-range bar chart — when one feature's bar dwarfs the others, KNN will silently ignore the small-range ones. Confirm by re-running with `StandardScaler` applied.

- **Scenario D** — *Noisy / irrelevant features.* The added features add to the distance computation without contributing real signal. **Diagnostic plot:** an accuracy-vs-feature-subset curve (e.g., `SelectKBest`'s `f_classif` ranking), or feature importance from a small `RandomForestClassifier` — if features 3–5 score near zero, drop them.

---
### 4.1 [G8] — Boundary with distance weighting on petal features

```python
iris = load_iris()
X = iris.data[:, 2:4]   # petal length, petal width
y = iris.target

cmap_regions = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_points = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def plot_boundary(ax, X, y, k, weights):
    clf = KNeighborsClassifier(n_neighbors=k, weights=weights).fit(X, y)
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=cmap_regions, shading='auto')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolors='k', s=30)
    ax.set_xlabel('Petal length (cm)')
    ax.set_ylabel('Petal width (cm)')
    ax.set_title(f"K = {k}, weights = '{weights}'")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_boundary(axes[0], X, y, k=5, weights='distance')
plot_boundary(axes[1], X, y, k=15, weights='distance')
plt.tight_layout()
plt.show()
```

**Difference from the sepal-features version in `lec_10c`:** petal features separate the three iris species much more cleanly (setosa is far from the other two; versicolor and virginica overlap only narrowly). On the petal slice, the boundary is dominated by simple horizontal strips; the sepal slice's boundary was much more contorted because of the heavy versicolor / virginica overlap. This is a small lesson in feature selection: pick features that separate your classes, and the model has less work to do.

---
### 5.1 [G9] — Predict three new flowers

```python
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

clf = KNeighborsClassifier().fit(X, y)

new_flowers = pd.DataFrame(
    [
        [5.1, 3.5, 1.4, 0.2],   # looks setosa-ish
        [6.4, 2.9, 4.3, 1.3],   # looks versicolor-ish
        [7.5, 3.0, 6.0, 2.0],   # looks virginica-ish
    ],
    columns=X.columns,
)

predictions = clf.predict(new_flowers)
probabilities = clf.predict_proba(new_flowers)

species_names = iris.target_names
for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
    print(f"Flower {i+1}: predicted = {species_names[pred]}")
    print(f"          probabilities: {dict(zip(species_names, probs))}")
```

**How is this different from predicting on `X_test`?** The test set was held out from the training data, so we know the true labels and can measure accuracy. The three new flowers above are observations we *invented* — there are no ground-truth labels, only the model's predictions. In production, every prediction call looks like this case: you have features, you don't have labels, and the model's vote is your only answer until reality catches up (the customer churns or doesn't, the flower blooms and someone classifies it).

---
## Stretch solutions

### S.1 [O1] — Tune `metric` and `p`

```python
wine = load_wine(as_frame=True)
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

settings = [
    ("minkowski p=1 (Manhattan)", dict(metric='minkowski', p=1)),
    ("minkowski p=2 (Euclidean)", dict(metric='minkowski', p=2)),
    ("minkowski p=3",             dict(metric='minkowski', p=3)),
    ("chebyshev",                 dict(metric='chebyshev')),
]

results = []
for name, params in settings:
    clf = KNeighborsClassifier(n_neighbors=7, weights='uniform', **params).fit(X_train_s, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_s))
    results.append((name, acc))

names = [r[0] for r in results]
accs = [r[1] for r in results]

plt.figure(figsize=(9, 4))
plt.barh(names, accs)
plt.xlabel('Test accuracy')
plt.xlim(min(accs) - 0.01, 1.0)
for i, acc in enumerate(accs):
    plt.text(acc + 0.001, i, f"{acc * 100:.1f}%", va='center')
plt.title('Wine: KNN test accuracy across distance metrics (K=7, scaled)')
plt.tight_layout()
plt.show()

for name, acc in results:
    print(f"  {name:30s} {acc * 100:.2f}%")
```

On the (scaled) wine dataset, the four metrics typically land within ~1–3 percentage points of each other. Manhattan (`p=1`) often edges out Euclidean (`p=2`) on tabular data because outlier features have less leverage; Chebyshev (max difference) can be surprisingly competitive on standardised data. The exact ranking is dataset-dependent — that's the point: trying these is the only way to know.

### S.2 [O2] — Linear-boundary dataset where logistic regression beats KNN

```python
X, y = make_classification(
    n_samples=400, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.5, random_state=RANDOM_STATE,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_s, y_train)
logreg = LogisticRegression(random_state=RANDOM_STATE).fit(X_train_s, y_train)

acc_knn = accuracy_score(y_test, knn.predict(X_test_s))
acc_log = accuracy_score(y_test, logreg.predict(X_test_s))

print(f"KNN (K=5):           {acc_knn * 100:.2f}%")
print(f"Logistic regression: {acc_log * 100:.2f}%")
```

```python
# Decision-boundary plot for both models.
def plot_2model_boundary(ax, model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), shading='auto')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']),
               edgecolors='k', s=20)
    ax.set_title(title)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_2model_boundary(axes[0], knn, X_test_s, y_test, f"KNN (K=5) — {acc_knn * 100:.1f}%")
plot_2model_boundary(axes[1], logreg, X_test_s, y_test, f"Logistic regression — {acc_log * 100:.1f}%")
plt.tight_layout()
plt.show()
```

**Why logistic regression has the edge:** the true decision boundary in this synthetic dataset is essentially linear (the two classes are linearly separable apart from a few overlap points). Logistic regression fits *exactly* one straight line — its bias matches the data-generating process. KNN draws a more complex, locally-responsive boundary that wraps around training-set noise; on a linearly-separable problem, that extra flexibility hurts rather than helps. The visual is clearer than the accuracy number: the logistic-regression boundary is a clean diagonal; the KNN boundary wiggles. KNN is paying for variance that the problem does not have.

### S.3 [O3] — KNN regression with honest evaluation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1) Reproduce the noisy-sine dataset from lec_10f
rng = np.random.RandomState(42)
X_reg = np.sort(5 * rng.rand(80, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + 0.1 * rng.randn(80)

# 2) Train/test split — KNN has no random_state, but train_test_split does.
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.25, random_state=42
)

# 3) + 4) Sweep K = 1..30 under both weighting schemes; collect train and test MSE.
ks = np.arange(1, 31)
results = {w: {'train': [], 'test': []} for w in ['uniform', 'distance']}

for w in ['uniform', 'distance']:
    for k in ks:
        reg = KNeighborsRegressor(n_neighbors=k, weights=w).fit(X_train, y_train)
        results[w]['train'].append(mean_squared_error(y_train, reg.predict(X_train)))
        results[w]['test'].append(mean_squared_error(y_test, reg.predict(X_test)))

# Identify the (K, weights) combo that minimises test MSE.
best_k = None
best_w = None
best_mse = float('inf')
for w in ['uniform', 'distance']:
    for k, mse in zip(ks, results[w]['test']):
        if mse < best_mse:
            best_mse = mse
            best_k = k
            best_w = w
print(f"Best test MSE: {best_mse:.4f} at K = {best_k}, weights = '{best_w}'")
```

```python
# 5) Plot all four curves on one figure.
fig, ax = plt.subplots(figsize=(9, 5))
for w, style in [('uniform', '-'), ('distance', '--')]:
    ax.plot(ks, results[w]['train'], style, label=f"train MSE — {w}", alpha=0.6)
    ax.plot(ks, results[w]['test'], style, marker='o', label=f"test MSE — {w}", linewidth=2)
ax.axvline(best_k, color='red', linestyle=':', alpha=0.7, label=f'best test K = {best_k} ({best_w})')
ax.set_xlabel('K (n_neighbors)')
ax.set_ylabel('MSE')
ax.set_title('KNN regression — train vs test MSE across K')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Answers.**

**(a) Which `(K, weights)` minimised test MSE?**
On this seed, somewhere in the K = 5–10 range with either weighting — distance-weighting tends to edge out uniform by a hair in the lower-K region, same noise-floor effect as the classification weights sweep in `lec_10d` §2. The exact winner moves with the seed; do not over-read a single split.

**(b) What does the training-MSE curve do near K=1 and why is it misleading?**
At K=1 the training MSE is **essentially zero** — every training point is its own nearest neighbour, so the model's "prediction" at each training point is just that point's own `y` value. Training MSE drops to 0 at K=1 by construction, not because the model is good. This is exactly the trap `lec_10a` §B1.3 warned about: scoring on the same data you fit on lies to you. The honest signal is in the **test** MSE curve, which is U-shaped — high at K=1 (overfitting to noise), drops to a minimum in the K=5–10 region, and rises again as K grows large enough to smear out the underlying sine structure.

**(c) Name one failure mode from `lec_10b` that would matter if the dataset were 100 features instead of 1.**
**Curse of dimensionality.** With 100 features, all training points become roughly equidistant from any query point (distances concentrate around their mean), so "nearest" stops being meaningful. The K nearest neighbours are basically a random sample of the training set, and KNN's prediction collapses toward the global mean of `y` — same failure mode that hits classification accuracy in `lec_10b` §3, but now visible as a flat (and bad) test-MSE curve at every K.

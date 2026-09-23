<!-- source: lectures_07_13_pandas_plots_scikit/lecture_10_knn_train_test_split/reading_material/lec_10d_knn_other_parameters.ipynb @ 0cc874704aaa -->

# Lecture 10d. KNN: the other useful parameters (besides `n_neighbors`)

> *Optional / Career-track* — read **after** the mandatory `lec_10a` (end-to-end pipeline), `lec_10b` (failure modes), and `lec_10c` (decision boundaries).

In `lec_10a` we focused on the single hyperparameter that matters the most for KNN: `n_neighbors` (the K). But the `KNeighborsClassifier` constructor has several other parameters that quietly shape the **accuracy** of the model, the **speed** of fit/predict, and whether the model behaves the same way in production as it did on your laptop.

Goal of this notebook: open up the `KNeighborsClassifier` constructor and **explain the remaining parameters with a small example for each**.

Full constructor (sklearn 1.x):

```python
KNeighborsClassifier(
    n_neighbors=5,
    *,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=None,
)
```

We will work through these in the order in which they actually matter in practice:

1. `weights` — should closer neighbours count more?
2. `metric` — what does "distance" even mean?
3. `p` — the order of the Minkowski distance.
4. `algorithm` — the data structure used to find the neighbours.
5. `leaf_size` — a tuning knob for the tree-based algorithms.

We skip `metric_params` (only used by a few exotic metrics) and `n_jobs` (parallel-prediction knob).

## 1. Imports and data setup

We use two datasets:

- **iris** (150 rows, 4 features) for accuracy comparisons — small enough to read every number.
- A **synthetic 10 000-sample, 20-feature** dataset from `make_classification` for the timing comparisons in sections 4 and 5 — only there to make the speed difference between `algorithm` choices visible.

```python
import time

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

```python
# Iris — small, clean, 3 well-separated classes.
iris = sns.load_dataset('iris')
FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris[FEATURES].values
y = iris['species'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y,
)

# Always scale features before KNN (see lec_10b).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('X_train:', X_train_scaled.shape, '  X_test:', X_test_scaled.shape)
```

```python
# A larger synthetic dataset for the timing comparisons later.
X_big, y_big = make_classification(
    n_samples=10_000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    random_state=42,
)

X_big_train, X_big_test, y_big_train, y_big_test = train_test_split(
    X_big, y_big, test_size=0.20, random_state=42, stratify=y_big,
)

scaler_big = StandardScaler()
X_big_train = scaler_big.fit_transform(X_big_train)
X_big_test = scaler_big.transform(X_big_test)

print('X_big_train:', X_big_train.shape, '  X_big_test:', X_big_test.shape)
```

## 2. `weights` — should closer neighbours count more?

When KNN predicts the class of a new point, it looks at the K closest training points and **votes**. `weights` controls how that vote is counted:

- `'uniform'` *(default)*: every one of the K neighbours gets one equal vote.
- `'distance'`: each neighbour's vote is weighted by `1 / distance`, so points that are **closer to the query** count more than points sitting at the edge of the neighbourhood.
- A **callable** (function) that maps an array of distances to an array of weights. Rare — only needed for custom weighting schemes.

Let's compare `'uniform'` vs `'distance'` on iris.

```python
# Sweep K from 1 to 15 and compare uniform vs distance weighting on iris.
# Print a side-by-side table so we can see the per-K behaviour, not just one number.

print(f"  K  | uniform | distance |  delta")
print(f" ----|---------|----------|--------")
for k in range(1, 16):
    accs = {}
    for w in ['uniform', 'distance']:
        clf = KNeighborsClassifier(n_neighbors=k, weights=w)
        clf.fit(X_train_scaled, y_train)
        accs[w] = accuracy_score(y_test, clf.predict(X_test_scaled))
    delta = accs['distance'] - accs['uniform']
    print(f"  {k:2d} |  {accs['uniform']:.3f}  |  {accs['distance']:.3f}   | {delta:+.3f}")
```

On iris, `distance` ties-or-edges-out `uniform` on most K values, but the maximum accuracy is the same under both (0.956 here). Try a different train/test seed and the picture can flip — on a clean dataset, the choice is at the noise floor. The real reasons to prefer one over the other are the failure-mode bullets below. If we set `random_state=123,` in `train_test_split`, then 'uniform' wights outperform 'distance'.

**When does `weights='distance'` actually help?**

- When the local density is **uneven**: some regions of feature space have many training points, others have few. Distance-weighting prevents far-away neighbours from drowning out a tight cluster of close neighbours.
- On **close calls** (a test point sitting right between two classes): the uniform vote is a coin flip; the distance-weighted vote leans toward whichever class is *physically nearer*.
- In **imbalanced datasets**: distance-weighting partly compensates because a minority class point that is actually the closest neighbour gets a big weight, instead of being out-voted by majority-class points slightly further away.

Drawback: distance-weighting amplifies the impact of any single very close neighbour, which makes the model **more sensitive to noisy labels** in the training set.

## 3. `metric` — what does "distance" even mean?

KNN needs a distance function to decide who is "close". `metric` picks that function:

- `'minkowski'` *(default)* — the generalised distance controlled by `p` (see section 4). With `p=2` it equals Euclidean, with `p=1` it equals Manhattan.
- `'euclidean'` — straight-line distance, $\sqrt{\sum_i (x_i - y_i)^2}$. The geometric intuition most people have.
- `'manhattan'` — sum of absolute differences, $\sum_i |x_i - y_i|$. Think city blocks. Less sensitive to a single very-different coordinate than Euclidean.
- `'chebyshev'` — the largest single-coordinate difference, $\max_i |x_i - y_i|$. Two points are "close" only if they are close on **every** feature.

There are more (`'cosine'`, `'hamming'`, …); these four are the common ones for dense numeric features.

Let's compare them on iris.

```python
for m in ['minkowski', 'euclidean', 'manhattan', 'chebyshev']:
    clf = KNeighborsClassifier(n_neighbors=5, metric=m)
    clf.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    print(f'metric={m:>10s}   test accuracy = {acc:.3f}')
```

Notice that `'minkowski'` and `'euclidean'` give **identical** accuracy on iris — that is because the default `p=2` makes Minkowski exactly Euclidean (we verify this explicitly in section 4).

**When to reach for which metric?**

- `'euclidean'` (or `'minkowski'` with `p=2`) — the default. Good when features are on comparable scales (after `StandardScaler`) and you trust straight-line geometry.
- `'manhattan'` — preferred in **high dimensions** (it is less prone to the curse of dimensionality than Euclidean, see lec_10b) and when features are roughly independent grid-like coordinates.
- `'chebyshev'` — useful when you want "two points are similar only if they agree on every feature". Niche; common in some chess/grid problems.
- `'minkowski'` with a custom `p` — when you want a tunable in-between of Manhattan and Euclidean.

```python
# Same idea as the §2 K-sweep, now across EVERY (metric, weights) combination.
# Each row: one (metric, weights) pair; columns: K=1..15.

metrics = ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
weights_list = ['uniform', 'distance']

header = f"{'metric':>10s}  {'weights':>9s}   " + "  ".join(f"K={k:2d}" for k in range(1, 16)) + "   peak"
print(header)
print("-" * len(header))

for m in metrics:
    for w in weights_list:
        row = []
        for k in range(1, 16):
            clf = KNeighborsClassifier(n_neighbors=k, weights=w, metric=m)
            clf.fit(X_train_scaled, y_train)
            row.append(accuracy_score(y_test, clf.predict(X_test_scaled)))
        peak = max(row)
        k_at_peak = row.index(peak) + 1
        print(
            f"{m:>10s}  {w:>9s}   "
            + "  ".join(f"{a:.3f}" for a in row)
            + f"   {peak:.3f}@K={k_at_peak}"
        )
```

**Reading the full table.**

- `minkowski` and `euclidean` produce **identical** numbers across every (K, weights) combination. This is not a coincidence — with the default `p=2`, Minkowski *is* Euclidean. We verify this rigorously in section 4 below using `np.array_equal` on the predictions.
- `minkowski` / `euclidean` peak at **0.956** test accuracy on iris (43/45 flowers correct), reached at K=8 with `distance` and K=9 with `uniform` — and at a handful of larger K with both weighting schemes.
- `manhattan` peaks at **0.933** (42/45) at K=1, never reaching 0.956.
- `chebyshev` peaks at **0.933** (at K=4 with distance, K=7 with uniform) and dips as low as **0.844** at K=15 with uniform — visibly the weakest of the four metrics on iris.
- The choice of **metric** matters more on iris than the choice of **weights** within a metric: the metric peak-to-peak gap is ~1 flower (0.956 vs 0.933), while the weights gap within any one metric is at the noise floor (see §2).

**Why `chebyshev` is the worst on iris.** Chebyshev distance says "two points are close only if they agree on *every* feature." On iris, no single feature cleanly separates the three species — petal length and petal width carry most of the signal, but sepal length and width overlap heavily between versicolor and virginica. Forcing closeness on the noisiest coordinates throws away information the other metrics use.

**Same seed caveat as §2.** All of these numbers come from one `train_test_split` partition (`random_state=42`). Change the seed and the per-K details move. The metric-level ordering (Euclidean ≥ Manhattan ≥ Chebyshev on iris) is more stable than the weights ordering, but neither is set in stone for a 45-sample test set — try a couple of seeds before treating any one ranking as final.

## 4. `p` — the order of the Minkowski distance

Minkowski distance is

$$ d(x, y) = \left( \sum_i |x_i - y_i|^p \right)^{1/p} $$

and the value of `p` selects a familiar metric as a special case:

- `p=1` → Manhattan distance
- `p=2` *(default)* → Euclidean distance
- `p → ∞` → Chebyshev distance

`p` is only used when `metric='minkowski'`. Setting `p=3` (or any non-integer) gives an in-between metric that is rarely useful in practice but is sometimes worth a try as a hyperparameter.

Let's verify the equivalence on iris by showing that `metric='minkowski', p=1` produces **exactly the same predictions** as `metric='manhattan'`.

```python
clf_minkowski_p1 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)
clf_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')

preds_minkowski_p1 = clf_minkowski_p1.fit(X_train_scaled, y_train).predict(X_test_scaled)
preds_manhattan = clf_manhattan.fit(X_train_scaled, y_train).predict(X_test_scaled)

print('Identical predictions:', np.array_equal(preds_minkowski_p1, preds_manhattan))
print('Accuracy (either way):', accuracy_score(y_test, preds_manhattan))
```

```python
# Same idea for p=2 vs metric='euclidean'.
clf_minkowski_p2 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
clf_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

preds_minkowski_p2 = clf_minkowski_p2.fit(X_train_scaled, y_train).predict(X_test_scaled)
preds_euclidean = clf_euclidean.fit(X_train_scaled, y_train).predict(X_test_scaled)

print('Identical predictions:', np.array_equal(preds_minkowski_p2, preds_euclidean))
```

**Rule of thumb.** Leave `metric='minkowski'` and tune `p` if you want a numeric knob, or set `metric` explicitly to `'manhattan'` / `'euclidean'` / `'chebyshev'` if you prefer a self-documenting name in your code. The result is the same.

## 5. `algorithm` — the data structure used to find neighbours

Given a query point, KNN has to find its K closest training points. With $N$ training points and $D$ features, the naive way is to compute all $N$ distances and keep the K smallest — that costs $O(N D)$ per query. For small datasets, that is fine; for larger ones, **spatial index structures** can prune most of the search.

`algorithm` picks which structure to use:

- `'brute'` — the naive all-pairs scan. Most reliable, only choice for some exotic metrics, gets slow on large datasets.
- `'kd_tree'` — a **k-d tree**: recursively splits feature space on alternating axes. Fast in **low dimensions** (≲ 20 features); degenerates toward brute force in high dimensions.
- `'ball_tree'` — a **ball tree**: groups points into nested hyperspheres. Handles higher dimensions and more metrics than k-d tree.
- `'auto'` *(default)* — sklearn picks one of the above based on the metric, the number of features, and the size of the training set. This is the right default for almost everyone.

**Critical fact:** all four algorithms compute neighbour identities the same way given the same data and metric. They return the **same predictions**, just at different speeds. They are a fit-time / query-time optimisation, not a modelling choice.

```python
# Confirm: all four algorithms give identical predictions on iris.
predictions_by_algo = {}
for algo in ['auto', 'ball_tree', 'kd_tree', 'brute']:
    clf = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
    clf.fit(X_train_scaled, y_train)
    predictions_by_algo[algo] = clf.predict(X_test_scaled)

baseline = predictions_by_algo['brute']
for algo, preds in predictions_by_algo.items():
    print(f'algorithm={algo:>10s}   identical to brute? {np.array_equal(preds, baseline)}')
```

Predictions agree across all four. Now let's measure the **timing difference** on the larger synthetic dataset (10 000 samples, 20 features). We time `.fit()` and `.predict()` separately — they have different costs:

- `.fit()` for KNN just stores the data and (for tree algorithms) builds the index. `'brute'` is essentially free here; `'ball_tree'` / `'kd_tree'` pay an up-front cost.
- `.predict()` is where the choice pays off: the tree algorithms prune the search; `'brute'` walks every training point.

```python
timing_rows = []
for algo in ['auto', 'ball_tree', 'kd_tree', 'brute']:
    clf = KNeighborsClassifier(n_neighbors=5, algorithm=algo)

    t0 = time.perf_counter()
    clf.fit(X_big_train, y_big_train)
    fit_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    preds = clf.predict(X_big_test)
    predict_ms = (time.perf_counter() - t0) * 1000

    timing_rows.append({
        'algorithm': algo,
        'fit_ms': round(fit_ms, 1),
        'predict_ms': round(predict_ms, 1),
        'accuracy': round(accuracy_score(y_big_test, preds), 3),
    })

pd.DataFrame(timing_rows)
```

Notice three things in the table above:

1. **Accuracy is identical** across algorithms (last column). The choice does not affect what is predicted.
2. **Fit time** is tiny for `'brute'` (nothing to build) and larger for the tree variants (they build their index).
3. **Predict time** is where it actually matters. The tree algorithms can be substantially faster at predict-time for datasets where pruning is effective.

**When to reach for which algorithm:**

- `'auto'` — the right answer 95 % of the time. Trust sklearn's heuristic.
- `'kd_tree'` — small number of features (≲ 20) and many query points.
- `'ball_tree'` — more features, or a metric that k-d tree does not support.
- `'brute'` — very small datasets, or exotic metrics where indexes do not help. Also useful as a correctness baseline when debugging.

## 6. `leaf_size` — a tuning knob for the tree algorithms

`leaf_size` only affects `'ball_tree'` and `'kd_tree'` (it is ignored for `'brute'`). It controls **when the tree stops splitting** — once a node contains fewer than `leaf_size` points, the algorithm switches to a brute-force scan inside that leaf.

- **Small `leaf_size`** (e.g. `1`, `5`) — deeper tree, more index overhead at fit-time, but a more selective prune at query-time. Helps when the dataset is large.
- **Large `leaf_size`** (e.g. `100`, `500`) — shallower tree, cheaper to build, but each leaf falls back to brute search over many points.
- **Default `30`** — a good balance for most dense numeric datasets.

Like `algorithm`, `leaf_size` is a **speed knob, not a modelling choice** — the predictions are identical regardless of its value.

```python
# Confirm: leaf_size does not change predictions.
predictions_by_leaf = {}
for ls in [1, 10, 30, 100, 500]:
    clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', leaf_size=ls)
    clf.fit(X_big_train, y_big_train)
    predictions_by_leaf[ls] = clf.predict(X_big_test)

baseline = predictions_by_leaf[30]
for ls, preds in predictions_by_leaf.items():
    print(f'leaf_size={ls:4d}   identical to leaf_size=30? {np.array_equal(preds, baseline)}')
```

```python
# But it does change the fit/predict timings.
leaf_rows = []
for ls in [1, 10, 30, 100, 500]:
    clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', leaf_size=ls)

    t0 = time.perf_counter()
    clf.fit(X_big_train, y_big_train)
    fit_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    clf.predict(X_big_test)
    predict_ms = (time.perf_counter() - t0) * 1000

    leaf_rows.append({
        'leaf_size': ls,
        'fit_ms': round(fit_ms, 1),
        'predict_ms': round(predict_ms, 1),
    })

pd.DataFrame(leaf_rows)
```

The predictions are identical across all `leaf_size` values (first cell), but fit-time and predict-time shift (second cell). In practice you only ever tune `leaf_size` if `algorithm='ball_tree'` or `'kd_tree'` is a measured bottleneck — the default of `30` is fine for almost all datasets.

## 7. Summary — when to reach for each parameter

- **`weights`** — switch from `'uniform'` to `'distance'` when local density is uneven, when you have many close calls between classes, or when you have class imbalance. Watch out: it amplifies noisy labels.
- **`metric`** — keep the default (`'minkowski'` with `p=2`, i.e. Euclidean) unless you have a reason. Try `'manhattan'` in high dimensions or when features are independent grid coordinates; `'chebyshev'` is niche.
- **`p`** — only meaningful when `metric='minkowski'`. `p=1` gives Manhattan, `p=2` gives Euclidean. Set it instead of `metric` when you want a tunable hyperparameter.
- **`algorithm`** — leave at `'auto'`. The four choices give **identical predictions**; they only differ in fit/predict speed. Override only when you have a measured bottleneck.
- **`leaf_size`** — leave at `30`. Like `algorithm`, it never changes predictions — it is purely a speed knob for the tree algorithms.

**Take-away:** the parameters that move the model's *predictions* are `n_neighbors` (covered in `lec_10a`), `weights`, and `metric`/`p`. The parameters that move *only the speed* are `algorithm` and `leaf_size`. Knowing which is which is exactly what makes the difference between tuning a KNN pipeline blindly and tuning it on purpose.

**Further reading.** The official scikit-learn docs for [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) describe every attribute and every supported metric, and the [Neighbors user guide](https://scikit-learn.org/stable/modules/neighbors.html) explains the tree algorithms in detail.

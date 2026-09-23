<!-- source: lectures_07_13_pandas_plots_scikit/lecture_09_clustering_deploy_hf_app/reading_material/lec_09d_kmeans_other_parameters.ipynb @ 0cc874704aaa -->

# Lecture 09d. KMeans: the other useful parameters (besides `n_clusters`)

In `lec_09a` we focused on the most important hyperparameter of `KMeans`: **the number of clusters `K`**. But the `KMeans` constructor has several other parameters that quietly shape the result, the speed of the fit, and whether your experiments are reproducible.

Goal of this notebook: open up the `KMeans` constructor and **explain what every other parameter does, with a small example for each** on the mall customers dataset.

Full constructor (sklearn 1.8):

```python
KMeans(
    n_clusters=8,
    *,
    init='k-means++',
    n_init='auto',
    max_iter=300,
    tol=0.0001,
    verbose=0,
    random_state=None,
    copy_x=True,
    algorithm='lloyd',
)
```

We will work through these in the order in which they actually matter in practice:

1. `init` — how the first centroids are chosen.
2. `n_init` — how many independent runs to try.
3. `random_state` — controls the randomness for reproducibility.
4. `max_iter` — cap on iterations within one run.
5. `tol` — when to stop refining.
6. `algorithm` — `'lloyd'` vs `'elkan'`.

We will skip `verbose` (just prints progress) and `copy_x` (a memory/numerical-precision knob that rarely needs touching).

## 1. Imports and data setup

Same dataset as `lec_09a`. We do the minimum preprocessing: drop `CustomerID`, encode `Genre` as numeric `Gender`, and keep four features.

```python
import time

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

```python
df = pd.read_csv('mall_customers.csv')
df = df.drop('CustomerID', axis=1)
df['Gender'] = pd.get_dummies(df['Genre'], drop_first=True, prefix='Genre')

FEATURES = ['Age', 'Annual_Income_(k$)', 'Spending_Score', 'Gender']
X = df[FEATURES].values
print('X shape:', X.shape)
df.head(3)
```

## 2. `init` — how the first centroids are chosen

`KMeans` is an iterative algorithm: it starts from a guess of where the centroids are, then refines them. **The starting guess matters.** A bad start can make the algorithm converge to a worse local minimum.

`init` controls how that initial guess is made. Three options:

- `'k-means++'` *(default)*: a smart procedure that picks initial centroids that are far   apart from each other. The first centroid is a random point; each subsequent centroid is   chosen with probability proportional to its squared distance from the closest already-chosen   centroid. This usually converges faster and to a better solution.
- `'random'`: pick `n_clusters` rows of `X` uniformly at random. Simple, but if you happen   to pick two starting centroids in the same dense region you can get stuck in a poor local   minimum.
- A **NumPy array** of shape `(n_clusters, n_features)`: force the algorithm to start from   centroids you specify. Useful when you have prior knowledge — for example, when you   re-train a model on new data and want to keep the cluster identities stable.

Below we fit the same K (=6) with `'k-means++'` and `'random'` and compare inertia (lower is better) and silhouette (higher is better).

```python
def fit_and_score(X, **kwargs):
    km = KMeans(n_clusters=6, random_state=0, **kwargs)
    km.fit(X)
    sil = silhouette_score(X, km.labels_)
    return km.inertia_, sil, km.n_iter_

for init in ['k-means++', 'random']:
    inertia, sil, n_iter = fit_and_score(X, init=init, n_init=1)
    print(f'init={init:>10s}  inertia={inertia:8.1f}  silhouette={sil:.3f}  iterations={n_iter}')
```

With a single run (`n_init=1`), `'k-means++'` converges in fewer iterations than `'random'` (5 vs 9 here) because it starts from a smarter spread of centroids. The final inertia and silhouette can swing either way on a single seed — that variance is exactly what `n_init` (next section) averages out.

**Custom-array init.** You can also hand `init` a NumPy array — this is the most precise way to control where the algorithm starts:

```python
# Use the means of three deliberately chosen subgroups as the seed centroids.
young_low = df[(df['Age'] < 30) & (df['Spending_Score'] < 40)][FEATURES].mean().values
young_high = df[(df['Age'] < 30) & (df['Spending_Score'] > 60)][FEATURES].mean().values
older_mid  = df[df['Age'] > 50][FEATURES].mean().values

seed_centroids = np.vstack([young_low, young_high, older_mid])

km_custom = KMeans(n_clusters=3, init=seed_centroids, n_init=1, random_state=0)
km_custom.fit(X)
print('Final centroids (Age, Income, Spending, Gender):')
print(np.round(km_custom.cluster_centers_, 1))
```

## 3. `n_init` — how many independent runs to try

Even with `'k-means++'`, the algorithm can land on different local minima depending on the initial random draw. `n_init` says: **fit the algorithm this many times from different random seeds, and keep the run with the lowest inertia.**

- `n_init=1`: a single fit. Fastest, most variable.
- `n_init=10` (and similar small integers): much more reliable, ~10× the work.
- `n_init='auto'` *(default in sklearn ≥ 1.4)*: picks `1` for `'k-means++'` or a custom   array (which already start from a good place) and `10` for `'random'`. This is the   sensible modern default.

Below we run `KMeans` with `n_init=1` ten times, varying only the seed, to see how much the inertia *can* swing — then we run it once with `n_init=10` to confirm we land on (or very close to) the best of those.

```python
single_run_inertias = []
for seed in range(10):
    km = KMeans(n_clusters=6, init='random', n_init=1, random_state=seed)
    km.fit(X)
    single_run_inertias.append(km.inertia_)

best = KMeans(n_clusters=6, init='random', n_init=10, random_state=0).fit(X)

print('Inertia across 10 single-init random runs:')
print('  min  =', round(min(single_run_inertias), 1))
print('  max  =', round(max(single_run_inertias), 1))
print('  mean =', round(sum(single_run_inertias) / 10, 1))
print()
print('Inertia from one fit with n_init=10:', round(best.inertia_, 1))
```

Notice how the single-run inertia varies from one seed to another — that's the variance `n_init` averages out for you. The `n_init=10` fit matches (or beats) the best of those ten single runs.

**Rule of thumb.** Leave `n_init='auto'` unless you have a reason to do otherwise. Increase it (e.g. to 20 or 50) when you suspect your data has many similar local minima and you want to be extra safe.

## 4. `random_state` — reproducibility

`KMeans` is randomized: the centroid initialization step uses random numbers. `random_state` is the seed for that randomness.

- `random_state=None` *(default)*: a different seed every time. Two consecutive fits on   the same data may give slightly different cluster labels and centroids.
- `random_state=<integer>`: fully reproducible. Same data + same code + same seed =   identical centroids and identical labels every time.

**Always set `random_state` for teaching, debugging, papers, and tests** — it is the only way to get the exact same numbers in the output that we discuss in the text.

```python
# Without random_state, results may shift between runs (especially with init='random').
labels_run1 = KMeans(n_clusters=6, init='random', n_init=1).fit_predict(X)
labels_run2 = KMeans(n_clusters=6, init='random', n_init=1).fit_predict(X)
agree = (labels_run1 == labels_run2).mean()
print(f'Two unseeded runs agree on {agree*100:.0f}% of points '
      "(remember: cluster *labels* are arbitrary integers — they may permute even when the partition is the same).")

# With random_state, results are bit-for-bit identical.
a = KMeans(n_clusters=6, random_state=42, n_init=10).fit(X)
b = KMeans(n_clusters=6, random_state=42, n_init=10).fit(X)
print('Identical centroids with same random_state:',
      np.allclose(a.cluster_centers_, b.cluster_centers_))
```

## 5. `max_iter` — cap on iterations within one run

Each run of KMeans alternates two steps until convergence:

1. **Assign** every point to its nearest centroid.
2. **Update** every centroid to the mean of the points assigned to it.

`max_iter` caps how many of these assign+update cycles a single run is allowed. Default is `300`, which is far more than enough for almost any reasonable dataset — KMeans on tabular data of this size usually converges in single-digit iterations.

Why would you change it?
- **Lower it** to see the algorithm mid-run (educational purpose) or to bound the   computation in a real-time setting.
- **Raise it** if you set `tol=0` and want to fight every last decimal of inertia (rare).

Below we restrict KMeans to 1, 2, 5, and 300 iterations and watch the inertia drop and then plateau.

```python
rows = []
for m in [1, 2, 3, 5, 10, 300]:
    km = KMeans(n_clusters=6, n_init=1, max_iter=m, tol=0,
                random_state=0).fit(X)
    rows.append({'max_iter': m, 'inertia': round(km.inertia_, 1),
                 'iterations_used': km.n_iter_})
pd.DataFrame(rows)
```

After just a handful of iterations the inertia is already very close to its final value. Notice that `iterations_used` is capped by `max_iter` until the algorithm has truly converged — at that point it stops on its own and `iterations_used` becomes smaller than `max_iter`.

## 6. `tol` — when to stop refining

Even before `max_iter` is reached, KMeans stops as soon as the centroids barely move between two iterations. Specifically, it stops when the **relative change in the centroid positions** falls below `tol` (technically, the Frobenius norm of the centroid shift, scaled by the variance of `X`).

- Default `tol=1e-4` — generous; the algorithm stops well before machine precision.
- Larger `tol` (e.g. `1e-2`, `1`) → stops earlier, fewer iterations, slightly worse   inertia.
- Smaller `tol` (e.g. `1e-10`) → keeps refining until improvements are tiny; usually   no practical benefit on real data.
- `tol=0` → only stop when centroids do not move at all (or `max_iter` is hit).

Below we vary `tol` from very loose to very strict and look at iterations used and final inertia.

```python
rows = []
for t in [1.0, 1e-1, 1e-2, 1e-4, 1e-8, 0.0]:
    km = KMeans(n_clusters=6, n_init=1, tol=t, max_iter=500,
                random_state=0).fit(X)
    rows.append({'tol': t, 'iterations_used': km.n_iter_,
                 'inertia': round(km.inertia_, 2)})
pd.DataFrame(rows)
```

On this dataset the inertia is essentially the same once `tol ≤ 1e-2`. Tightening `tol` further only buys more iterations, not a meaningfully better solution. The default of `1e-4` is well chosen — leave it alone unless you are doing something unusual.

## 7. `algorithm` — `'lloyd'` vs `'elkan'`

Both options produce the **same final clustering** (modulo floating-point noise). They differ only in how they compute the assign step internally.

- `'lloyd'` *(default)*: the classical algorithm. Each iteration computes the distance   from every point to every centroid. Simple and memory-light.
- `'elkan'`: uses the **triangle inequality** to skip distance computations that   cannot possibly change a point's nearest-centroid assignment. Often faster on dense   data with a small number of clusters, but uses extra memory to cache distance bounds.

On a tiny dataset like ours the difference is microseconds either way. Below we make a larger synthetic copy of `X` (10 000 points) so the timing difference is visible, then fit with each algorithm and confirm the inertias agree.

```python
rng = np.random.default_rng(0)
X_big = np.vstack([X + rng.normal(scale=0.5, size=X.shape) for _ in range(50)])
print('X_big shape:', X_big.shape)

for algo in ['lloyd', 'elkan']:
    t0 = time.perf_counter()
    km = KMeans(n_clusters=8, algorithm=algo, n_init=1,
                random_state=0).fit(X_big)
    elapsed = time.perf_counter() - t0
    print(f'algorithm={algo:6s}  time={elapsed*1000:6.1f} ms  inertia={km.inertia_:.1f}')
```

The two algorithms reach the same inertia, just with different timings. Which one wins depends on `n_clusters`, the number of features, and the data density — `'elkan'` tends to win when `n_clusters` is small relative to the number of points, while `'lloyd'` is preferable for sparse data or very large `n_clusters`. In practice you only need to switch if you have measured a real performance bottleneck.

## 8. Summary — what to actually set, in practice

| Parameter      | Default        | When to change it                                                                  |
|----------------|----------------|------------------------------------------------------------------------------------|
| `init`         | `'k-means++'`  | Use a custom array when you want to seed centroids from prior knowledge.           |
| `n_init`       | `'auto'`       | Increase to 20–50 if you suspect many similar local minima.                        |
| `random_state` | `None`         | **Always set an integer** for reproducible experiments and teaching.               |
| `max_iter`     | `300`          | Rarely change. Lower it for educational mid-run snapshots.                         |
| `tol`          | `1e-4`         | Rarely change. Loosen it only if you need every millisecond of speed.              |
| `algorithm`    | `'lloyd'`     | Try `'elkan'` if KMeans is a measured bottleneck and clusters are few.             |

**Take-away:** for nearly every project, the only knobs you actually touch are `n_clusters` (covered in `lec_09a`) and `random_state` (so your results are reproducible). The rest are good defaults that survive almost any real-world dataset.

**Further reading.** The official scikit-learn docs for [`KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) describe every attribute returned by `.fit()` (`labels_`, `cluster_centers_`, `inertia_`, `n_iter_`) and link to [`MiniBatchKMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) for very large datasets.

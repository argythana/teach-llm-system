<!-- source: lectures_07_13_pandas_plots_scikit/lecture_10_knn_train_test_split/reading_material/lec_10f_knn_regression_teaser.ipynb @ 0cc874704aaa -->

# Lecture 10f — KNN regression teaser

> *Optional / Career-track* — read **after** the mandatory `lec_10a` / `lec_10b` / `lec_10c`. Pairs well with `lec_10d_knn_other_parameters.ipynb` (the parameters carry over verbatim) but does not depend on it.

Everything in lectures 10a–10c has been about **classification** — predicting a discrete label like `'setosa'` or `'virginica'`. The same KNN machinery works for **regression** — predicting a continuous number — with one small change: instead of *voting* on the K neighbours' classes, you *average* their target values.

In scikit-learn the regressor lives in the same submodule as the classifier:

```python
from sklearn.neighbors import KNeighborsRegressor
```

It takes the same constructor parameters covered in `lec_10d` (`n_neighbors`, `weights`, `metric`, `p`, `algorithm`, `leaf_size`) — they mean the same things and behave the same way. The only API differences are: `.predict()` returns a float (or an array of floats), and there is no `.predict_proba()` or `.classes_`.

This notebook is a **quick visual walkthrough** so you can *see* how the algorithm looks in regression mode. The deep regression treatment — linear models, the validation set, bias-variance — is **Lecture 11**.

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
```

```python
# Synthetic 1D problem: a noisy sine wave. Perfect for visualisation because we can
# plot the model's prediction curve against the data.
rng = np.random.RandomState(42)
X_reg = np.sort(5 * rng.rand(80, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + 0.1 * rng.randn(80)

X_plot = np.linspace(0, 5, 500).reshape(-1, 1)  # dense grid for smooth prediction lines
print(f'X_reg: {X_reg.shape}   y_reg: {y_reg.shape}')
```

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, k in zip(axes, [2, 5, 15]):
    reg = KNeighborsRegressor(n_neighbors=k)
    reg.fit(X_reg, y_reg)
    ax.scatter(X_reg, y_reg, s=20, alpha=0.6, label='training data')
    ax.plot(X_plot, reg.predict(X_plot), color='red', linewidth=2, label=f'KNN prediction (K={k})')
    ax.set_title(f'K = {k}')
    ax.set_xlabel('x')
    ax.legend(loc='lower left')
axes[0].set_ylabel('y')
plt.tight_layout()
plt.show()
```

**Same K trade-off you already know from classification.** Small K (=2) chases every training point — the prediction curve is wiggly and over-fits the noise. Large K (=15) averages so many neighbours that the curve flattens, missing the genuine sine structure (under-fitting). Somewhere around K=5 the curve recovers the underlying signal cleanly.

The aggregation rule is the only difference vs. classification:

- **Classification**: take the K neighbours' class labels and vote (most common class wins).
- **Regression**: take the K neighbours' target values and **average** them.

Everything else — distance metric, weighting, algorithm choice, scaling discipline — is identical. The §2 / §3 sweeps from `lec_10d` translate directly: the same trade-offs that move classification accuracy also move regression MSE.

```python
# Confirm the visual story numerically. We score on the training set here purely to
# show the K-vs-fit trade-off; honest evaluation needs a held-out test set, which is
# the topic of lecture 11.
for k in [1, 2, 5, 15, 50, 80]:
    reg = KNeighborsRegressor(n_neighbors=k).fit(X_reg, y_reg)
    mse = mean_squared_error(y_reg, reg.predict(X_reg))
    print(f'K = {k:2d}:  training MSE = {mse:.4f}')
```

**When to reach for KNN regression instead of a linear or tree-based regressor:**

- The target is **locally smooth** but globally non-linear — sine curves, calibration curves, dose-response data.
- You have a small number of features and enough data that K ≈ 5–15 covers a meaningful neighbourhood.
- You want a non-parametric baseline before fitting a "real" regression model.

**When to avoid it:**

- Same failure modes as the classifier: scale sensitivity, curse of dimensionality, prediction-time cost at large N (see `lec_10b_knn_assumptions_caveats.ipynb`).
- **Extrapolation**: KNN cannot predict outside the range of training-data targets. If your training `y` runs from 0 to 10 but the true relationship goes to 100 at the edge of the feature space, KNN will cap at the highest training value — it has no way to extend the pattern outwards.

The full regression chapter — including the held-out evaluation we glossed over in the MSE table above — is **Lecture 11**.

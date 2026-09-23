<!-- source: lectures_07_13_pandas_plots_scikit/lecture_11_regression_linear_polynomial/reading_material/lec_11f_regression_advanced_topics.ipynb @ 0cc874704aaa -->

# Lecture 11f — Advanced regression topics

> 🎓 **Status: Optional / career-track.** Not walked through in class. Not required for the next lecture. Covers diagnostic and modelling techniques you will routinely meet in a data-science or analytics role — but which beginners do not need on day one.

It extends the mandatory `lec_11a`–`lec_11d` (and the optional `lec_11e`) with three diagnostic and modelling techniques you will routinely meet in a data-science or analytics role.

## What this notebook covers

- **§B — Variance Inflation Factor (VIF).** A principled collinearity diagnostic; the standard inferential complement to `lec_11d`'s bootstrap.
- **§C — ElasticNet.** The L1 + L2 hybrid that the read_agents reading and `lec_11e §F` flagged as the natural extension of Ridge and Lasso.
- **§D — Robust regression (Huber, Theil-Sen).** What to fit when the residuals are not normal — when a handful of outliers would dominate plain OLS.
- **§E — Where to go next.** Pointer table for further study: Bayesian regression, generalised additive models (GAMs), and the cross-validated tooling of Lecture 13.

## What this notebook does *not* do

- It does not re-teach polynomial features, regularization basics, or coefficient pitfalls — read `lec_11d` and `lec_11e` first.
- It does not cover cross-validation or `GridSearchCV` — those are Lecture 13.

## Career-track framing

- **VIF** is the table you produce when someone in a meeting asks *"why did the coefficient flip when we dropped that feature?"*.
- **ElasticNet** is the regulariser large-scale prediction shops reach for first when they want some sparsity but cannot tolerate Lasso dropping correlated features arbitrarily.
- **Robust regression** is the answer to *"the model's R² collapsed when we got that one weird customer in our dataset"*.

None of the three is required for the next lecture. All three are routinely used on the job.

## §A. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, TheilSenRegressor, RANSACRegressor, BayesianRidge,
    QuantileRegressor,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
sns.set_theme(style="whitegrid")

def vif_manual(X_features):
    """Compute VIF for each column of X_features (DataFrame) using OLS.

    VIF_j = 1 / (1 - R_j^2), where R_j^2 is the R-squared from regressing
    feature j on every other feature. High VIF means the feature is
    well-predicted by the others -- collinearity risk.
    """
    out = {}
    for j in X_features.columns:
        X_others = X_features.drop(columns=[j]).values
        y_target = X_features[j].values
        r2 = LinearRegression().fit(X_others, y_target).score(X_others, y_target)
        out[j] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
    return pd.Series(out, name="VIF").sort_values(ascending=False)
```

## §B. Variance Inflation Factor (VIF) — the principled collinearity diagnostic

`lec_11d §P3` used the bootstrap to detect coefficient instability under correlated features. The bootstrap is non-parametric and visual; the **VIF** is its parametric, single-number complement.

**Definition.** For feature `Xj`, fit a regression of `Xj` against every other feature. Let R²ⱼ be its R². Then

> VIFⱼ = 1 / (1 − R²ⱼ)

When `Xj` is **uncorrelated** with the others, R²ⱼ ≈ 0 and VIF ≈ 1 (no inflation). When `Xj` is well-predicted by the others (R²ⱼ → 1), VIF blows up.

**Rules of thumb.**

| VIF | Interpretation |
| --- | --- |
| < 5 | Tolerable collinearity |
| 5–10 | Suspect; investigate |
| > 10 | Severe; consider dropping or combining features |

```python
df = pd.read_excel("grades_factors.xlsx")
df.columns = df.columns.str.replace(" ", "_").str.lower()
feat_cols = ["calc_hs", "act_math", "alg_place", "alg2_grade", "hs_rank", "gender_code"]
X_grades = df[feat_cols].astype(float)
```

```python
vif_grades = vif_manual(X_grades).round(2)
vif_grades
```

### §B1. Where does the VIF number come from?

We used a small helper `vif_manual` in the setup cell.
The formula behind it is a one-pager you can implement in five lines — and seeing it makes the diagnostic stop feeling like magic.

For each feature `Xⱼ`:

1. Fit an OLS regressing `Xⱼ` on every *other* feature.
2. Record that fit's R² — call it `Rⱼ²`.
3. `VIFⱼ = 1 / (1 − Rⱼ²)`.

**Intuition.** If the other features can already predict `Xⱼ` well (high `Rⱼ²`), then `Xⱼ` carries little *independent* information; its coefficient in the main regression has to be teased out from a sliver of variance, and the variance of the estimate inflates.

`statsmodels` ships an equivalent function (`statsmodels.stats.outliers_influence.variance_inflation_factor`); the numbers match `vif_manual` up to a small difference in how the constant column is handled.

**Reading the grades_factors VIFs.**

All six numbers sit between 1.0 and 1.8 — well below the "5 = suspect" line.
On this dataset the math-ability proxies (`calc_hs`, `act_math`, `alg2_grade`) are *correlated* (the §D heatmap of `lec_11c` shows it) but not *collinear* in the structural sense VIF detects: each one carries enough independent information that the others cannot reconstruct it. The non-parametric bootstrap in `lec_11d §P3` reaches the same conclusion from the visual side.

That makes `grades_factors` a useful **benign baseline** — you see what "no problem" looks like. To see what the diagnostic does when it *does* fire, and to walk the question of **how to fix high VIF without throwing away information**, we move to California housing.

> **Aside on tolerance.** `tolerance = 1 / VIF` is the same number with a different name; SPSS, Stata, and R commonly report tolerance instead. Read it as *"the fraction of `Xⱼ` that is **not** explained by the other features"*.

### §B2. Where VIF actually fires — California housing

California housing has *two* known collinearity pairs:

- **`AveRooms` / `AveBedrms`** — bedrooms are a subset of rooms; the two correlate at 0.85. Bedrooms is *informationally* redundant given total rooms. Dropping it loses almost nothing.
- **`Latitude` / `Longitude`** — California's geography means latitude and longitude correlate ≈ -0.92 (anything north of San Francisco is also west of LA). But Latitude and Longitude are **NOT informationally redundant** — they jointly encode 2D position. Dropping either *destroys* half of "where the house is".

The Lat/Long case is the interesting one and the place where students go wrong. VIF flags the coordinates as collinear, but the answer is **not** to drop. It is to **refactor the pair into one or more engineered features that preserve location while breaking the linear collinearity**. The cells below walk both pairs in order: the easy redundant pair first, then the geographic pair — with a numerical demo of what the wrong fix actually costs.

```python
# Fetch California housing (same dataset used later in §F).
cal = fetch_california_housing(as_frame=True)
X_cal = cal.frame.drop(columns=["MedHouseVal"])
y_cal = cal.frame["MedHouseVal"]

vif_manual(X_cal).round(2)
```

**Reading the table.**

| Feature | VIF | Verdict |
| --- | --- | --- |
| `Latitude`, `Longitude` | ≈ 9 each | **Suspect.** -0.92 geographic correlation. |
| `AveRooms`, `AveBedrms` | ≈ 8 and 7 | **Suspect.** Bedrooms are a subset of rooms. |
| `MedInc` | 2.5 | Tolerable. |
| `HouseAge`, `Population`, `AveOccup` | ≈ 1 | Independent. |

**The redundant pair (rooms / bedrooms).** Drop `AveBedrms`. Bedrooms are *informationally* redundant given total rooms — domain knowledge says so, and dropping costs essentially zero R². This is the standard recipe.

**The geographic pair (Lat / Lon) is different — and this is where students go wrong.** Both coordinates are essential because they jointly encode 2D position. Tobler (1970) [C1] framed it as the *first law of geography* — near things are more related than distant things — so *both* coordinates carry irreducible information that pairwise correlation does not capture. The source paper for this exact dataset, Pace & Barry (1997) [C2], demonstrated empirically: a spatial-autoregressive model that uses location lifts R² from 0.61 (plain OLS) to 0.86 — a 25-point gap that is the cost of treating location naively.

We must combine Latitude and Longitude into one or more *new* features that span the same geographic information without the linear collinearity. The cell below shows what happens if we instead take the lazy shortcut and just drop a coordinate.

```python
# WRONG FIX -- shown so we can see what it costs.
# Drop Longitude to make the VIFs behave; watch the R^2 collapse.
X_dropLong = X_cal.drop(columns=["Longitude"])

print("VIFs after dropping Longitude:")
print(vif_manual(X_dropLong).round(2))
print()
print(f"R^2 with all features (incl. Lat & Lon): {LinearRegression().fit(X_cal, y_cal).score(X_cal, y_cal):.4f}")
print(f"R^2 after dropping Longitude:            {LinearRegression().fit(X_dropLong, y_cal).score(X_dropLong, y_cal):.4f}")
```

**Reading the output.**

Dropping `Longitude` makes every VIF behave (all below 8) — but the cost shows up where VIF cannot see it: **R² falls from 0.6062 to 0.5427 — six percentage points of variance explained, lost.** That gap is the price of throwing away half of "where". VIF was satisfied; the model is worse.

The right move is to **refactor `(Latitude, Longitude)` into a different parameterisation** that preserves the joint geographic information while breaking the linear collinearity. The peer-reviewed taxonomy in Mai et al. (2022) [C3] groups the options into:

1. **k-means clustering of `(Lat, Lon)`** → one-hot cluster IDs. Discretises geography into *k* regions; the cluster indicator is a single categorical feature (expanded to *k − 1* dummies). **Works in scikit-learn out of the box**, no extra dependencies. Demonstrated below.
2. **Geohash / H3 hexagonal cells** — industry-standard hierarchical spatial indexing; same principle as k-means but with predefined geographic cells. Requires `h3-py` or similar.
3. **Distance to landmarks** (e.g., great-circle distance to coast, distance to nearest major metro) — engineered scalar features. *Subtle pitfall:* a pure *linear* reparameterisation of `(Lat, Lon)` (such as `(Lon − coast_lon) * 111 km * cos(Lat)`) does **not** reduce VIF — the new column spans the same 2D linear subspace as the original. The fix has to be a *nonlinear* transform (great-circle distance to a point, polar coordinates with a non-origin pivot) to actually break the collinearity.
4. **Skip the gymnastics: regularise or use a proper spatial model.** Ridge / ElasticNet (§C next) handles collinearity by construction; the spatial-autoregressive approach in Pace & Barry (1997) [C2] is implemented in PySAL [C4] for the Python ecosystem and lifted R² from 0.61 to 0.86 on this exact dataset.

We demo option 1 — k-means cluster IDs — because it requires nothing beyond what is already imported.

```python
# RIGHT FIX (approach 1 from the list above).
# Combine (Latitude, Longitude) into a single CATEGORICAL feature via
# k-means clustering, then one-hot encode the cluster IDs. Both
# coordinates' information is preserved -- in 10 region indicators
# instead of 2 real-valued columns.
from sklearn.cluster import KMeans

km = KMeans(n_clusters=10, n_init=10, random_state=RANDOM_STATE).fit(
    X_cal[["Latitude", "Longitude"]]
)
cluster_dummies = pd.get_dummies(km.labels_, prefix="geo_cluster", drop_first=True).astype(float)
cluster_dummies.index = X_cal.index

X_cal_eng = X_cal.drop(columns=["Latitude", "Longitude"]).join(cluster_dummies)

print("VIFs after replacing (Lat, Lon) with k=10 one-hot cluster dummies:")
print(vif_manual(X_cal_eng).round(2).head(12))
print()
print(f"R^2 with raw (Lat, Lon):           {LinearRegression().fit(X_cal, y_cal).score(X_cal, y_cal):.4f}")
print(f"R^2 with k-means cluster IDs:      {LinearRegression().fit(X_cal_eng, y_cal).score(X_cal_eng, y_cal):.4f}")
```

**Reading the k-means output.**

The nine `geo_cluster_*` dummies all come back with VIFs below 4 — well within tolerable. The `AveRooms` / `AveBedrms` pair is still high (that is the *other* collinearity; the easy fix is to drop `AveBedrms`). And — the punchline — **R² holds up: 0.5960 with cluster IDs vs 0.6062 with raw `(Lat, Lon)`** — only one percentage point lost, compared with the six points lost by the drop-Longitude shortcut.

The k-means approach gives us geography we can interpret (each cluster is a region of California you can colour on a map) AND coefficients that are stable enough to read. That is the principled answer to a high-VIF geographic feature pair.

**One philosophical aside.** VIF is an **inference** diagnostic — it tells you whether you can trust the *coefficient table*. For pure *prediction*, the raw-coord model is actually fine (R² = 0.6062 is the *highest* of our three scenarios). If you only care about predictions, **Ridge / ElasticNet on the raw coords** sidesteps the problem entirely: regularisation handles unstable coefficients without rebalancing the features. The right "fix" depends on whether you need interpretable coefficients (engineer or discretise) or just good predictions (regularise; §C next).

---

**References for §B (verified via the do-literature-review skill):**

- **[C1] Tobler, W. R. (1970).** A computer movie simulating urban growth in the Detroit region. *Economic Geography*, 46, 234. DOI: [10.2307/143141](https://doi.org/10.2307/143141). The foundational citation for the *first law of geography*: "everything is related to everything else, but near things are more related than distant things." Used here as a pointer — the source paper itself sits behind a JSTOR paywall.
- **[C2] Pace, R. K., & Barry, R. (1997).** Sparse spatial autoregressions. *Statistics & Probability Letters*, 33(3), 291–297. DOI: [10.1016/S0167-7152(96)00140-X](https://doi.org/10.1016/S0167-7152(96)00140-X). The source paper for the California housing dataset. They showed a spatial-autoregressive model lifts R² from 0.6078 (plain OLS, no spatial structure) to 0.8594 — direct empirical evidence that location carries roughly 25 R² points on this dataset and must not be discarded.
- **[C3] Mai, G., Janowicz, K., Hu, Y., Gao, S., Yan, B., Zhu, R., Cai, L., & Lao, N. (2022).** A review of location encoding for GeoAI: methods and applications. *International Journal of Geographical Information Science*, 36(4), 639–673. DOI: [10.1080/13658816.2021.2004602](https://doi.org/10.1080/13658816.2021.2004602). Open-access preprint at [arXiv:2111.04006](https://arxiv.org/abs/2111.04006). The peer-reviewed taxonomy of location-encoding approaches — traditional discretisation (Geohash, H3, Open Location Code, what3words) versus modern neural location encoders.
- **[C4] Rey, S. J., & Anselin, L. (2007).** PySAL: A Python library of spatial analytical methods. *Review of Regional Studies*, 37(1), 5–27. DOI: [10.52324/001c.8285](https://doi.org/10.52324/001c.8285). Open access. The canonical Python coding reference for the spatial machinery beyond scikit-learn — spatial weights matrices, ESDA, and the spatial-autoregression workflow that Pace & Barry pioneered.

## §C. ElasticNet — the L1 + L2 hybrid

`lec_11e` taught:

- **Ridge** (L2) — shrinks all coefficients.
- **Lasso** (L1) — zeros some out.

**ElasticNet** combines both:

> minimise   RSS  +  α · [ρ · ‖β‖₁  +  (1 − ρ) · ‖β‖²]

where `ρ` (sklearn's `l1_ratio`) is the L1 mixing weight:

- `l1_ratio=1` → pure Lasso,
- `l1_ratio=0` → pure Ridge,
- in-between values → mix.

**Why use ElasticNet over pure Lasso.**
When features are highly correlated, pure Lasso picks **one** representative from the cluster and zeros the rest — arbitrarily, depending on tiny noise differences.
ElasticNet's L2 component shares credit *across* the correlated cluster, which gives a more stable feature-selection answer.

```python
# Use California housing (same as lec_11e §D) so the comparison is direct.
data = fetch_california_housing(as_frame=True)
X_full = data.data
y_full = data.target

X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y_full, test_size=0.20, random_state=RANDOM_STATE,
)

# Polynomial features (deg 2) + scaling on train only.
poly = PolynomialFeatures(degree=2, include_bias=False)
X_tr_p = poly.fit_transform(X_tr)
X_te_p = poly.transform(X_te)

sc = StandardScaler().fit(X_tr_p)
X_tr_s = sc.transform(X_tr_p)
X_te_s = sc.transform(X_te_p)

print(f"Feature count after poly expansion: {X_tr_s.shape[1]}")
```

```python
rows = []
for name, mdl in [
    ("Ridge(alpha=1)",                       Ridge(alpha=1.0)),
    ("Lasso(alpha=0.01)",                    Lasso(alpha=0.01, max_iter=20000)),
    ("ElasticNet(alpha=0.01, l1_ratio=0.5)", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000)),
    ("ElasticNet(alpha=0.01, l1_ratio=0.9)", ElasticNet(alpha=0.01, l1_ratio=0.9, max_iter=20000)),
]:
    mdl.fit(X_tr_s, y_tr)
    nz = int(np.sum(np.abs(mdl.coef_) > 1e-8))
    rows.append({
        "model": name,
        "train_R2": mdl.score(X_tr_s, y_tr),
        "test_R2":  mdl.score(X_te_s,  y_te),
        "nonzero_coefs": nz,
    })

pd.DataFrame(rows).round(3)
```

### §C1. The Lasso path — watch coefficients shrink to zero

A single `alpha` value gives a single coefficient vector.  
But what we usually want to *see* is **how** the coefficients move as `alpha` grows — which features fight hardest to stay non-zero, which fold immediately.  
The *"Lasso path"* plot has one line per feature and `alpha` on the x-axis.  
It is one of the most informative diagnostic plots a regression workflow produces.

```python
alphas = np.logspace(-3, 1, 30)
paths = []
for a in alphas:
    lo = Lasso(alpha=a, max_iter=20000).fit(X_tr_s, y_tr)
    paths.append(lo.coef_)
paths = np.array(paths)  # shape (n_alphas, n_features)

fig, ax = plt.subplots(figsize=(7, 4))
# Plot every feature; highlight the 8 strongest by final magnitude
final_strength = np.abs(paths[0])  # at smallest alpha (closest to OLS)
top_idx = np.argsort(final_strength)[-8:]
for j in range(paths.shape[1]):
    color = plt.cm.tab10(list(top_idx).index(j) % 10) if j in top_idx else "lightgrey"
    lw = 2.0 if j in top_idx else 0.5
    ax.plot(alphas, paths[:, j], color=color, linewidth=lw)
ax.set_xscale("log")
ax.set_xlabel("alpha (log scale)")
ax.set_ylabel("coefficient value")
ax.set_title("Lasso path on poly(deg=2) California housing — coefficients shrink as alpha grows")
ax.axhline(0, color="black", linewidth=0.4)
plt.tight_layout()
plt.show()
```

Reading this plot in production:

- **Far left (small α)** — close to plain OLS; most coefficients non-zero, many of them small.
- **Middle** — features peel off the zero line in *reverse* order of importance; the last 5–8 lines that remain non-zero are the features Lasso most wants to keep.
- **Far right (large α)** — every line has collapsed to zero; the model has degenerated to the intercept-only baseline.

The "right" α for your problem lives in the middle band:

- close enough to OLS to retain predictive power,
- far enough into shrinkage that the surviving coefficients tell a clean story.

Lecture 13 picks α with `LassoCV` (cross-validated). Until then, eyeballing the path against a held-out validation MSE is a defensible substitute.

Read the table left to right. Ridge keeps every coefficient (`nonzero = max`); Lasso aggressively zeros. ElasticNet at `l1_ratio=0.5` lies between the two — partial sparsity. At `l1_ratio=0.9` it behaves close to Lasso but with a Ridge backbone that stabilises the cluster of correlated polynomial features.

**Practical recipe.** When in doubt: start with `l1_ratio=0.5`. Cross-validate `alpha` (Lecture 13 covers `ElasticNetCV`). If sparsity matters more than stability, push `l1_ratio` toward 1; if stability matters more, push it toward 0.

## §D. Robust regression — when outliers dominate plain OLS

OLS minimises *squared* residuals.
A single outlier with a residual of 10 contributes 100 to the loss; an inlier with residual 1 contributes 1.
One rogue point can tilt the entire fit.

**Two robust alternatives** in scikit-learn:

- **Huber regression** — quadratic loss for small residuals, linear loss above a threshold `epsilon`. A controlled compromise between OLS (squared everywhere) and least-absolute-deviation regression.
- **Theil-Sen regression** — fits the median slope over all sample pairs. Tolerates up to ~29.3% contamination but is O(n²) in samples — unsuitable for large data.

Demo: 50 clean points + 5 outliers. Watch what happens to plain OLS.

```python
# 50 clean points around y = 2 + 1.3*x
n_clean = 50
x_clean = rng.uniform(0, 10, size=n_clean)
y_clean = 2.0 + 1.3 * x_clean + rng.normal(0, 0.6, size=n_clean)

# 5 outliers — extreme y, normal x
n_out = 5
x_out = rng.uniform(0, 10, size=n_out)
y_out = rng.uniform(20, 30, size=n_out)
x = np.concatenate([x_clean, x_out]).reshape(-1, 1)
y = np.concatenate([y_clean, y_out])

# Three fits.
fits = {
    "OLS":       LinearRegression().fit(x, y),
    "Huber":     HuberRegressor().fit(x, y),
    "Theil-Sen": TheilSenRegressor(random_state=RANDOM_STATE).fit(x, y),
}

grid = np.linspace(0, 10, 200).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(x_clean, y_clean, color="#1f77b4", label="clean (n=50)", alpha=0.7)
ax.scatter(x_out,   y_out,   color="red",     label="outliers (n=5)", s=80, marker="x")
for name, mdl in fits.items():
    ax.plot(grid, mdl.predict(grid), label=f"{name}: slope = {mdl.coef_[0]:.2f}")
ax.axhline(y=2 + 1.3*5, color="grey", linestyle=":", alpha=0.4, label="(true mean at x=5)")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("OLS bends toward outliers; Huber and Theil-Sen do not")
plt.tight_layout()
plt.show()
```

### §D1. A fourth contender — RANSAC

`HuberRegressor` and `TheilSenRegressor` weight every observation, just less harshly than OLS for the outliers.  
**RANSAC** (RANdom SAmple Consensus) takes a different approach: it explicitly splits the data into **inliers** and **outliers** by repeatedly fitting on random small subsets and seeing which model maximises the number of agreeing inliers.  
It is the standard tool in computer vision (where outliers are common) and works well when up to ~50% of the data is contaminated.

```python
fits["RANSAC"] = RANSACRegressor(random_state=RANDOM_STATE).fit(x, y)

# Number all four slopes side by side.
true_slope = 1.3
rows = []
for name, mdl in fits.items():
    # RANSAC's estimator is .estimator_; the others expose .coef_ directly.
    coef = mdl.estimator_.coef_[0] if isinstance(mdl, RANSACRegressor) else mdl.coef_[0]
    rows.append({"model": name, "estimated_slope": coef, "abs_error": abs(coef - true_slope)})

pd.DataFrame(rows).round(3)
```

Read the `abs_error` column: how far each fit's slope sits from the true 1.3.

- **OLS** — typically the worst by a wide margin.
- **Huber / Theil-Sen / RANSAC** — cluster together close to truth.
- **RANSAC's distinguishing feature** — the explicit inlier/outlier split lives in `.inlier_mask_`, which lets you *see* which rows the fit excluded.

**One caveat for RANSAC.**
It needs enough data that random subsets are usable. On a 55-row dataset like this one, the inlier mask is noisy; in production you want hundreds or thousands of rows.

### §D2. A practical recipe for outlier-heavy data

1. **Plot the data**. If you can see outliers, you do not need a statistical test.
2. **Run plain OLS once**. Note R² and the residual plot. Heavy-tailed residuals → robust regression is justified.
3. **Try Huber first**. Fastest, OLS-like, single tuning parameter (`epsilon`, default 1.35 — generally fine).
4. **Try Theil-Sen on small data** (< few thousand rows) for a non-parametric second opinion.
5. **Reach for RANSAC** when contamination is high (> 20%) and you have enough data (n > few hundred).
6. **Document the choice**. A reviewer should be able to read *"we used Huber because the QQ plot had a heavy right tail"* without searching the appendix.

Plain OLS's slope is dragged *upward* by the five outliers; Huber and Theil-Sen recover slopes near the true 1.3.

The cost: robust regressors are slower (Theil-Sen especially) and have fewer downstream tools than plain OLS.

**When to reach for these.**

- **Huber** — when you want OLS-like speed and a single tuning knob (`epsilon`); good default for prediction tasks where ~5–15% of rows might be outliers.
- **Theil-Sen** — when you want a non-parametric guarantee and your dataset is small (n < a few thousand).
- **OLS + a separate outlier-detection step** (e.g., `IsolationForest` upstream, then fit OLS on clean data) — when you can defensibly throw outliers away.

## §D3. Bayesian linear regression — coefficients with uncertainty *built in*

- **OLS** gives a single point estimate per coefficient.
- **Frequentist standard errors** give an interval around that point estimate — *assuming the OLS residual assumptions hold*.
- **Bayesian linear regression** flips the framing: it returns a full **posterior distribution** over every coefficient, encoding everything the data tells you about uncertainty.

`sklearn.linear_model.BayesianRidge` implements one of the simpler Bayesian regressions:

- a Ridge-like prior on the coefficients (zero-mean Gaussian),
- a prior on the noise variance,
- both fit by type-II maximum likelihood.

It is a drop-in replacement for `Ridge` that *also* returns coefficient uncertainties via `return_std=True` at predict time.

```python
# Apply BayesianRidge to the same California-housing setup from §C.
br = BayesianRidge(max_iter=300).fit(X_tr_s, y_tr)
y_pred_mean, y_pred_std = br.predict(X_te_s, return_std=True)

# Compare the point prediction to the +/- 1 SD band on the first 20 test rows.
demo = pd.DataFrame({
    "y_true":          y_te.iloc[:20].values,
    "y_pred":          y_pred_mean[:20].round(3),
    "predicted_std":   y_pred_std[:20].round(3),
}).round(3)
print(f"BayesianRidge test R^2: {br.score(X_te_s, y_te):.3f}")
print(f"Mean predictive std across all test rows: {y_pred_std.mean():.3f}")
demo.head(8)
```

**Reading the output.**
`predicted_std` is the model's *self-reported* uncertainty about each prediction.
Wider stds mean *"I am not confident about this row"* — typically rows with feature combinations the training set had little of.

This is the killer feature: a single Bayesian fit gives you a per-row uncertainty estimate that OLS / Ridge / Lasso need a bootstrap to produce.

**When to reach for Bayesian regression.**

- You need a calibrated per-prediction uncertainty (decisions depend on confidence, not just the point estimate).
- You have prior information about coefficients you would like to encode (`BayesianRidge` uses an uninformative prior; PyMC / NumPyro let you place real priors).
- The dataset is small and over-fitting risk is high — the prior acts as built-in regularization.

`BayesianRidge` is the gentle on-ramp; production Bayesian workflows use **PyMC** or **NumPyro** to encode richer priors and inspect posteriors via MCMC.

## §D4. Quantile regression — when "the average" hides the action

OLS estimates the **conditional mean** of `y` given `X`.
For many business questions — pricing, risk, capacity planning — the *tails* of the distribution matter more than the mean.

> *"What is the 10th percentile of next-quarter sales for stores like this one?"*

is a question OLS cannot answer directly.

**Quantile regression** estimates the conditional q-th percentile:

- q = 0.5 → the median,
- q = 0.9 → the 90th-percentile.

It is the right tool whenever the residual distribution is skewed or you care about *which side* of the spread you are on.

```python
# Generate a heteroscedastic dataset where the spread of y grows with x.
np.random.seed(RANDOM_STATE)
x_q = np.linspace(0, 10, 200)
y_q = 3 + 1.2 * x_q + np.random.normal(0, 0.4 * x_q + 0.1, size=x_q.shape)

X_q = x_q.reshape(-1, 1)

# Conditional mean (plain OLS) and three conditional quantiles (10th, 50th, 90th).
ols_q = LinearRegression().fit(X_q, y_q)
# QuantileRegressor with alpha=0 means no L1 regularisation -- pure quantile regression.
q_results = {q: QuantileRegressor(quantile=q, alpha=0, solver="highs").fit(X_q, y_q)
             for q in (0.1, 0.5, 0.9)}

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(x_q, y_q, alpha=0.35, color="lightgrey", label="data")
ax.plot(x_q, ols_q.predict(X_q), color="black", linewidth=2.0, label="OLS (mean)")
for q, model in q_results.items():
    ax.plot(x_q, model.predict(X_q), label=f"Quantile q={q}", linewidth=1.5)
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Quantile regression captures the spread of y; OLS captures only the mean")
plt.tight_layout()
plt.show()
```

The OLS line goes through the *centre* of the data — but the spread of the data fans out as `x` grows.  
The 10th- and 90th-quantile lines fan out with it, giving you a usable predictive band.  
The 50th-quantile (median) line is close to OLS at low `x` and slightly below it at high `x` — they only coincide when the residuals are symmetric, which heteroscedastic data is not.

**Three production-ready uses of quantile regression.**

1. **Inventory and capacity planning.** Plan for the 90th-percentile of demand, not the mean — running out is more expensive than over-stocking.
2. **Risk and pricing.** Insurance, lending, and option pricing all care about tail behaviour, not the mean.
3. **SLA and worst-case latency.** When the question is *"what is the p99 response time?"*, quantile regression is the direct answer; OLS on the same data gives you the mean, which can be 10× off.

## §D5. The seven-fit summary

A compact reference: which regressor is the right reflex for which symptom?

| Symptom | First reflex | Where it lives |
| --- | --- | --- |
| Many features, want stability | Ridge | `lec_11e §C` |
| Many features, want feature selection | Lasso | `lec_11e §C` |
| Many *correlated* features, want both | ElasticNet | this notebook §C |
| Feature collinearity, inferential setting | Inspect VIF; consider dropping or combining | this notebook §B |
| Residuals have outliers | Huber → Theil-Sen → RANSAC | this notebook §D |
| Need per-row prediction uncertainty | BayesianRidge → PyMC | this notebook §D3 |
| Care about tails (p10 / p90 / median) | Quantile regression | this notebook §D4 |

Treat this table as the **first sentence** of a problem-shaping conversation, not as a decision rule.
The real choice depends on the size of the dataset, the cost of a wrong prediction, and what part of the residual distribution matters for the decision you are about to make.

## §E. Where to go next

A pointer table for students continuing past this lecture's mandatory material.
None of these is required for the next lecture, but each is the natural extension if a specific scenario hits you on the job.

| Scenario | What to study | Where |
| --- | --- | --- |
| Pick `alpha` / `l1_ratio` / polynomial degree robustly | Cross-validation; `GridSearchCV`; `ElasticNetCV` | **Lecture 13** of this course |
| Quantify coefficient uncertainty with priors | Bayesian linear regression | PyMC, NumPyro |
| Non-linear effects without a fixed polynomial degree | Generalised Additive Models (GAMs); splines | `pyGAM`, `statsmodels.gam` |
| The target is a probability, count, or rate | Generalised Linear Models (GLMs); logistic / Poisson regression | **Lecture 12** of this course, `statsmodels.glm` |
| Skewed target with heteroscedastic residuals | Quantile regression; target log-transformation | `sklearn.linear_model.QuantileRegressor` (used in §D4 above), or `statsmodels.regression.quantile_regression` |
| Predicting time-correlated outcomes | Time-series regression; ARIMA; state-space models | `statsmodels.tsa` |

The **single biggest leap** in capability from this lecture is **Lecture 13** — `Pipeline` + `GridSearchCV` + cross-validation replace hand-rolled three-way splits and α grids.

Read Lecture 13 next if your day-job involves any of:

- predicting from real data,
- defending your hyperparameter choices in code review,
- scaling beyond toy datasets.

## §F. End-to-end case study — California housing with a twist

This closing section walks the full advanced-regression workflow from §B through §D on a single dataset. The goal: show how the diagnostics drive the model choice in practice, rather than picking a regressor and hoping.

**Scenario.** We are predicting California housing prices.  
To make the exercise realistic, we inject a small fraction of high-leverage outliers (5% of rows with implausibly high prices — think bad data entry or a mis-recorded zip code).  
A real-world pipeline would not see clean data; it would see something like this.

```python
data = fetch_california_housing(as_frame=True)
X_clean = data.data
y_clean = data.target.copy()

# Contaminate 5% of rows with outliers.
n = len(y_clean)
n_out = int(0.05 * n)
out_idx = rng.choice(n, size=n_out, replace=False)
y_contam = y_clean.copy()
y_contam.iloc[out_idx] = y_contam.iloc[out_idx] * 4.0  # 4x inflation on these rows

print(f"Rows: {n} ({n_out} outlier-contaminated, target inflated 4x)")
print(f"Original y range:   [{y_clean.min():.2f}, {y_clean.max():.2f}]")
print(f"Contaminated range: [{y_contam.min():.2f}, {y_contam.max():.2f}]")
```

### §F1. Step 1 — Diagnose collinearity (VIF)

Before picking a model, check whether the 8 features have collinearity strong enough to make plain OLS coefficients unstable. If yes, lean toward regularization; if no, plain OLS is fine.

```python
vif_features = vif_manual(X_clean)
print("VIF on the 8 raw features:")
print(vif_features.round(2))
print()
high_vif = vif_features[vif_features > 5].index.tolist()
print(f"Features with VIF > 5: {high_vif if high_vif else 'none'}")
```

California housing typically shows moderate-to-high VIFs on `AveRooms`, `AveBedrms`, `HouseAge`, and `Population` — they correlate with each other. Plain OLS coefficients on the raw features would be unstable; this points us toward Ridge or ElasticNet rather than unregularised OLS.

### §F2. Step 2 — Train/test split + scaling

```python
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_clean, y_contam, test_size=0.20, random_state=RANDOM_STATE,
)
sc_c = StandardScaler().fit(X_tr_c)
X_tr_c_s = sc_c.transform(X_tr_c)
X_te_c_s = sc_c.transform(X_te_c)
```

### §F3. Step 3 — Try three models, score them, pick on evidence

We try plain OLS, Ridge (handles collinearity), and Huber (handles outliers). Each is fit and scored on the same held-out test set; the table tells us which one we should pick for this data.

```python
candidates = {
    "Plain OLS":         LinearRegression(),
    "Ridge(alpha=1.0)":  Ridge(alpha=1.0),
    "Huber(epsilon=1.35)": HuberRegressor(),
}

table = []
for name, mdl in candidates.items():
    mdl.fit(X_tr_c_s, y_tr_c)
    train_r2 = mdl.score(X_tr_c_s, y_tr_c)
    test_r2 = mdl.score(X_te_c_s, y_te_c)
    table.append({
        "model":        name,
        "train_R2":     train_r2,
        "test_R2":      test_r2,
        "train_test_gap": train_r2 - test_r2,
    })

pd.DataFrame(table).round(3)
```

### §F4. Step 4 — Interpret the table and pick

The contamination pattern of this dataset typically produces:

- **Plain OLS** — train R² is poor and test R² is worse; the outliers warp every coefficient.
- **Ridge** — slightly better; it shrinks coefficients but does not down-weight outlier rows.
- **Huber** — best test R² in this setup; it explicitly down-weights the contaminated rows.

If you cared about the *clean* relationship → Huber is the right call here.
If you cared about *predicting future contaminated data* (very unusual — usually contamination is what you want to detect and clean) → OLS or Ridge would actually be the honest choice.

**The takeaway from this case study.**
Advanced regression in practice is not *"pick the fanciest model"*; it is *"diagnose first, then pick the model that addresses the symptom"*.

- **VIF** tells you about feature redundancy.
- **Residual plots** tell you about outliers and non-normality.
- **The seven-fit summary table in §D5** then becomes a lookup: symptom on the left, model on the right.

### §F5. What to take away from this notebook

- **VIF (§B)** — the principled way to talk about collinearity; the bootstrap from `lec_11d` is the visual counterpart.
- **ElasticNet (§C)** — the safer choice over pure Lasso when correlated features must not be dropped arbitrarily.
- **Huber / Theil-Sen / RANSAC (§D1)** — the standard robust regressors; reach for them whenever a residuals plot shows heavy tails or visible outliers.
- **BayesianRidge (§D3)** — per-row uncertainty from a single fit; useful when downstream decisions depend on prediction confidence.
- **Quantile regression (§D4)** — the right tool for tail-focused questions: capacity planning, risk, SLA targets.
- **All of these become standard practice in Lecture 13** — `Pipeline` + `GridSearchCV` + cross-validation replace the hand-rolled three-way split and α sweep used through this lecture.

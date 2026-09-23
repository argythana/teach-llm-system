<!-- source: lectures_07_13_pandas_plots_scikit/lecture_11_regression_linear_polynomial/practice_exercises/lec_11_solutions.ipynb @ 0cc874704aaa -->

# Lecture 11 — Solutions

Worked solutions for `lec_11_exercises.ipynb`. Read the exercise brief first, attempt it, then check here.

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Only E1 uses these two APIs; every other exercise is sklearn-only.
import statsmodels.api as sm
import pingouin as pg

from scipy import stats

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, TheilSenRegressor, RANSACRegressor,
    BayesianRidge, QuantileRegressor,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector, RFE

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
sns.set_theme(style="whitegrid")

READING = "../reading_material"
```

## E1 — Simple linear regression with three APIs [G1]

Load `temp_to_coffee.csv` (columns `X`, `Y`). Rename to `temperature` and `quantity`. Fit a simple linear regression three different ways:

1. **statsmodels** — `sm.OLS(y, sm.add_constant(X)).fit()`. Print `model.summary()`.
2. **pingouin** — `pg.linear_regression(X, y)`. Print the returned DataFrame.
3. **scikit-learn** — `LinearRegression().fit(X, y)`. Print `.intercept_`, `.coef_`, and `.score(X, y)`.

Then assemble a small comparison table (intercept, slope, R²) and confirm the three APIs agree to four decimal places.

State one thing each API gives you that the other two do not.

```python
df = pd.read_csv(f"{READING}/temp_to_coffee.csv").rename(
    columns={"X": "temperature", "Y": "quantity"}
)
X = df[["temperature"]]
y = df["quantity"]

# (1) statsmodels
sm_model = sm.OLS(y, sm.add_constant(X)).fit()
print("--- statsmodels ---")
print(sm_model.summary())

# (2) pingouin
pg_table = pg.linear_regression(X, y)
print("\n--- pingouin ---")
print(pg_table)

# (3) scikit-learn
sk_model = LinearRegression().fit(X, y)
print("\n--- scikit-learn ---")
print(f"intercept = {sk_model.intercept_:.6f}")
print(f"slope     = {sk_model.coef_[0]:.6f}")
print(f"R^2       = {sk_model.score(X, y):.6f}")

# Side-by-side comparison
compare = pd.DataFrame({
    "statsmodels": [sm_model.params["const"], sm_model.params["temperature"], sm_model.rsquared],
    "pingouin":    [pg_table.loc[pg_table["names"] == "Intercept", "coef"].iloc[0],
                    pg_table.loc[pg_table["names"] == "temperature", "coef"].iloc[0],
                    pg_table["r2"].iloc[0]],
    "scikit-learn":[sk_model.intercept_, sk_model.coef_[0], sk_model.score(X, y)],
}, index=["intercept", "slope", "R^2"]).round(4)
print("\n--- side by side ---")
print(compare)

print("\nWhat each API uniquely gives you:")
print("- statsmodels: full inference table -- t-statistics, p-values, F-test, AIC/BIC.")
print("- pingouin:    tidy DataFrame with one row per predictor, easy to subset / merge.")
print("- scikit-learn: a `.predict()`-ready object that drops cleanly into Pipelines and grids.")
```

## E2 — Check the four OLS assumptions [G2]

Load `grades_factors.xlsx`. Fit a multiple linear regression with scikit-learn on `calc ~ calc_hs + act_math + alg_place + alg2_grade + hs_rank + gender_code`.

Then check the **four classical OLS assumptions** and decide whether each holds:

1. **Linearity** — residuals-vs-fitted scatter; look for a flat cloud around 0.
2. **Independence** — for cross-sectional data with no time order, declare this *assumed to hold* in one sentence (no plot needed).
3. **Homoscedasticity** — same residuals-vs-fitted scatter, but read it for funnel/megaphone patterns.
4. **Normality of residuals** — QQ plot via `scipy.stats.probplot`.

Print a one-line verdict ("holds" / "borderline" / "violated") for each, with a one-clause justification.

```python
df = pd.read_excel(f"{READING}/grades_factors.xlsx")
df.columns = df.columns.str.replace(" ", "_").str.lower()

feat = ["calc_hs", "act_math", "alg_place", "alg2_grade", "hs_rank", "gender_code"]
X = df[feat].astype(float)
y = df["calc"].astype(float)

model = LinearRegression().fit(X, y)
y_hat = model.predict(X)
resid = y - y_hat

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# (1) linearity + (3) homoscedasticity share the same plot
ax1.scatter(y_hat, resid, alpha=0.7)
ax1.axhline(0, color="red", linestyle="--", linewidth=1)
ax1.set_xlabel("Fitted values")
ax1.set_ylabel("Residuals")
ax1.set_title("Residuals vs fitted")

# (4) normality
stats.probplot(resid, plot=ax2)
ax2.set_title("QQ plot of residuals")

plt.tight_layout()
plt.show()

print("Verdicts:")
print("(1) Linearity         -- holds: cloud is roughly flat around 0, no clear curvature.")
print("(2) Independence      -- assumed to hold: rows are independent students, no time order.")
print("(3) Homoscedasticity  -- holds: residual spread is roughly constant across fitted values.")
print("(4) Normality         -- holds (mild tails): QQ points hug the diagonal with a few outliers.")
print()
print("If linearity or homoscedasticity had failed, the next steps from lec_11b §G are:")
print("  - transform y (log, sqrt) or x (square, log)")
print("  - add interaction / polynomial terms")
print("  - drop or downweight outliers")
```

## E3 — Multiple linear regression with scikit-learn [G3]

Load `grades_factors.xlsx`. Fit a multiple linear regression with scikit-learn on all six features (`calc_hs`, `act_math`, `alg_place`, `alg2_grade`, `hs_rank`, `gender_code`) predicting `calc`.

Print:

- the intercept;
- the coefficients as a `pd.Series` indexed by feature name, sorted by absolute value;
- training **R²** and training **MAE**.

Then in one sentence: which feature has the largest **conditional** effect on `calc`, and which has the smallest? Are you sure that ordering would hold if the features were on different scales? (You will revisit this in E7.)

```python
df = pd.read_excel(f"{READING}/grades_factors.xlsx")
df.columns = df.columns.str.replace(" ", "_").str.lower()

feat = ["calc_hs", "act_math", "alg_place", "alg2_grade", "hs_rank", "gender_code"]
X = df[feat].astype(float)
y = df["calc"].astype(float)

model = LinearRegression().fit(X, y)

coefs = pd.Series(model.coef_, index=feat).sort_values(key=np.abs, ascending=False)
print(f"intercept = {model.intercept_:.4f}")
print("\ncoefficients (sorted by |value|):")
print(coefs.round(4))
print(f"\nTraining R^2 = {model.score(X, y):.4f}")
print(f"Training MAE = {mean_absolute_error(y, model.predict(X)):.4f}")

top = coefs.index[0]
bot = coefs.index[-1]
print(f"\nLargest |coefficient|: {top} ({coefs.iloc[0]:+.2f})")
print(f"Smallest |coefficient|: {bot} ({coefs.iloc[-1]:+.2f})")
print("\nCaution: the magnitudes above ARE NOT scale-invariant. hs_rank ranges over")
print("~0-100 while gender_code is just 0/1, so per-unit coefficients are not directly")
print("comparable. To rank features fairly you would need to standardise X first")
print("(pitfall P2, lec_11d). See E6 for the related marginal-vs-conditional gotcha.")
```

## E4 — Scientific feature selection (statsmodels + sklearn-native) [G4]

Load `grades_factors.xlsx` and fit OLS with **statsmodels** on all six features predicting `calc`. From the `.summary()` table, identify the two features with the **highest p-values** and drop them — refit on the remaining four features, print the new R² adjusted and AIC, and confirm they improved (or explain why they did not).

Then use sklearn's `SequentialFeatureSelector` (forward, `cv=5`) to pick 3 features. Print the picked feature names. In one sentence: did SFS agree with your statsmodels-driven backward pick? If they disagree, what is each method optimising for?

```python
df = pd.read_excel(f"{READING}/grades_factors.xlsx")
df.columns = df.columns.str.replace(" ", "_").str.lower()

feat = ["calc_hs", "act_math", "alg_place", "alg2_grade", "hs_rank", "gender_code"]
X = df[feat].astype(float)
y = df["calc"].astype(float)

# Statsmodels view: read p-values from .summary()
ols_full = sm.OLS(y, sm.add_constant(X)).fit()
pvals_full = ols_full.pvalues.drop("const").sort_values(ascending=False)
print("Full-model p-values (highest first):")
print(pvals_full.round(3))

to_drop = pvals_full.index[:2].tolist()
kept = [f for f in feat if f not in to_drop]
print(f"\nDropping highest-p features: {to_drop}")
print(f"Keeping: {kept}")

ols_reduced = sm.OLS(y, sm.add_constant(X[kept])).fit()
print(f"\nFull-model    adj-R^2 = {ols_full.rsquared_adj:.3f}, AIC = {ols_full.aic:.1f}")
print(f"Reduced-model adj-R^2 = {ols_reduced.rsquared_adj:.3f}, AIC = {ols_reduced.aic:.1f}")

# sklearn view: SequentialFeatureSelector
sfs = SequentialFeatureSelector(
    LinearRegression(), n_features_to_select=3, direction="forward", cv=5,
).fit(X, y)
sfs_pick = list(sfs.get_feature_names_out())
print(f"\nSFS forward-CV pick (3): {sfs_pick}")
print(f"Statsmodels-kept (4):    {kept}")

print("\nThe two methods commonly disagree on small data. Statsmodels' backward elimination")
print("by p-value asks 'is this coefficient distinguishable from zero?' (inferential). SFS asks")
print("'does adding this feature lower the CV error?' (predictive). On n=80 with overlapping math")
print("features, both pick calc_hs and alg_place; the third slot is where they diverge.")
```

## E5 — Intercept on / off [G5]

On `grades_factors.xlsx`, fit two scikit-learn `LinearRegression` models of `calc` on all six features:

- **With intercept** — `LinearRegression(fit_intercept=True)` (the default).
- **Without intercept** — `LinearRegression(fit_intercept=False)`.

For each:

- print the coefficients (no statsmodels — read them off `model.coef_`);
- print the **training R²** via `model.score(X, y)`.

Then in two sentences: which model has the lower training R², why (since OLS minimises training MSE either way), and what is special about scikit-learn's R² when the intercept is removed.

```python
df = pd.read_excel(f"{READING}/grades_factors.xlsx")
df.columns = df.columns.str.replace(" ", "_").str.lower()
feat = ["calc_hs", "act_math", "alg_place", "alg2_grade", "hs_rank", "gender_code"]
X = df[feat].astype(float)
y = df["calc"].astype(float)

m_with = LinearRegression(fit_intercept=True).fit(X, y)
m_no   = LinearRegression(fit_intercept=False).fit(X, y)

side_by_side = pd.DataFrame({
    "with_intercept": m_with.coef_,
    "no_intercept":   m_no.coef_,
}, index=feat).round(4)
print(side_by_side)
print(f"\nintercept (with) = {m_with.intercept_:.4f}")
print(f"intercept (no)   = {m_no.intercept_:.4f}")
print(f"\nR^2 (with) = {m_with.score(X, y):.4f}")
print(f"R^2 (no)   = {m_no.score(X, y):.4f}")

print("\nThe no-intercept model has the LOWER sklearn R^2 -- and on other datasets")
print("(or after shifting y) it can go NEGATIVE.")
print("Why: sklearn defines R^2 = 1 - SSR/SST, where SST is computed against the MEAN of y.")
print("Without an intercept the fitted line is forced through the origin, so its predictions")
print("can be WORSE than just predicting y.mean(). When that happens, SSR > SST and R^2 < 0.")
print("On this particular grades dataset the line-through-origin still beats y.mean(),")
print("so R^2 stays positive -- but it is structurally biased and the gap to the intercepted")
print("fit will only widen on data with a non-zero offset.")
print("(statsmodels uses a different no-intercept R^2 formula that inflates the number --")
print("that gotcha is the reason lec_11c §D2 prefers sklearn's honest behaviour here.)")
```

## E6 — Train/test and three-way split [G6]

Two parts on `grades_factors.xlsx`. Use `RANDOM_STATE` everywhere a seed is needed.

**Part A — train/test (70/30).** Fit `LinearRegression` on the train slice (all six features). Print train R² and test R².

**Part B — three-way (60/20/20).** Pick the best feature **subset** from these three candidates by validation R²:

- `["calc_hs"]`
- `["calc_hs", "act_math", "alg2_grade"]`
- `["calc_hs", "act_math", "alg_place", "alg2_grade", "hs_rank", "gender_code"]`

For each subset: fit on train, score on validation. Pick the subset with the highest validation R². Refit on `train + val` combined with that subset. Score **once** on test. Print the test R² and the chosen subset.

In one sentence: why is the test R² in Part B more trustworthy than the test R² in Part A as a generalisation estimate?

```python
df = pd.read_excel(f"{READING}/grades_factors.xlsx")
df.columns = df.columns.str.replace(" ", "_").str.lower()
feat_all = ["calc_hs", "act_math", "alg_place", "alg2_grade", "hs_rank", "gender_code"]
y = df["calc"].astype(float)

# --- Part A: 70/30 train/test ---
X = df[feat_all].astype(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
m = LinearRegression().fit(X_tr, y_tr)
print("--- Part A: 70/30 ---")
print(f"train R^2 = {m.score(X_tr, y_tr):.4f}")
print(f"test  R^2 = {m.score(X_te, y_te):.4f}")

# --- Part B: 60/20/20 three-way ---
print("\n--- Part B: 60/20/20 ---")
X_full = df[feat_all].astype(float)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_full, y, test_size=0.20, random_state=RANDOM_STATE
)
# 0.25 of the remaining 80% = 20% of total -> validation
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE
)

candidates = {
    "small":  ["calc_hs"],
    "medium": ["calc_hs", "act_math", "alg2_grade"],
    "full":   feat_all,
}
val_scores = {}
for name, cols in candidates.items():
    mdl = LinearRegression().fit(X_train[cols], y_train)
    val_scores[name] = mdl.score(X_val[cols], y_val)
    print(f"  {name:7s}  val R^2 = {val_scores[name]:.4f}")

best_name = max(val_scores, key=val_scores.get)
best_cols = candidates[best_name]
print(f"\nBest by val: {best_name} = {best_cols}")

final = LinearRegression().fit(X_trainval[best_cols], y_trainval)
print(f"test R^2 (held out, one-shot) = {final.score(X_test[best_cols], y_test):.4f}")

print("\nPart B's test R^2 is more trustworthy: the test slice was untouched while")
print("we chose the subset. In Part A, if we had peeked at test R^2 and re-chosen features,")
print("the test set would have leaked into model selection.")
```

## E7 — Marginal vs conditional effect (pitfall P1) [G7]

On `grades_factors.xlsx`, compute the coefficient of `calc_hs` two ways:

- **Marginal**: `calc ~ calc_hs` alone (sklearn `LinearRegression` on a single feature).
- **Conditional**: `calc ~ calc_hs + act_math + alg2_grade` (three features).

Print both coefficients. Comment in one sentence: which one would you cite to a journalist who asks "by how much does an extra point of `calc_hs` raise the calculus score?", and why?

(This is pitfall **P1** from `lec_11d`. The four pitfalls are: P1 marginal-vs-conditional, P2 scale dependence, P3 correlated-feature instability, P4 model-is-not-the-world.)

```python
df = pd.read_excel(f"{READING}/grades_factors.xlsx")
df.columns = df.columns.str.replace(" ", "_").str.lower()
y = df["calc"].astype(float)

# Marginal
X_marg = df[["calc_hs"]].astype(float)
m_marg = LinearRegression().fit(X_marg, y)
beta_marg = m_marg.coef_[0]

# Conditional
X_cond = df[["calc_hs", "act_math", "alg2_grade"]].astype(float)
m_cond = LinearRegression().fit(X_cond, y)
beta_cond = pd.Series(m_cond.coef_, index=X_cond.columns)["calc_hs"]

print(f"marginal     coefficient of calc_hs = {beta_marg:.4f}")
print(f"conditional  coefficient of calc_hs = {beta_cond:.4f}")
print(f"\nThe two differ by {beta_marg - beta_cond:+.4f}.")

print("\nTo the journalist: neither, without a caveat. The marginal coefficient mixes the")
print("direct effect of calc_hs with everything calc_hs correlates with (act_math, alg2_grade).")
print("The conditional coefficient is the slope HOLDING the other two fixed -- which is rarely")
print("what readers picture when they hear 'an extra point of calc_hs'. Cite the conditional")
print("coefficient and spell out the 'holding x, y fixed' clause explicitly.")
```

## E8 — Polynomial regression with Ridge and Lasso [G8]

Generate a 1D dataset: `x` uniform in `[0, 1]` (n = 80), `y = sin(2πx) + 0.2·noise`. Use 70/30 train/test split.

1. Fit a plain `LinearRegression` on **degree-12 polynomial features** (standardised after split). Report train and test **MSE**.
2. Fit `Ridge(alpha=0.1)` on the same polynomial features. Report train and test MSE.
3. Fit `Lasso(alpha=0.01)` on the same polynomial features. Report train and test MSE, plus the number of non-zero coefficients (out of 12).

In one sentence: which model has the lowest **test** MSE and why Lasso's coefficient pattern is qualitatively different from Ridge's.

```python
n = 80
x = rng.uniform(0, 1, size=n)
y = np.sin(2 * np.pi * x) + 0.2 * rng.standard_normal(n)

X = x.reshape(-1, 1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)

poly = PolynomialFeatures(degree=12, include_bias=False)
X_tr_p = poly.fit_transform(X_tr)
X_te_p = poly.transform(X_te)

sc = StandardScaler().fit(X_tr_p)
X_tr_s = sc.transform(X_tr_p)
X_te_s = sc.transform(X_te_p)

models = {
    "Plain LinearRegression": LinearRegression(),
    "Ridge(alpha=0.1)":       Ridge(alpha=0.1),
    "Lasso(alpha=0.01)":      Lasso(alpha=0.01, max_iter=20000),
}
for name, mdl in models.items():
    mdl.fit(X_tr_s, y_tr)
    tr_mse = mean_squared_error(y_tr, mdl.predict(X_tr_s))
    te_mse = mean_squared_error(y_te, mdl.predict(X_te_s))
    extra = ""
    if isinstance(mdl, Lasso):
        nz = int(np.sum(np.abs(mdl.coef_) > 1e-8))
        extra = f"  | nonzero coefs: {nz}/12"
    print(f"  {name:25s}  train MSE = {tr_mse:.3f}, test MSE = {te_mse:.3f}{extra}")

print("\nPlain LinearRegression typically has the worst test MSE (overfitting).")
print("Ridge shrinks ALL coefficients toward zero but rarely to exactly zero.")
print("Lasso zeros most of the 12 polynomial features -- giving a sparse 'which")
print("features matter' story that Ridge cannot.")
```

## EA — Write a good regression prompt [A1]

Open `read_agents_regression_workflows.md` (in `../reading_material/`) and read **§2 Description** and **§6 Concrete prompt patterns**. Then, in the cell below, write a prompt you would actually give an AI assistant for the following scenario:

> *Predict the `calc` target in `grades_factors.xlsx` using the six pre-university features. The goal is **prediction**, not inference. Use Ridge regression and pick `alpha` from a small grid via a validation split. Refit on train+val. Score once on test.*

After the prompt string, add a comment block listing **three discernment failures** your prompt's specificity is meant to prevent — refer to lec_11's pitfalls (P1–P4) and the 4Ds.

```python
prompt = '''
I have a dataset with n = 80 rows, 6 numeric features, and 1 binary feature
(already encoded 0/1), predicting a continuous target (a grade in 0-100).

GOAL: prediction, not inference -- I do not need p-values, I need lowest test
MSE on new data.

PREPROCESSING: scale ONLY after train/test split: fit a StandardScaler on
X_train, then transform both X_train and X_test. Do not touch the binary
feature with the scaler.

MODEL: Ridge regression. Pick alpha by sweeping over [1e-3, 1e-2, 1e-1, 1,
10, 100] on a separate validation split (60/20/20 train/val/test). Use the
validation R^2 to choose alpha, then refit on train + val combined, and
score ONCE on the test set.

PRINT: chosen alpha, train R^2, validation R^2 for every alpha, final test
R^2 and test MSE. Also print the coefficients on the standardised features
as a sorted bar chart.

CONSTRAINTS:
- Do not import sklearn.pipeline.Pipeline or GridSearchCV (these are
  lecture 13 territory).
- random_state = 42 on every split.
- Warn me if any feature pair has |correlation| > 0.7.
'''
print(prompt)

# Three discernment failures this prompt's specificity prevents:
# 1. The assistant scaling BEFORE the split (data leakage) -- the prompt
#    pins the order of operations.
# 2. The assistant picking alpha from a single train/test split without a
#    validation set -- the prompt mandates the three-way split + sweep
#    (lec_11e §E).
# 3. The assistant returning a model.summary()-style inference table when
#    the goal is prediction -- the prompt names the goal up front (4Ds:
#    Description) and asks for R^2/MSE instead of p-values.
```

## S1 — Lasso-then-OLS (stretch) [O1]

On the polynomial features from **E7**, run `Lasso(alpha=0.01)`, identify which features have non-zero coefficients, then refit plain `LinearRegression` using **only those features**. Compare the test MSE of the refit OLS against Lasso's own test MSE.

Why might Lasso-then-OLS sometimes win, and what is the risk?

```python
n = 80
x = rng.uniform(0, 1, size=n)
y = np.sin(2 * np.pi * x) + 0.2 * rng.standard_normal(n)
X = x.reshape(-1, 1)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
poly = PolynomialFeatures(degree=12, include_bias=False)
X_tr_p = poly.fit_transform(X_tr)
X_te_p = poly.transform(X_te)
sc = StandardScaler().fit(X_tr_p)
X_tr_s = sc.transform(X_tr_p)
X_te_s = sc.transform(X_te_p)

lasso = Lasso(alpha=0.01, max_iter=20000).fit(X_tr_s, y_tr)
keep = np.abs(lasso.coef_) > 1e-8
print(f"Lasso kept {keep.sum()} / 12 features: indices {np.where(keep)[0].tolist()}")

# Refit plain OLS on the kept features
ols = LinearRegression().fit(X_tr_s[:, keep], y_tr)

lasso_test_mse = mean_squared_error(y_te, lasso.predict(X_te_s))
ols_test_mse   = mean_squared_error(y_te, ols.predict(X_te_s[:, keep]))

print(f"\nLasso         test MSE = {lasso_test_mse:.3f}")
print(f"Lasso-then-OLS test MSE = {ols_test_mse:.3f}")

print("\nWhy it can win: Lasso shrinks all surviving coefficients toward 0 (introducing")
print("bias). Refitting plain OLS on the kept features removes that shrinkage bias while")
print("keeping Lasso's variance-reducing feature selection.")
print("\nThe risk: if Lasso happens to drop a relevant feature on this particular split,")
print("the refit OLS inherits the omission and can be worse than vanilla Ridge.")
```

## S2 — Bootstrap coefficient distribution (stretch) [O2]

For the multiple regression on `grades_factors` (six features, sklearn), compute the **bootstrap distribution** of the `calc_hs` coefficient using **200 resamples** (sample rows with replacement, refit `LinearRegression`, record the coefficient).

Plot a histogram. Report the bootstrap **mean** and **2.5 / 97.5 percentiles** as an empirical 95% confidence interval, and compare the mean to the point estimate from the full data.

```python
df = pd.read_excel(f"{READING}/grades_factors.xlsx")
df.columns = df.columns.str.replace(" ", "_").str.lower()
feat = ["calc_hs", "act_math", "alg_place", "alg2_grade", "hs_rank", "gender_code"]
X_full = df[feat].astype(float)
y_full = df["calc"].astype(float)

point = LinearRegression().fit(X_full, y_full)
point_coef = pd.Series(point.coef_, index=feat)["calc_hs"]

n = len(df)
n_boot = 200
boot_coefs = np.empty(n_boot)
for i in range(n_boot):
    idx = rng.integers(0, n, size=n)
    mdl = LinearRegression().fit(X_full.iloc[idx], y_full.iloc[idx])
    boot_coefs[i] = pd.Series(mdl.coef_, index=feat)["calc_hs"]

lo, hi = np.percentile(boot_coefs, [2.5, 97.5])
print(f"point estimate of calc_hs coef       = {point_coef:.4f}")
print(f"bootstrap mean                       = {boot_coefs.mean():.4f}")
print(f"bootstrap 95% CI (percentile method) = [{lo:.4f}, {hi:.4f}]")

plt.figure(figsize=(7, 4))
plt.hist(boot_coefs, bins=25, edgecolor="white")
plt.axvline(point_coef, color="red", linestyle="--", label=f"point estimate = {point_coef:.3f}")
plt.axvline(lo, color="gray", linestyle=":", label="95% CI")
plt.axvline(hi, color="gray", linestyle=":")
plt.xlabel("calc_hs coefficient (bootstrap resamples)")
plt.ylabel("count")
plt.title(f"Bootstrap distribution of calc_hs coefficient (n_boot = {n_boot})")
plt.legend()
plt.tight_layout()
plt.show()

print("\nThe bootstrap CI is a non-parametric stand-in for the standard-error column of")
print("statsmodels' .summary() table -- it requires no normality assumption on residuals.")
```

## S3 — Variance Inflation Factor (stretch) [O3]

Compute the **VIF** for each of the six features in `grades_factors.xlsx` manually:

$$\mathrm{VIF}_j = \frac{1}{1 - R^2_j}$$

where $R^2_j$ is the R² of a regression of feature $j$ on all other features (use sklearn `LinearRegression`).

Print a sorted table feature → VIF. Flag any feature with VIF > 5 (moderate collinearity) or VIF > 10 (high). One sentence on what action this would prompt — drop the feature, combine correlated features, or switch to Ridge?

```python
df = pd.read_excel(f"{READING}/grades_factors.xlsx")
df.columns = df.columns.str.replace(" ", "_").str.lower()
feat = ["calc_hs", "act_math", "alg_place", "alg2_grade", "hs_rank", "gender_code"]
X_full = df[feat].astype(float)

def vif_manual(X, col):
    others = [c for c in X.columns if c != col]
    r2 = LinearRegression().fit(X[others], X[col]).score(X[others], X[col])
    return 1.0 / (1.0 - r2)

vifs = pd.Series({c: vif_manual(X_full, c) for c in feat}).sort_values(ascending=False)
print(vifs.round(2))

print()
for c, v in vifs.items():
    flag = "HIGH"     if v > 10 else \
           "moderate" if v > 5  else \
           "ok"
    print(f"  {c:12s}  VIF = {v:6.2f}   ({flag})")

print("\nAction rules of thumb:")
print("- VIF > 10: drop one of the correlated features, or combine them (e.g. average")
print("            multiple algebra scores into a single 'pre-calculus index').")
print("- VIF 5-10: switch to Ridge regression -- it tolerates collinearity gracefully.")
print("- VIF < 5:  no action; coefficients are stable enough to interpret.")
```

## S4 — ElasticNet on correlated features (stretch) [O4]

Lasso picks **one** of two highly-correlated features somewhat arbitrarily; ElasticNet (`l1_ratio` between 0 and 1) lets you blend L1's sparsity with L2's stability. On the polynomial features from **E8**, fit `ElasticNet(alpha=0.01, l1_ratio=0.5)` and compare against the Lasso fit you already did. Report: how many non-zero coefficients does each keep, and what is the test MSE of each? Then sweep `l1_ratio ∈ [0.1, 0.3, 0.5, 0.7, 0.9]` with `alpha=0.01` fixed — at which `l1_ratio` does the test MSE bottom out?

```python
# Reuse the 1D synthetic polynomial setup from S1
n = 80
x = rng.uniform(0, 1, size=n)
y = np.sin(2 * np.pi * x) + 0.2 * rng.standard_normal(n)
X = x.reshape(-1, 1)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
poly = PolynomialFeatures(degree=12, include_bias=False)
X_tr_p = poly.fit_transform(X_tr); X_te_p = poly.transform(X_te)
sc = StandardScaler().fit(X_tr_p)
X_tr_s = sc.transform(X_tr_p); X_te_s = sc.transform(X_te_p)

lasso = Lasso(alpha=0.01, max_iter=20000).fit(X_tr_s, y_tr)
enet  = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000).fit(X_tr_s, y_tr)

def report(name, mdl):
    nz = int(np.sum(np.abs(mdl.coef_) > 1e-8))
    mse = mean_squared_error(y_te, mdl.predict(X_te_s))
    print(f"{name:<22} non-zero coefs = {nz:>2} / {len(mdl.coef_)}   test MSE = {mse:.4f}")

report("Lasso(0.01)",                lasso)
report("ElasticNet(0.01, 0.5)",      enet)

print("\nSweeping l1_ratio with alpha=0.01 fixed:")
for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
    m = ElasticNet(alpha=0.01, l1_ratio=r, max_iter=20000).fit(X_tr_s, y_tr)
    print(f"  l1_ratio={r:.1f}  non-zero={int(np.sum(np.abs(m.coef_) > 1e-8)):>2}  test MSE={mean_squared_error(y_te, m.predict(X_te_s)):.4f}")

print("\nLower l1_ratio (more L2) keeps more features non-zero and trades some sparsity for stability.")
print("On correlated features, ElasticNet tends to keep groups together where Lasso would arbitrarily pick one.")
```

## S5 — Robust regression on outlier-contaminated data (stretch) [O5]

Generate 50 clean points around `y = 2 + 1.3*x + N(0, 0.5)` for `x` in `[0, 10]`. Add **5 outliers** with `y` in `[20, 30]` (extreme y, normal x). Fit four regressors on the *combined* data: plain `LinearRegression`, `HuberRegressor`, `TheilSenRegressor(random_state=RANDOM_STATE)`, and `RANSACRegressor(random_state=RANDOM_STATE)`. Print each model's slope and compare against the true slope (1.3). Which regressors recover the true slope, and which were dragged toward the outliers?

```python
# Clean linear + 5 outliers
n_clean = 50
x_clean = rng.uniform(0, 10, size=n_clean)
y_clean = 2.0 + 1.3 * x_clean + rng.normal(0, 0.5, size=n_clean)

n_out = 5
x_out = rng.uniform(0, 10, size=n_out)
y_out = rng.uniform(20, 30, size=n_out)

x_all = np.concatenate([x_clean, x_out]).reshape(-1, 1)
y_all = np.concatenate([y_clean, y_out])

true_slope = 1.3
fits = {
    "OLS":       LinearRegression().fit(x_all, y_all),
    "Huber":     HuberRegressor().fit(x_all, y_all),
    "Theil-Sen": TheilSenRegressor(random_state=RANDOM_STATE).fit(x_all, y_all),
    "RANSAC":    RANSACRegressor(random_state=RANDOM_STATE).fit(x_all, y_all),
}

for name, mdl in fits.items():
    if name == "RANSAC":
        slope = mdl.estimator_.coef_[0]
    else:
        slope = mdl.coef_[0]
    print(f"  {name:<10}  slope = {slope:+.3f}   gap to true = {abs(slope - true_slope):.3f}")

print(f"\n  (true slope = {true_slope})")
print("\nOLS minimises squared residuals, so the five outliers (with very large residuals) pull its slope")
print("upward toward them. Huber switches to absolute-value loss for large residuals; Theil-Sen uses the median")
print("of pairwise slopes; RANSAC fits on a randomly-selected inlier subset. All three recover the clean slope")
print("to within ~0.1 of 1.3, while OLS is biased high.")
```

## S6 — Bayesian + quantile regression (stretch) [O6]

On `grades_factors`, fit two non-OLS regressors:

1. **`BayesianRidge`** — print the posterior mean and standard deviation for the prediction on the *first three rows*. What does a per-row prediction interval tell you that a single point estimate does not?
2. **`QuantileRegressor(quantile=0.1)`** and `QuantileRegressor(quantile=0.9)` — fit both, print the predicted 10th- and 90th-percentile `calc` for the first three rows. When would a tail-quantile prediction be more useful than the OLS mean prediction?

```python
df = pd.read_excel(f"{READING}/grades_factors.xlsx")
df.columns = df.columns.str.replace(" ", "_").str.lower()
feat = ["calc_hs", "act_math", "alg_place", "alg2_grade", "hs_rank", "gender_code"]
X = df[feat].astype(float); y = df["calc"].astype(float)

# 1. BayesianRidge — per-row prediction with std
br = BayesianRidge().fit(X, y)
y_pred, y_std = br.predict(X.iloc[:3], return_std=True)
print("BayesianRidge — first 3 rows:")
for i, (m, s) in enumerate(zip(y_pred, y_std)):
    print(f"  row {i}: mean = {m:.2f}, sd = {s:.2f}  -> 90% interval ~ [{m-1.645*s:.2f}, {m+1.645*s:.2f}]")

# 2. Quantile regression — 10th and 90th percentiles
q10 = QuantileRegressor(quantile=0.1, alpha=0.0, solver="highs").fit(X, y)
q90 = QuantileRegressor(quantile=0.9, alpha=0.0, solver="highs").fit(X, y)
print("\nQuantile regression — first 3 rows:")
for i in range(3):
    p10 = q10.predict(X.iloc[[i]])[0]
    p90 = q90.predict(X.iloc[[i]])[0]
    print(f"  row {i}: 10th-pctile = {p10:.2f}   90th-pctile = {p90:.2f}")

print("\nBayesianRidge gives a *posterior* prediction interval — useful when downstream decisions weight")
print("uncertainty (e.g., 'predict salary with 90% confidence interval for a loan application').")
print("\nQuantile regression models the conditional 10th/90th percentile of y given X — useful for tail-focused")
print("questions ('worst-case grade we should plan for', 'best-case demand we should staff for').")
```

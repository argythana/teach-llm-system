<!-- source: lectures_07_13_pandas_plots_scikit/lecture_11_regression_linear_polynomial/practice_exercises/lec_11_exercises.ipynb @ 0cc874704aaa -->

# Lecture 11 — Practice Exercises

Seven required exercises (E1–E7) mapped 1-to-1 to learning goals G1–G7, one AI-fluency exercise (EA), and three stretch exercises (S1–S3) for the optional goals.

**Read** `../reading_material/goals_11.md` for the full goal definitions.

**Conventions**

- `RANDOM_STATE = 42` everywhere a seed is needed.
- Datasets live in `../reading_material/`.
- E1 uses **all three APIs** (statsmodels, pingouin, scikit-learn) — this is the only exercise where statsmodels / pingouin appear. Every exercise from E2 onwards is **scikit-learn only**.
- Solutions: `lec_11_solutions.ipynb`.

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
# Your code here.
# Hint:
#   df = pd.read_csv(f"{READING}/temp_to_coffee.csv").rename(columns={"X": "temperature", "Y": "quantity"})
#   X = df[["temperature"]]; y = df["quantity"]
#   sm.OLS(y, sm.add_constant(X)).fit().summary()
#   pg.linear_regression(X, y)
#   LinearRegression().fit(X, y)
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
# Your code here.
# Hint:
#   df = pd.read_excel(f"{READING}/grades_factors.xlsx")
#   df.columns = df.columns.str.replace(" ", "_").str.lower()
#   feat = ["calc_hs", "act_math", "alg_place", "alg2_grade", "hs_rank", "gender_code"]
#   model = LinearRegression().fit(df[feat], df["calc"])
#   residuals = df["calc"] - model.predict(df[feat])
#   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
#   ax1.scatter(model.predict(df[feat]), residuals); ax1.axhline(0)
#   stats.probplot(residuals, plot=ax2)
```

## E3 — Multiple linear regression with scikit-learn [G3]

Load `grades_factors.xlsx`. Fit a multiple linear regression with scikit-learn on all six features (`calc_hs`, `act_math`, `alg_place`, `alg2_grade`, `hs_rank`, `gender_code`) predicting `calc`.

Print:

- the intercept;
- the coefficients as a `pd.Series` indexed by feature name, sorted by absolute value;
- training **R²** and training **MAE**.

Then in one sentence: which feature has the largest **conditional** effect on `calc`, and which has the smallest? Are you sure that ordering would hold if the features were on different scales? (You will revisit this in E7.)

```python
# Your code here.
# Hint:
#   model = LinearRegression().fit(X, y)
#   pd.Series(model.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
#   mean_absolute_error(y, model.predict(X))
```

## E4 — Scientific feature selection (statsmodels + sklearn-native) [G4]

Load `grades_factors.xlsx` and fit OLS with **statsmodels** on all six features predicting `calc`. From the `.summary()` table, identify the two features with the **highest p-values** and drop them — refit on the remaining four features, print the new R² adjusted and AIC, and confirm they improved (or explain why they did not).

Then use sklearn's `SequentialFeatureSelector` (forward, `cv=5`) to pick 3 features. Print the picked feature names. In one sentence: did SFS agree with your statsmodels-driven backward pick? If they disagree, what is each method optimising for?

```python
# Your code here.
# Hint:
#   sm.OLS(y, sm.add_constant(X)).fit().summary()              # statsmodels view
#   SequentialFeatureSelector(LinearRegression(), n_features_to_select=3, cv=5)
#       .fit(X, y).get_feature_names_out()                     # sklearn view
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
# Your code here.
# Hint:
#   m_with = LinearRegression(fit_intercept=True).fit(X, y)
#   m_no   = LinearRegression(fit_intercept=False).fit(X, y)
#   pd.DataFrame({"with": m_with.coef_, "no": m_no.coef_}, index=X.columns)
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
# Your code here.
# Hint Part A: train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
# Hint Part B: split off 20% test first, then split the remaining 80% into
#              60/20 train/val (so test_size = 0.25 on the second split).
```

## E7 — Marginal vs conditional effect (pitfall P1) [G7]

On `grades_factors.xlsx`, compute the coefficient of `calc_hs` two ways:

- **Marginal**: `calc ~ calc_hs` alone (sklearn `LinearRegression` on a single feature).
- **Conditional**: `calc ~ calc_hs + act_math + alg2_grade` (three features).

Print both coefficients. Comment in one sentence: which one would you cite to a journalist who asks "by how much does an extra point of `calc_hs` raise the calculus score?", and why?

(This is pitfall **P1** from `lec_11d`. The four pitfalls are: P1 marginal-vs-conditional, P2 scale dependence, P3 correlated-feature instability, P4 model-is-not-the-world.)

```python
# Your code here.
# Hint:
#   m_marg = LinearRegression().fit(df[["calc_hs"]], df["calc"])
#   m_cond = LinearRegression().fit(df[["calc_hs", "act_math", "alg2_grade"]], df["calc"])
#   print marginal and conditional coefficients of calc_hs
```

## E8 — Polynomial regression with Ridge and Lasso [G8]

Generate a 1D dataset: `x` uniform in `[0, 1]` (n = 80), `y = sin(2πx) + 0.2·noise`. Use 70/30 train/test split.

1. Fit a plain `LinearRegression` on **degree-12 polynomial features** (standardised after split). Report train and test **MSE**.
2. Fit `Ridge(alpha=0.1)` on the same polynomial features. Report train and test MSE.
3. Fit `Lasso(alpha=0.01)` on the same polynomial features. Report train and test MSE, plus the number of non-zero coefficients (out of 12).

In one sentence: which model has the lowest **test** MSE and why Lasso's coefficient pattern is qualitatively different from Ridge's.

```python
# Your code here.
# Hint: rng.uniform; np.sin; PolynomialFeatures(degree=12, include_bias=False).fit_transform;
#       StandardScaler; LinearRegression, Ridge, Lasso
```

## EA — Write a good regression prompt [A1]

Open `read_agents_regression_workflows.md` (in `../reading_material/`) and read **§2 Description** and **§6 Concrete prompt patterns**. Then, in the cell below, write a prompt you would actually give an AI assistant for the following scenario:

> *Predict the `calc` target in `grades_factors.xlsx` using the six pre-university features. The goal is **prediction**, not inference. Use Ridge regression and pick `alpha` from a small grid via a validation split. Refit on train+val. Score once on test.*

After the prompt string, add a comment block listing **three discernment failures** your prompt's specificity is meant to prevent — refer to lec_11's pitfalls (P1–P4) and the 4Ds.

```python
# Your prompt below (Python triple-quoted string).
prompt = '''
# Your prompt here.
'''
print(prompt)

# Comment: three discernment failures this prompt's specificity prevents:
# 1. ...
# 2. ...
# 3. ...
```

## S1 — Lasso-then-OLS (stretch) [O1]

On the polynomial features from **E7**, run `Lasso(alpha=0.01)`, identify which features have non-zero coefficients, then refit plain `LinearRegression` using **only those features**. Compare the test MSE of the refit OLS against Lasso's own test MSE.

Why might Lasso-then-OLS sometimes win, and what is the risk?

```python
# Your code here.
```

## S2 — Bootstrap coefficient distribution (stretch) [O2]

For the multiple regression on `grades_factors` (six features, sklearn), compute the **bootstrap distribution** of the `calc_hs` coefficient using **200 resamples** (sample rows with replacement, refit `LinearRegression`, record the coefficient).

Plot a histogram. Report the bootstrap **mean** and **2.5 / 97.5 percentiles** as an empirical 95% confidence interval, and compare the mean to the point estimate from the full data.

```python
# Your code here.
# Hint:
#   n_boot = 200
#   for i in range(n_boot):
#       idx = rng.integers(0, n, size=n)
#       LinearRegression().fit(X.iloc[idx], y.iloc[idx])
```

## S3 — Variance Inflation Factor (stretch) [O3]

Compute the **VIF** for each of the six features in `grades_factors.xlsx` manually:

$$\mathrm{VIF}_j = \frac{1}{1 - R^2_j}$$

where $R^2_j$ is the R² of a regression of feature $j$ on all other features (use sklearn `LinearRegression`).

Print a sorted table feature → VIF. Flag any feature with VIF > 5 (moderate collinearity) or VIF > 10 (high). One sentence on what action this would prompt — drop the feature, combine correlated features, or switch to Ridge?

```python
# Your code here.
# Hint:
#   for j in features:
#       others = [c for c in features if c != j]
#       r2 = LinearRegression().fit(X[others], X[j]).score(X[others], X[j])
#       vif = 1 / (1 - r2)
```

## S4 — ElasticNet on correlated features (stretch) [O4]

Lasso picks **one** of two highly-correlated features somewhat arbitrarily; ElasticNet (`l1_ratio` between 0 and 1) lets you blend L1's sparsity with L2's stability. On the polynomial features from **E8**, fit `ElasticNet(alpha=0.01, l1_ratio=0.5)` and compare against the Lasso fit you already did. Report: how many non-zero coefficients does each keep, and what is the test MSE of each? Then sweep `l1_ratio ∈ [0.1, 0.3, 0.5, 0.7, 0.9]` with `alpha=0.01` fixed — at which `l1_ratio` does the test MSE bottom out?

```python
# Your code here.
```

## S5 — Robust regression on outlier-contaminated data (stretch) [O5]

Generate 50 clean points around `y = 2 + 1.3*x + N(0, 0.5)` for `x` in `[0, 10]`. Add **5 outliers** with `y` in `[20, 30]` (extreme y, normal x). Fit four regressors on the *combined* data: plain `LinearRegression`, `HuberRegressor`, `TheilSenRegressor(random_state=RANDOM_STATE)`, and `RANSACRegressor(random_state=RANDOM_STATE)`. Print each model's slope and compare against the true slope (1.3). Which regressors recover the true slope, and which were dragged toward the outliers?

```python
# Your code here.
```

## S6 — Bayesian + quantile regression (stretch) [O6]

On `grades_factors`, fit two non-OLS regressors:

1. **`BayesianRidge`** — print the posterior mean and standard deviation for the prediction on the *first three rows*. What does a per-row prediction interval tell you that a single point estimate does not?
2. **`QuantileRegressor(quantile=0.1)`** and `QuantileRegressor(quantile=0.9)` — fit both, print the predicted 10th- and 90th-percentile `calc` for the first three rows. When would a tail-quantile prediction be more useful than the OLS mean prediction?

```python
# Your code here.
```

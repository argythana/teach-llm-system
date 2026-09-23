<!-- source: lectures_07_13_pandas_plots_scikit/lecture_12_logistic_regression_naive_bayes_svm/reading_material/lec_12b_logit_probit_values_predictions.ipynb @ 0cc874704aaa -->

# Lecture 12b — Logistic Regression: Logit, Probit, Values, Predictions

> **Status: Mandatory reading.** Owns goals **G2**, **G3**, **G7**.  
> Builds on `lec_12a` (intuition) and `lec_11a` (three-API tour for regression).

This notebook stays on **binary** logistic regression and shows it from four angles.

**The pitch in four bullets:**

- Three Python APIs — `statsmodels.Logit`, `pingouin.logistic_regression`, `sklearn.LogisticRegression`.
- Two link functions — logit and probit — that disagree on coefficient scale but agree on predictions.
- Four output quantities — (a) logit value (linear predictor / log-odds), (b) probit z-score, (c) predicted probability, (d) predicted class.
- Lecture map: §A imports · §B Spector data · §C `Logit` · §D `Probit` · §E pingouin · §F sklearn · §G one observation end-to-end · §H when to reach for which · §I multi-class · §J recap.

## §A. Imports and setup

One imports cell, in the order the notebook uses them: pandas/numpy/matplotlib for data, the three modelling libraries, sklearn helpers for the multi-class slice.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# API 1 — statsmodels (inference-flavoured; gives Logit AND Probit)
import statsmodels.api as sm
from statsmodels.datasets import spector

# API 2 — pingouin (tidy-DataFrame wrapper)
import pingouin as pg

# API 3 — scikit-learn (prediction-flavoured)
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

RANDOM_STATE = 42
np.set_printoptions(suppress=True, precision=3)

print(f"sklearn version: {sklearn.__version__}")
```

## §B. The Spector dataset

We use the **Spector and Mazzeo (1980)** "program effectiveness" dataset, a classic small binary-LR teaching dataset reproduced in Greene's *Econometric Analysis*.  
It ships with statsmodels — no download needed.

**32 rows, 3 features, 1 binary target:**

- `GPA` — student's grade-point average before the course.
- `TUCE` — score on a prior test of economics understanding.
- `PSI` — 1 if the student was taught with the new pedagogical method (Personalised System of Instruction), 0 otherwise.
- `GRADE` — **the target.** 1 if the student's grade in the course improved, 0 otherwise.

Small enough to print in full, real enough to behave like real data.

```python
spec = spector.load_pandas()
df = spec.data.copy()
df.head()
```

```python
print(f"shape: {df.shape}")
print(f"target balance:\n{df['GRADE'].value_counts()}")
df.describe().round(2)
```

## Vocabulary: coefficient values, log-odds, probit

Three concepts the rest of this notebook mixes constantly — easy to confuse if you have not seen them named separately:

- **Coefficient values** — the raw numbers a fitted model returns (e.g. `2.826` for the GPA coefficient on Spector). Not probabilities; not directly comparable across model families. They are **slope units on the linear-predictor scale of the model you chose**.
- **Log-odds (logit)** — `log(p / (1 − p))`. The **Logit model's** linear predictor `z = a + b·x` is in **log-odds units**. A Logit coefficient `b` means *one extra unit of `x` adds `b` to the log-odds*, which **multiplies the odds by `e^b`**. (Same vocabulary as `lec_12a §C`.)
- **Probit** — uses the **inverse standard-Normal CDF** `Φ⁻¹` as the link function instead of the logit. The **Probit model's** linear predictor is in **standard-Normal z-score units**. A Probit coefficient `b` shifts a latent z-score by `b` per unit of `x`. **Not the same units as log-odds.**

So **"coefficient"** does not have one meaning — its meaning depends on the link function. Logit and Probit are two different links applied to the same statistical problem; their coefficients are not on the same scale, even when they are fitting the same data.

**Probit in plain English.**   
Imagine each person has a hidden "risk score" that lives somewhere on a **bell curve** (mean 0, standard deviation 1). If their score is above zero, the event happens; if below zero, it doesn't. Probit regression predicts where on the bell curve each person sits, from the features. A probit coefficient `b` tells you: *one extra unit of `x` moves a person `b` standard deviations along the bell curve toward (or away from) the event-happens side*. To turn a risk score back into a probability, you ask: *what fraction of the bell curve sits below this score?* — that fraction is the predicted probability. (The symbol `Φ` is just the name of that "fraction-of-the-bell-curve-to-the-left" function.)

### Why Logit ≈ 1.6 × Probit (same effect, different yardsticks)

The logistic distribution has variance `π²/3 ≈ 3.29`; the standard normal has variance `1`. The standard-deviation ratio is `√3.29 ≈ 1.81`, so **Logit coefficients are typically 1.6–1.8× the corresponding Probit coefficients** on the same data.

| quantity | scale | what "+1 unit of `x`" means |
| --- | --- | --- |
| **Logit coefficient** `b_logit`   | log-odds          | `log-odds += b_logit`; **odds × `e^b_logit`** |
| **Probit coefficient** `b_probit` | standard-normal z | latent score shifts by `b_probit` SDs |
| **Predicted probability**         | `[0, 1]`          | `sigmoid(logit linear predictor)` **or** `Φ(probit linear predictor)` |

**Bottom line**:

- **Probabilities are the only quantity directly comparable across model families.** Logit and Probit produce nearly identical fitted probabilities (usually within 0.01–0.02 of each other) — even though their coefficients look very different.
- **Coefficients are not.** Reading a Probit coefficient as if it were a log-odds is a classic beginner mistake.
- §C fits the Logit (you read in log-odds). §D fits the Probit (you read in z-scores). §F numerically verifies that the **probability** outputs converge across libraries.

## §C. Fit with `statsmodels.Logit`

`statsmodels` is the inference-flavoured library: its `.summary()` table is the canonical stats-report output.  
Two quirks to remember from `lec_11a`:

- The intercept is **not added automatically** — call `sm.add_constant(X)`.
- The signature is `(endog, exog)` — **target first**, features second. Opposite of sklearn.

```python
y = df['GRADE']
X = df[['GPA', 'TUCE', 'PSI']]
X_with_const = sm.add_constant(X)
X_with_const.head()
```

```python
logit_res = sm.Logit(y, X_with_const).fit(disp=0)
logit_res.params.round(4)
```

### ⏱ Pause-point — read the summary table before moving on

The next cell prints the full `.summary()`. Take a minute to read it top-to-bottom before the walk-through below.

```python
print(logit_res.summary())
```

### How to read the Logit summary

The columns you actually use day-to-day:

- **`coef`** — the logit (log-odds) coefficient for each feature.  
  For `PSI` (binary), a coef of ~2.38 means: switching from PSI=0 to PSI=1 raises the **log-odds** of improving by 2.38. Equivalently, the **odds** are multiplied by exp(2.38) ≈ 10.7.
- **`std err`** — the standard error of the coefficient estimate.
- **`z` and `P>|z|`** — the significance test on the coefficient. Logit uses a z-test, not a t-test (large-sample approximation, no residual variance to estimate).
- **`Pseudo R-squ.`** — McFadden's pseudo R². Useful as a relative comparison between logistic models on the **same** dataset. **Not** on the same scale as linear-regression R² — don't compare a 0.37 here to a 0.37 in `lec_11a`.
- **`Log-Likelihood`** — the model's log-likelihood at the maximum-likelihood estimates (the objective the fitter maximises).

## §D. Fit with `statsmodels.Probit`

### Probit in everyday words

Think of every observation as having a hidden **risk number** that lives somewhere on a **standard bell curve** (the one with mean 0 and standard deviation 1):

- If the risk number is **above zero**, the event happens. If **below zero**, it doesn't.
- The features push the risk number left or right. A **positive probit coefficient** moves the risk number to the right (toward the event); a **negative coefficient** moves it left.
- To turn a risk number into a probability, ask: *what fraction of the bell curve sits to the left of this point?* That fraction is the predicted probability:
  - risk number = **0**  → **50%** chance (half the bell is to the left of zero),
  - risk number = **+1** → about **84%** chance,
  - risk number = **+2** → about **97.5%** chance,
  - risk number = **−1** → about **16%** chance,
  - risk number = **−2** → about **2.5%** chance.

The math symbol `Φ` (capital Greek phi) is just a short name for that *fraction-of-the-bell-curve-to-the-left* function — so `Φ(0) = 0.5`, `Φ(+1) ≈ 0.84`, and so on.

### The formal definition

Probit has the **same model family** as logit — a GLM with a binary target — but a **different link function**.

- **Logit** maps the linear predictor through the logistic CDF: `p = 1 / (1 + exp(-z))`.
- **Probit** maps the linear predictor through the standard normal CDF: `p = Φ(z)`.

Both produce a probability in (0, 1) from a linear predictor in (−∞, +∞). They disagree on the **scale** of the coefficients but agree on the **predictions**.

```python
probit_res = sm.Probit(y, X_with_const).fit(disp=0)
print(probit_res.summary())
```

```python
# Coefficient scale comparison — Logit ~ 1.6 x Probit is the textbook rule of thumb.
scale_df = pd.DataFrame({
    'logit_coef':  logit_res.params,
    'probit_coef': probit_res.params,
    'logit/probit ratio': logit_res.params / probit_res.params,
}).round(3)
scale_df
```

```python
# Prediction agreement — fitted probabilities on the training set.
logit_pred  = logit_res.predict()
probit_pred = probit_res.predict()
abs_diff = pd.Series(logit_pred - probit_pred).abs()
abs_diff.describe().round(4)
```

### Logit vs Probit — the bullet contrast

- **Same model family.** Both are GLMs with a binary target and a linear predictor.
- **Different link function.** Logit uses `log(p/(1-p))`; Probit uses `Φ⁻¹(p)`.
- **Coefficients are NOT directly comparable.** Logit coefficients ≈ 1.6 × Probit coefficients (rule of thumb). The ratio above is close to 1.6 for the non-intercept terms.
- **Predictions are nearly identical.** The absolute differences on the fitted probabilities are typically below 0.02 — the two links agree on **what** to predict, even when their coefficient scales differ.
- **Why ML defaults to Logit.** Coefficients are interpretable as **log-odds**, the loss is convex and fast, and the convention across sklearn / xgboost / PyTorch is the sigmoid (logit) link. Probit lives mostly in econometrics.

## §E. Fit with `pingouin.logistic_regression`

`pingouin` is the tidy-output library: a single function call returns a `DataFrame` you can sort, filter, and pipe.  
Internally it is a thin wrapper around sklearn's `LogisticRegression` with regularization disabled — so the coefficients match the maximum-likelihood estimates from statsmodels.

Signature: `pg.logistic_regression(X, y)` — features first, target second. Like sklearn. Unlike statsmodels.

```python
pg_results = pg.logistic_regression(X, y)
pg_results.round(4)
```

```python
# Odds ratios — pingouin returns coefficients on the log-odds scale; we add exp(coef).
pg_with_or = pg_results.copy()
pg_with_or['odds_ratio'] = np.exp(pg_with_or['coef'])
pg_with_or[['names', 'coef', 'odds_ratio', 'pval']].round(4)
```

### Reading the pingouin table

- `coef` matches `logit_res.params` (within rounding) — same MLE fit underneath.
- `odds_ratio = exp(coef)` is the multiplicative effect on the odds. For `PSI` ≈ 10.7: students taught with PSI have ~10× the odds of improving, holding GPA and TUCE constant.
- `pval` matches statsmodels' `P>|z|`.

Use pingouin when you want odds ratios in a one-liner and the output as a `DataFrame`. Use statsmodels when you also need confidence intervals, residual diagnostics, AIC/BIC.

## §F. Fit with `sklearn.LogisticRegression`

`sklearn` is the prediction-flavoured library. The same `.fit() / .predict() / .predict_proba() / .decision_function() / .score()` interface works for **every classifier** in the ecosystem — that is the killer feature.

Two settings we pin so the coefficients match statsmodels:

- `penalty=None` — sklearn's **silent default is L2 regularization** with `C=1.0`. We turn it off to recover the unregularized MLE. (`lec_12c §F` walks the default-regularization trap.)
- `solver='lbfgs'`, `max_iter=10000` — gives the optimiser room to converge on a small problem.

```python
# sklearn does NOT want a constant column — fit_intercept=True (default) handles it.
X_no_const = X.copy()

sk_logit = LogisticRegression(
    penalty=None,
    solver='lbfgs',
    max_iter=10000,
    random_state=RANDOM_STATE,
).fit(X_no_const, y)

print(f"intercept: {sk_logit.intercept_}")
print(f"coef:      {sk_logit.coef_}")
print(f"classes:   {sk_logit.classes_}")
```

```python
# Side-by-side coefficient check — should match statsmodels.Logit to several decimals.
compare = pd.DataFrame({
    'statsmodels.Logit': logit_res.params.values,
    'sklearn (penalty=None)': np.concatenate([sk_logit.intercept_, sk_logit.coef_[0]]),
}, index=['const', 'GPA', 'TUCE', 'PSI']).round(4)
compare
```

### The three sklearn output methods

On the same fitted model, sklearn exposes three prediction methods — and they are the **four output quantities** of the lecture title, with `predict_proba` carrying two of them in one array.

- `.decision_function(X)` → the **logit value** (linear predictor / log-odds) `z = b₀ + b·x`.
- `.predict_proba(X)` → the **predicted probability** of each class, columns ordered by `classes_`. For binary problems: `[P(class=0), P(class=1)]`.
- `.predict(X)` → the **predicted class label** (argmax of probabilities; for binary, equivalent to thresholding `P(class=1)` at 0.5).

```python
# Three predictions on the same rows — first 5 observations for readability.
z_vals  = sk_logit.decision_function(X_no_const)[:5]
probas  = sk_logit.predict_proba(X_no_const)[:5]
classes = sk_logit.predict(X_no_const)[:5]

preview = pd.DataFrame({
    'logit_value_z': z_vals.round(3),
    'P(GRADE=0)':    probas[:, 0].round(3),
    'P(GRADE=1)':    probas[:, 1].round(3),
    'predicted':     classes.astype(int),
    'actual':        y.iloc[:5].astype(int).values,
})
preview
```

```python
# Sanity check: sigmoid(z) should equal predict_proba(class=1).
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
from_z  = sigmoid(z_vals)
from_pp = probas[:, 1]
np.allclose(from_z, from_pp)
```

## §G. Walk ONE observation end-to-end

Take the first row. Recompute everything by hand from the statsmodels coefficients and check that sklearn agrees.

**The chain:** features → linear predictor `z` → probability `σ(z)` → class (threshold 0.5).

```python
obs = X.iloc[0]
print(f"Observation 0:\n{obs}")
print(f"\nActual GRADE: {int(y.iloc[0])}")
```

```python
# Step 1 — linear predictor z from the LOGIT coefficients (hand arithmetic).
b = logit_res.params
z_logit = (b['const']
           + b['GPA']  * obs['GPA']
           + b['TUCE'] * obs['TUCE']
           + b['PSI']  * obs['PSI'])
print(f"logit z (by hand)            : {z_logit:.4f}")
print(f"logit z (sklearn decision_fn): {sk_logit.decision_function(X_no_const.iloc[[0]])[0]:.4f}")

# Step 2 — the probit z-score for the SAME observation, from the PROBIT coefficients.
bp = probit_res.params
z_probit = (bp['const']
            + bp['GPA']  * obs['GPA']
            + bp['TUCE'] * obs['TUCE']
            + bp['PSI']  * obs['PSI'])
print(f"probit z (by hand): {z_probit:.4f}")
print("Note: this z is on a DIFFERENT scale than the logit z above.")
```

```python
# Step 3 — predicted probability from the LOGIT z, via the sigmoid.
p_hand = 1.0 / (1.0 + np.exp(-z_logit))
p_sk   = sk_logit.predict_proba(X_no_const.iloc[[0]])[0, 1]
print(f"P(GRADE=1) by hand : {p_hand:.4f}")
print(f"P(GRADE=1) sklearn : {p_sk:.4f}")

# Step 4 — predicted class, threshold = 0.5.
class_hand = 1 if p_hand >= 0.5 else 0
class_sk   = int(sk_logit.predict(X_no_const.iloc[[0]])[0])
print(f"predicted class by hand : {class_hand}")
print(f"predicted class sklearn : {class_sk}")
print(f"actual class            : {int(y.iloc[0])}")
```

**What just happened.**  Four output quantities, one observation, perfect agreement between hand arithmetic and sklearn.

- `z_logit` — the linear predictor from logit coefficients. Sklearn's `decision_function`.
- `z_probit` — the linear predictor from probit coefficients. Different scale, same role.
- `p_hand` — sigmoid of the logit z. Sklearn's `predict_proba` (class-1 column).
- `class_hand` — threshold the probability at 0.5. Sklearn's `predict`.

Internalise this chain. Every binary classifier in `lec_12e` (Naive Bayes) and `lec_12f` (SVM) exposes the same three method names — only the way they get to `z` changes.

## §H. When to reach for which API

| Use case | Reach for |
| --- | --- |
| Writing a stats report with p-values, std errors, AIC/BIC, confidence intervals | `statsmodels.Logit` |
| A tidy single-line API that returns a `DataFrame` with odds ratios | `pingouin.logistic_regression` |
| An ML pipeline with `.predict` / `.predict_proba` / `.decision_function` | `sklearn.LogisticRegression` |
| Probit link (uncommon in ML, common in econometrics) | `statsmodels.Probit` |

From `lec_12c` onwards the lecture uses **sklearn only** — the rest of the classification story (assumptions, hyperparameters, Naive Bayes, SVM, the model-selection workflow in lecture 13) is predictive and lives in the sklearn estimator ecosystem.

## §I. Multi-class extension on iris

### ⏱ Pause-point — optional bucket

Everything above is binary. Real classification problems often have more than two classes.  
Sklearn supports multi-class logistic regression two ways:

- **Multinomial** (the modern default) — one joint softmax fit over all classes.
- **One-vs-Rest (OVR)** — one binary classifier per class, then take the highest-scoring one.

**Version note.** Up to sklearn 1.4, `LogisticRegression` had a `multi_class` keyword to switch between these. It was deprecated in 1.5 and **removed in 1.7+**.  
On sklearn 1.7+ (this notebook runs on 1.8) the calling convention is:

- **Multinomial** — just `LogisticRegression(...)`. That is now the only built-in mode.
- **OVR** — wrap with `OneVsRestClassifier(LogisticRegression(...))` from `sklearn.multiclass`.

```python
# Load iris (4 features, 3 classes); split stratified 70/30.
iris = load_iris(as_frame=True)
X_iris, y_iris = iris.data, iris.target
```

```python

X_tr, X_te, y_tr, y_te = train_test_split(
    X_iris, y_iris, test_size=0.30, random_state=RANDOM_STATE, stratify=y_iris,
)
```

```python


# Strategy 1: multinomial (the modern default on sklearn ≥1.7) — one joint softmax fit.
multi_lr = LogisticRegremulti_lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE).fit(X_tr, y_tr)ssion(solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE).fit(X_tr, y_tr)
```

```python

# Strategy 2: One-vs-Rest — explicit wrapper since multi_class= is LogisticRegression(removed on sklearn ≥1.7.
ovr_lr = OneVsRestClassifier(
    LogisticRegression(solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE),
).fit(X_tr, y_tr)
```

```python

# Coefficient shapes + per-strategy test accuracy + prediction agreement.
print(f'multinomial coef_ shape: {multi_lr.coef_.shape}  test acc: {multi_lr.score(X_te, y_te):.3f}')
print(f'OVR (stacked)   shape:   ({len(ovr_lr.estimators_)}, {ovr_lr.estimators_[0].coef_.shape[1]})  test acc: {ovr_lr.score(X_te, y_te):.3f}')

agree = (multi_lr.predict(X_te) == ovr_lr.predict(X_te)).mean()
print(f'prediction agreement between the two strategies: {agree:.1%}')
```

**Reading the multi-class comparison.**

- **Coefficient shapes match.** Multinomial fits one row of weights per class; OVR fits one binary classifier per class and we stacked them. Both end up `(3 classes, 4 features)`.
- **Predictions agree on the vast majority of rows.** On iris (well-separated classes) the two strategies are nearly indistinguishable.
- **When they differ.** Multinomial is the modern default — it calibrates probabilities jointly across classes (they sum to 1 by construction). OVR's per-class probabilities have to be normalised. For most teaching examples either is fine; for production multi-class problems, trust the default.

## §J. Recap and what is next

- **Three APIs, one answer.** `statsmodels.Logit` for inference reports, `pingouin.logistic_regression` for tidy odds-ratio tables, `sklearn.LogisticRegression` for the ML pipeline. All three returned the same MLE coefficients.
- **Two link functions.** Logit and Probit. Coefficients on different scales (Logit ≈ 1.6 × Probit), predictions agree within ~0.02.
- **Four output quantities, one chain.** features → `decision_function` (logit z) → `predict_proba` (sigmoid of z) → `predict` (threshold at 0.5). Plus Probit's z as the alternative linear predictor.
- **Multi-class.** On sklearn 1.7+ the default is multinomial; for One-vs-Rest, wrap with `OneVsRestClassifier`.

**Forward pointers:**

- **`lec_12c`** — honest limitations and traps: linearity of the logit, multicollinearity, perfect separation, class imbalance, silent default L2.
- **`lec_12d`** (optional) — full hyperparameter tour: `C`, `penalty`, `solver`, `class_weight`, solver–penalty compatibility.

## Further reading

- [statsmodels — Discrete Choice Models](https://www.statsmodels.org/stable/discretemod.html) — the `Logit` and `Probit` model classes with their full summary-table output.
- [pingouin.logistic_regression — API reference](https://pingouin-stats.org/generated/pingouin.logistic_regression.html) — the tidy results format used in §E, including the explicit odds-ratio computation.
- [scikit-learn — `LogisticRegression` API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) — the `predict` / `predict_proba` / `decision_function` triple walked in §F and §G.
- [Greene, *Econometric Analysis* (Spector data source)](https://pages.stern.nyu.edu/~wgreene/Text/econometricanalysis.htm) — the textbook the Spector "program effectiveness" dataset comes from.

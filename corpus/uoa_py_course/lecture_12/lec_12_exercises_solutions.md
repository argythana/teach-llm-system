<!-- source: lectures_07_13_pandas_plots_scikit/lecture_12_logistic_regression_naive_bayes_svm/practice_exercises/lec_12_exercises_solutions.ipynb @ 0cc874704aaa -->

# Lecture 12 — Practice Exercises (Solutions)

Runnable solutions for every required exercise (E1–E7) and the stretch exercise (S1).

**Use this notebook only after a serious attempt at the exercises notebook.** Reading the solution before trying the exercise is the fastest way to convince yourself you understand the material when you do not.

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_STATE = 42
np.set_printoptions(suppress=True, precision=3)
```

## E1 — When to use logistic regression *(tied to G1)*

Below are three problem statements. For each one, decide whether **logistic regression** is the right tool and write one sentence explaining why.

1. **Predict next month's revenue** for a retail chain, given the last 24 months of weekly revenue, holiday flags, and a promotion calendar.
2. **Predict whether a customer will churn** next month, given their last 90 days of transaction count, customer-service contacts, and tenure.
3. **Predict house sale price** in euros, given square metres, number of bedrooms, postcode, and condition rating.

Fill in your answer in the markdown cell below (replace the `# Your answer here` lines).

**Solution:**

1. **Predict revenue → NOT logistic regression.** The target is continuous (euros), not a discrete class — this is regression (lecture 11).
2. **Predict churn → YES, logistic regression.** Binary target (will-churn / will-stay), labelled training set, every new customer needs a prediction. Classic LR use case.
3. **Predict house price → NOT logistic regression.** The target is continuous (euros) — regression (lecture 11). LR would only fit if we transformed price into a discrete bucket (e.g. high / medium / low), but information is lost in the bucketing.

## E2 — Fit logistic regression with three Python APIs *(tied to G2)*

Use the Spector "program effectiveness" dataset (`statsmodels.datasets.spector`) — 32 rows, 3 features (`GPA`, `TUCE`, `PSI`), 1 binary target (`GRADE`).

Fit binary logistic regression three times — once with each API:

- `statsmodels.Logit(y, sm.add_constant(X)).fit()`
- `pingouin.logistic_regression(X, y)`
- `sklearn.linear_model.LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000).fit(X, y)`

Print the coefficients from each and confirm they all agree (the three should be within `1e-3` of each other on each feature).

```python
import statsmodels.api as sm
import pingouin as pg
from statsmodels.datasets import spector

spec = spector.load_pandas()
X = spec.data[['GPA', 'TUCE', 'PSI']]
y = spec.data['GRADE'].astype(int)

# 1. statsmodels.Logit
X_const = sm.add_constant(X)
logit_res = sm.Logit(y, X_const).fit(disp=0)
sm_coefs = logit_res.params

# 2. pingouin
pg_res = pg.logistic_regression(X, y)
pg_coefs = pg_res.set_index('names')['coef']

# 3. sklearn (penalty=None to match the unregularized MLE)
sk = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000).fit(X, y)
sk_coefs = pd.Series(
    np.r_[sk.intercept_, sk.coef_.ravel()],
    index=['const', 'GPA', 'TUCE', 'PSI'],
)

comparison = pd.DataFrame({
    'statsmodels': sm_coefs,
    'pingouin':    pg_coefs,
    'sklearn':     sk_coefs,
})
print(comparison.round(4))

max_diff = (comparison.max(axis=1) - comparison.min(axis=1)).abs().max()
print(f'\nmax disagreement across the three APIs: {max_diff:.2e}')
assert max_diff < 1e-2, 'coefficients should agree to ~1e-2 or better'
```

## E3 — Walk one observation from features to class label *(tied to G3)*

Using the Spector fit from E2 (sklearn version), pick the **first observation** in the dataset. Compute by hand:

1. The **linear predictor** (logit / log-odds): `z = intercept + coef · x`
2. The **predicted probability**: `p = 1 / (1 + exp(-z))`
3. The **predicted class**: `1 if p >= 0.5 else 0`

Then verify that your by-hand values agree with `sklearn`'s `decision_function`, `predict_proba`, and `predict` for the same row.

```python
x0 = X.iloc[0].values  # the first observation

z_byhand = float(sk.intercept_[0] + sk.coef_[0] @ x0)
p_byhand = float(1.0 / (1.0 + np.exp(-z_byhand)))
class_byhand = int(p_byhand >= 0.5)

z_sklearn = float(sk.decision_function([x0])[0])
p_sklearn = float(sk.predict_proba([x0])[0, 1])
class_sklearn = int(sk.predict([x0])[0])

print(f'                by hand     sklearn')
print(f'linear predictor  {z_byhand:+.4f}    {z_sklearn:+.4f}')
print(f'probability       {p_byhand:.4f}     {p_sklearn:.4f}')
print(f'class label       {class_byhand}          {class_sklearn}')

assert np.isclose(z_byhand, z_sklearn)
assert np.isclose(p_byhand, p_sklearn)
assert class_byhand == class_sklearn
```

## E4 — The 0.5-threshold trap on imbalanced data *(tied to O1 — stretch, optional goal)*

Construct a binary classification problem with a **95 / 5 class imbalance** using `make_classification(weights=[0.95, 0.05], n_samples=2000, n_features=4, random_state=RANDOM_STATE)`.

Fit two logistic regressions on the same train/test split:

- `LogisticRegression()` (the default — `class_weight=None`).
- `LogisticRegression(class_weight='balanced')`.

For each, print accuracy + a `classification_report`. Compare the **recall on the minority class (label 1)** between the two. Comment in one line on why accuracy alone is misleading here.

```python
X_imb, y_imb = make_classification(
    weights=[0.95, 0.05],
    n_samples=2000, n_features=4, n_informative=2, n_redundant=0,
    random_state=RANDOM_STATE,
)

X_train, X_test, y_train, y_test = train_test_split(
    X_imb, y_imb, test_size=0.30, stratify=y_imb, random_state=RANDOM_STATE,
)

for cw, label in [(None, 'default'), ('balanced', 'balanced')]:
    clf = LogisticRegression(class_weight=cw, max_iter=1000, random_state=RANDOM_STATE).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'=== class_weight={label} ===')
    print(f'accuracy: {acc:.3f}')
    print(classification_report(y_test, y_pred, digits=3))
```

**Why accuracy is misleading here:**

- The dataset is 95 % class 0. A model that always predicts class 0 hits 95 % accuracy.
- The default LR therefore scores ~95 % accuracy but **misses most class-1 examples** (recall on minority class is near zero).
- `class_weight='balanced'` reweights so that misclassifying a class-1 example is ~19× as costly as misclassifying a class-0 one — minority recall jumps, often at a small accuracy cost.

## E5 — Gaussian NB and the violated independence assumption *(tied to G4)*

1. Fit `GaussianNB` on iris with a 70 / 30 stratified train/test split.
2. Print test accuracy and the confusion matrix.
3. **Then** compute the pairwise Pearson correlations between the **four iris features within class 0 (setosa)**. Identify at least one pair with `|corr| > 0.3` — this violates the conditional-independence assumption.
4. In a markdown cell, write one sentence on why NB still works despite the violation.

```python
iris = load_iris(as_frame=True)
X_iris = iris.data
y_iris = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.30, stratify=y_iris, random_state=RANDOM_STATE,
)

nb = GaussianNB().fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(f'GaussianNB test accuracy: {accuracy_score(y_test, y_pred):.3f}')
print('\nconfusion matrix:')
print(confusion_matrix(y_test, y_pred))

# Within-class correlations for setosa
corr_setosa = X_iris.loc[y_iris == 0].corr()
print('\nWithin-setosa pairwise correlations:')
print(corr_setosa.round(3))

# Largest absolute off-diagonal correlation
off_diag = corr_setosa.where(~np.eye(corr_setosa.shape[0], dtype=bool))
max_corr = off_diag.abs().max().max()
print(f'\nlargest |corr| off-diagonal: {max_corr:.3f}')
```

**Why NB still works:**

- NB does not need calibrated probabilities — it only needs the right **argmax**.
- Correlated features double-count evidence, which biases the posterior magnitudes but rarely flips the argmax on well-separated classes like iris.
- The probability outputs from NB are therefore overconfident (close to 0 or 1), but the predicted *class* is still typically correct.

## E6 — SVM with linear and RBF kernels on iris *(tied to G5)*

Use the iris 2D slice (`petal length (cm)`, `petal width (cm)`). With a 70 / 30 stratified train/test split + a `StandardScaler` fit on train only:

1. Fit `SVC(kernel='linear', C=1.0)`. Report test accuracy.
2. Fit `SVC(kernel='rbf', C=1.0, gamma='scale')`. Report test accuracy.
3. Plot the decision boundaries side-by-side on the same axes for visual comparison.

```python
iris = load_iris(as_frame=True)
X2 = iris.data[['petal length (cm)', 'petal width (cm)']]
y2 = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X2, y2, test_size=0.30, stratify=y2, random_state=RANDOM_STATE,
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

svm_lin = SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE).fit(X_train_s, y_train)
svm_rbf = SVC(kernel='rbf',    C=1.0, gamma='scale', random_state=RANDOM_STATE).fit(X_train_s, y_train)

print(f'Linear SVM test accuracy: {accuracy_score(y_test, svm_lin.predict(X_test_s)):.3f}')
print(f'RBF SVM    test accuracy: {accuracy_score(y_test, svm_rbf.predict(X_test_s)):.3f}')

# Decision boundary plot
def plot_db(clf, X_s, y, title, ax):
    xmin, xmax = X_s[:, 0].min() - 0.5, X_s[:, 0].max() + 0.5
    ymin, ymax = X_s[:, 1].min() - 0.5, X_s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.scatter(X_s[:, 0], X_s[:, 1], c=y, edgecolor='k', s=30, cmap='viridis')
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_db(svm_lin, X_train_s, y_train.values, 'Linear kernel', axes[0])
plot_db(svm_rbf, X_train_s, y_train.values, 'RBF kernel',    axes[1])
plt.tight_layout()
plt.show()
```

## E7 — Distinguish `predict`, `predict_proba`, `decision_function` *(tied to G6)*

Fit three classifiers on the **full 4-feature iris** (70 / 30 stratified split):

- `LogisticRegression(max_iter=1000)`
- `GaussianNB()`
- `SVC(kernel='rbf', probability=True)`  ← note `probability=True` is required for `predict_proba`.

For the **first row of the test set**, print three things per classifier:

- The class label from `.predict`.
- The probability vector from `.predict_proba`.
- The raw score from `.decision_function` (if available — `GaussianNB` does not expose this).

Comment in one line: which classifier exposes which outputs natively?

```python
iris = load_iris(as_frame=True)
X_iris = iris.data
y_iris = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.30, stratify=y_iris, random_state=RANDOM_STATE,
)

lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X_train, y_train)
nb = GaussianNB().fit(X_train, y_train)
svc = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE).fit(X_train, y_train)

x0 = X_test.iloc[[0]]

for name, clf in [('LR', lr), ('GaussianNB', nb), ('SVC', svc)]:
    print(f'=== {name} ===')
    print(f'  predict        : {clf.predict(x0)[0]}')
    print(f'  predict_proba  : {np.round(clf.predict_proba(x0)[0], 3)}')
    if hasattr(clf, 'decision_function'):
        try:
            df_vals = clf.decision_function(x0)[0]
            print(f'  decision_func  : {np.round(df_vals, 3)}')
        except AttributeError:
            print(f'  decision_func  : not exposed by this estimator')
    else:
        print(f'  decision_func  : not exposed by this estimator')
```

**Which output each classifier exposes:**

- **`LogisticRegression`** — exposes all three (`predict`, `predict_proba` natively, `decision_function` returns the logit values).
- **`GaussianNB`** — exposes `predict` and `predict_proba` (the posterior). **No `decision_function`** — the raw score is the joint log-likelihood, accessed via `predict_log_proba` if you need it.
- **`SVC`** — exposes `predict` and `decision_function` natively. `predict_proba` only works if you set `probability=True` at fit time (which runs an internal cross-validation + Platt scaling).

## S1 — Stretch: LR hyperparameter sweep on `C` *(tied to O2)*

Vary `C ∈ {0.001, 0.01, 0.1, 1, 10, 100, 1000}` on iris (full 4 features). For each value:

1. Fit `LogisticRegression(penalty='l2', C=C, max_iter=2000)` on the full data.
2. Record the **L2 norm** of `.coef_`.

Plot the L2 norm of the coefficient vector against `log10(C)`. Identify the C-range where coefficients **stop shrinking** (the regularization stops biting).

*No `GridSearchCV` — this is a one-parameter sweep, the way Lecture 12 does it. Lecture 13 covers the grid-search version.*

```python
iris = load_iris(as_frame=True)
X_iris = iris.data
y_iris = iris.target

C_values = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
norms = []
for C in C_values:
    clf = LogisticRegression(penalty='l2', C=C, max_iter=2000, random_state=RANDOM_STATE).fit(X_iris, y_iris)
    norms.append(np.linalg.norm(clf.coef_))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.log10(C_values), norms, marker='o')
ax.set_xlabel('log10(C)')
ax.set_ylabel('L2 norm of coef_')
ax.set_title('LR coefficient norm vs regularization strength')
ax.grid(True, alpha=0.3)
plt.show()

for C, n in zip(C_values, norms):
    print(f'C = {C:7.3f}    ||coef||₂ = {n:.3f}')
```

**Reading the plot:**

- For small `C` (strong regularization, ≤ 0.01) coefficients are heavily shrunk.
- Around `C = 1`–`10` they grow rapidly with `C`.
- Past `C ≈ 100` the norm flattens — the regularization has effectively stopped biting and the fit is close to unregularized MLE.
- This is why `C = 1.0` is a sensible default for small-data problems and why tuning often shows accuracy plateaus around `C ∈ [1, 100]`.

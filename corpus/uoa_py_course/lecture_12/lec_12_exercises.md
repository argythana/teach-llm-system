<!-- source: lectures_07_13_pandas_plots_scikit/lecture_12_logistic_regression_naive_bayes_svm/practice_exercises/lec_12_exercises.ipynb @ 0cc874704aaa -->

# Lecture 12 — Practice Exercises

One required exercise per required goal (G1, G2, G3, G4, G5, G6) and two stretch exercises tied to the optional goals O1 (assumptions/imbalance — exercise E4) and O2 (LR hyperparameters — exercise S1). E4 was originally labelled as required when assumptions were goal G4; the 2026-05-26 goals renumbering moved assumptions to O1 and E4 with it.

**How to use this notebook:**

- Each exercise has a markdown prompt + one or more placeholder code cells with `# Your code here`.
- Fill in the placeholders. Run the cell. Repeat.
- The solutions live in `lec_12_exercises_solutions.ipynb`. Use them after you have made a serious attempt yourself.
- Required exercises (E1–E7) cover the **mandatory** material. Skip the stretch (S1) if class is running short.

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

**Your answer:**

1. *# Your answer here*
2. *# Your answer here*
3. *# Your answer here*

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

# Your code here:
# 1. Fit with statsmodels.Logit (remember to add a constant column).
# 2. Fit with pingouin.logistic_regression.
# 3. Fit with sklearn.LogisticRegression (use penalty=None to match the others).
# 4. Print the three sets of coefficients side by side.
```

## E3 — Walk one observation from features to class label *(tied to G3)*

Using the Spector fit from E2 (sklearn version), pick the **first observation** in the dataset. Compute by hand:

1. The **linear predictor** (logit / log-odds): `z = intercept + coef · x`
2. The **predicted probability**: `p = 1 / (1 + exp(-z))`
3. The **predicted class**: `1 if p >= 0.5 else 0`

Then verify that your by-hand values agree with `sklearn`'s `decision_function`, `predict_proba`, and `predict` for the same row.

```python
# Assume `sk` is the sklearn LR fit from E2.
# x0 = X.iloc[0]   # the first observation

# Your code here:
# 1. Compute z by hand from sk.intercept_ + sk.coef_ @ x0.
# 2. Compute p = 1 / (1 + np.exp(-z)).
# 3. Compute the predicted class label by thresholding at 0.5.
# 4. Compare to sk.decision_function([x0]), sk.predict_proba([x0]), sk.predict([x0]).
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

# Your code here:
# 1. Fit two LRs: one default, one with class_weight='balanced'.
# 2. Print accuracy + classification_report for each.
# 3. Identify the minority-class recall difference.
```

## E5 — Gaussian NB and the violated independence assumption *(tied to G4)*

1. Fit `GaussianNB` on iris with a 70 / 30 stratified train/test split.
2. Print test accuracy and the confusion matrix.
3. **Then** compute the pairwise Pearson correlations between the **four iris features within class 0 (setosa)**. Identify at least one pair with `|corr| > 0.3` — this violates the conditional-independence assumption.
4. In a markdown cell, write one sentence on why NB still works despite the violation.

```python
iris = load_iris(as_frame=True)
X_iris = iris.data
y_iris = iris.target

# Your code here:
# 1. Train/test split, stratify=y_iris, test_size=0.30.
# 2. Fit GaussianNB, predict on test, print accuracy + confusion matrix.
# 3. Compute the 4x4 correlation matrix on X_iris.loc[y_iris == 0].
# 4. Find the largest absolute off-diagonal correlation; print it.
```

## E6 — SVM with linear and RBF kernels on iris *(tied to G5)*

Use the iris 2D slice (`petal length (cm)`, `petal width (cm)`). With a 70 / 30 stratified train/test split + a `StandardScaler` fit on train only:

1. Fit `SVC(kernel='linear', C=1.0)`. Report test accuracy.
2. Fit `SVC(kernel='rbf', C=1.0, gamma='scale')`. Report test accuracy.
3. Plot the decision boundaries side-by-side on the same axes for visual comparison.

```python
iris = load_iris(as_frame=True)
X2 = iris.data[['petal length (cm)', 'petal width (cm)']]
y2 = iris.target

# Your code here:
# 1. Train/test split, stratify.
# 2. StandardScaler — fit on train only.
# 3. Fit SVC(kernel='linear', C=1.0). Print test accuracy.
# 4. Fit SVC(kernel='rbf',    C=1.0, gamma='scale'). Print test accuracy.
# 5. Plot decision boundaries side by side.
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
# Your code here:
# 1. Load iris (4 features), train/test split, stratify.
# 2. Fit LogisticRegression, GaussianNB, SVC(probability=True).
# 3. For X_test.iloc[0]: print predict / predict_proba / decision_function for each.
```

## S1 — Stretch: LR hyperparameter sweep on `C` *(tied to O2)*

Vary `C ∈ {0.001, 0.01, 0.1, 1, 10, 100, 1000}` on iris (full 4 features). For each value:

1. Fit `LogisticRegression(penalty='l2', C=C, max_iter=2000)` on the full data.
2. Record the **L2 norm** of `.coef_`.

Plot the L2 norm of the coefficient vector against `log10(C)`. Identify the C-range where coefficients **stop shrinking** (the regularization stops biting).

*No `GridSearchCV` — this is a one-parameter sweep, the way Lecture 12 does it. Lecture 13 covers the grid-search version.*

```python
# Your code here:
# 1. Loop over C_values = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000].
# 2. For each, fit LogisticRegression(penalty='l2', C=C, max_iter=2000) on the full iris X, y.
# 3. Record np.linalg.norm(clf.coef_).
# 4. Plot the L2 norm vs log10(C).
```

<!-- source: lectures_07_13_pandas_plots_scikit/lecture_12_logistic_regression_naive_bayes_svm/reading_material/lec_12e_naive_bayes.ipynb @ 0cc874704aaa -->

# Lecture 12e — Gaussian Naive Bayes

**Status: Mandatory reading.**


## The pitch

- Naive Bayes is the **fastest** classifier in scikit-learn. It fits in `O(n)` over the training rows and predicts in `O(n_features)` per new observation — no distance lookups, no gradient steps, no kernel matrices.
- It rests on a single simplifying lie: **conditional independence** of the features given the class. That assumption is almost never true on real data, yet the model often works anyway.
- This notebook walks `GaussianNB` on the iris dataset end-to-end — fit, inspect, predict, evaluate — then names the right NB variant for non-Gaussian feature types.

## Three industry uses (classification mode)

- **Email spam filtering.** The original 1990s baseline. Production spam filters used (and still use) `MultinomialNB` on word counts. This notebook uses `GaussianNB` on continuous flower measurements; the multinomial variant for text is named in §H.
- **Sentiment classification.** A bag-of-words sentiment baseline before reaching for a transformer. NB trains in seconds on a laptop, gives a number to beat, and tells you whether the problem needs anything fancier at all.
- **Real-time medical triage.** When latency matters and a prediction must come back in microseconds, NB's `O(n_features)` prediction wins over logistic regression's matrix multiply and SVM's distance-to-support-vectors loop.

## Learning-goals home

This notebook owns **G5** (fit and evaluate Gaussian Naive Bayes; state the conditional independence assumption; name NB variants for non-Gaussian data) and contributes to **G7** (across all three classifiers, distinguish raw value / probability / class label).

## §A Imports

One imports cell at the top. Everything this notebook touches is loaded here.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

RANDOM_STATE = 42
np.set_printoptions(suppress=True, precision=3)
sns.set_theme(style="whitegrid")
```

## §B Bayes' theorem in one cell

> ⏱ **Pause-point — skippable.** If you have already met Bayes' theorem in a statistics class, you can skim the formulas below and jump to §C. If you have not, read them carefully — the rest of the notebook is one application of these four lines.

The whole algorithm comes from one equation — Bayes' rule:

$$
P(C \mid \mathbf{x}) \;=\; \frac{P(\mathbf{x} \mid C) \cdot P(C)}{P(\mathbf{x})}
$$

Reading the four pieces of that equation:

- $P(C \mid \mathbf{x})$ on the **left** — the **posterior**: the class probability given the features. *The thing we want.*
- $P(\mathbf{x} \mid C)$ — the **likelihood** (top-left factor): how likely the features are under each class.
- $P(C)$ — the **prior** (top-right factor): how common the class is, ignoring features.
- $P(\mathbf{x})$ — the **evidence** (denominator): the unconditional density of the features.

The evidence is the same for every class, so when we compare classes we can drop it and just compare numerators:

$$
P(C \mid \mathbf{x}) \;\propto\; P(\mathbf{x} \mid C) \cdot P(C)
$$

The three quantities, by name:

- $P(C = c)$ — the **prior**. How common is each class in the training data? In iris, with a stratified 70/30 split, each class is ≈ 33 %.
- $P(\mathbf{x} \mid C = c)$ — the **likelihood**. Given that a flower really *is* a setosa, how likely are these specific measurements?
- $P(C = c \mid \mathbf{x})$ — the **posterior**. The thing we actually want — given the measurements we observed, which class is most probable?

**The classification rule** is then: *pick the class with the largest posterior*:

$$
\hat{c} \;=\; \arg\max_{c} \; P(C = c \mid \mathbf{x})
       \;=\; \arg\max_{c} \; P(\mathbf{x} \mid C = c) \cdot P(C = c)
$$

### The **"naive"** assumption — the simplifying lie

Computing $P(\mathbf{x} \mid C)$ for a 4-dimensional feature vector would in general require a full joint distribution over all four iris measurements *within each class* — 4-dimensional histograms, with not enough data to fill them.

Naive Bayes sidesteps this by assuming the features are **independent given the class**:

$$
P(\mathbf{x} \mid C) \;=\; \prod_{i=1}^{p} P(x_i \mid C)
$$

This is the lie. On iris, petal length and petal width are clearly correlated *within* the versicolor class (long petals are also wide). The assumption that they are independent given the class is false. Yet NB often classifies correctly anyway — see §G for why.

Combining the independence assumption with the classification rule from §B gives the **Naive Bayes decision rule**:

$$
\hat{c} \;=\; \arg\max_{c} \; P(C = c) \prod_{i=1}^{p} P(x_i \mid C = c)
$$

**What this equation says in plain words.** For each possible class, compute two things and multiply them: *(1)* how common that class is in the training data, and *(2)* how typical each observed feature value looks under that class — multiplied together across all the features. The class with the biggest resulting score is the prediction.

In practice sklearn computes this in **log-space** to avoid underflow when multiplying many small probabilities:

$$
\hat{c} \;=\; \arg\max_{c} \; \log P(C = c) \;+\; \sum_{i=1}^{p} \log P(x_i \mid C = c)
$$

### The **"Gaussian"** assumption

Each per-feature likelihood $P(x_i \mid C = c)$ is modelled as a Normal distribution:

$$
P(x_i \mid C = c) \;=\; \frac{1}{\sqrt{2\pi\,\sigma_{i,c}^{2}}}
\,\exp\!\left( -\frac{(x_i - \mu_{i,c})^{2}}{2\,\sigma_{i,c}^{2}} \right)
$$

So `GaussianNB` only has to learn **one mean $\mu_{i,c}$ and one variance $\sigma_{i,c}^{2}$ per (feature, class) pair**. For iris with 4 features and 3 classes, that is **12 means + 12 variances + 3 priors = 27 numbers** — the entire model.

## §C Load iris, split, and eyeball the feature shapes

Iris has 150 rows, 4 continuous features, 3 balanced classes. We use a stratified 70/30 train/test split so each class is represented in both sides.

```python
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target 
class_names = iris.target_names

print("X shape:", X.shape)
print("Classes:", list(class_names))
print("Class counts:")
print(pd.Series(y).map(dict(enumerate(class_names))).value_counts())
```

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE,
)
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
```

### Eye-check: are the within-class distributions roughly Gaussian?

`GaussianNB` assumes each feature is Normal **within each class**. We do not need a perfect bell curve — just no wildly skewed or bimodal shapes inside a class. A quick histogram-by-class plot is the right check before fitting.

```python
train_df = X_train.copy()
train_df["species"] = pd.Categorical.from_codes(y_train, class_names)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, feature in zip(axes.ravel(), X.columns):
    for species in class_names:
        subset = train_df.loc[train_df["species"] == species, feature]
        ax.hist(subset, bins=12, alpha=0.55, label=species)
    ax.set_title(feature)
    ax.set_xlabel(feature)
    ax.set_ylabel("count")
axes[0, 0].legend()
plt.suptitle("Within-class distributions on the training split", y=1.02)
plt.tight_layout()
plt.show()
```

The within-class shapes are approximately bell-curve-like for sepal width, and somewhat skewed but unimodal for the petal features. There is no class where a feature is bimodal or wildly non-Normal. Good enough for `GaussianNB` to have a real chance — and a fair illustration of how forgiving the Gaussian assumption is in practice.

## §D Fit `GaussianNB` and inspect what it learned

Three lines of code. The whole model.

```python
clf = GaussianNB()
clf.fit(X_train, y_train)
clf
```

### What the fitted estimator actually stores

After `.fit()`, three arrays carry everything `GaussianNB` will ever know about the data.

```python
print("class_prior_  (prior P(class)):", clf.class_prior_)
print("                shape:", clf.class_prior_.shape)
```

```python
theta_df = pd.DataFrame(clf.theta_, index=class_names, columns=X.columns)
print("theta_  (per-class feature MEANS):")
theta_df.round(3)
```

```python
var_df = pd.DataFrame(clf.var_, index=class_names, columns=X.columns)
print("var_  (per-class feature VARIANCES):")
var_df.round(4)
```

That is the whole model:

- `class_prior_` — shape `(n_classes,)`. The fraction of training rows in each class. With a stratified split it is ~`[1/3, 1/3, 1/3]`.
- `theta_` — shape `(n_classes, n_features)`. The per-class feature means. Setosa's petals are clearly the smallest, virginica's the largest — the means make that visible.
- `var_` — shape `(n_classes, n_features)`. The per-class feature variances. Used (with `theta_`) to evaluate the Normal density at any new observation.

Contrast this with what a fitted logistic regression stores: coefficients + an intercept (`n_classes × n_features + n_classes` numbers, conceptually similar count) — but logistic regression *learned* those coefficients via an iterative optimiser. `GaussianNB` did not optimise anything. It read off sample means and sample variances. That is why it is `O(n)` to fit.

## §E Predict on the test set and evaluate

The standard supervised-classification workflow from `lec_10a`: predict, accuracy, confusion matrix, classification report.

```python
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
```

```python
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(cmap="rocket_r", ax=ax, colorbar=False)
ax.set_title("GaussianNB on iris - test confusion matrix")
plt.show()
```

```python
print(classification_report(y_test, y_pred, target_names=class_names))
```

### Walk one observation end-to-end (G7 — raw value / probability / class)

`GaussianNB` does **not** expose `decision_function`; the "raw" quantity is the **joint log-likelihood** `log P(features, class) = log P(features | class) + log P(class)`. The class with the largest joint log-likelihood is the predicted class. The probabilities you get from `predict_proba` are these joint log-likelihoods, exponentiated and normalised to sum to 1.

We pick the first test observation, compute its joint log-likelihood under each class **by hand**, and check that the argmax matches `.predict`.

```python
# Pick the first test row.
x0 = X_test.iloc[0]
true_label = class_names[y_test.iloc[0]]
print("Features of the chosen test row:")
print(x0)
print("True label:", true_label)
```

```python
# For each class, compute the joint log-likelihood by hand.
#     log P(features | class) = sum_i  log Normal_pdf(x_i ; mean_{class,i}, var_{class,i})
#     log P(features, class)  = log P(class) + log P(features | class)
# Use the log of the Gaussian density:  -0.5 * log(2 pi var) - 0.5 * (x - mean)^2 / var

log_priors = np.log(clf.class_prior_)
log_joint = np.zeros(len(class_names))

for k in range(len(class_names)):
    means_k = clf.theta_[k]
    vars_k = clf.var_[k]
    log_likelihood_per_feature = (
        -0.5 * np.log(2 * np.pi * vars_k)
        - 0.5 * (x0.values - means_k) ** 2 / vars_k
    )
    log_joint[k] = log_priors[k] + log_likelihood_per_feature.sum()

hand_result = pd.DataFrame({
    "class": class_names,
    "log_prior": log_priors.round(3),
    "log_joint (raw score)": log_joint.round(3),
})
hand_result
```

```python
# The class with the largest log_joint is the prediction.
hand_predicted = class_names[int(np.argmax(log_joint))]
sklearn_predicted = class_names[clf.predict(x0.to_frame().T)[0]]

print("By-hand argmax of log_joint :", hand_predicted)
print("sklearn .predict()          :", sklearn_predicted)
print("Match:", hand_predicted == sklearn_predicted)
```

```python
# And the probabilities (predict_proba) are the joint likelihoods, normalised.
# Exponentiate after subtracting the max for numerical stability.
shifted = log_joint - log_joint.max()
probs_by_hand = np.exp(shifted) / np.exp(shifted).sum()
probs_sklearn = clf.predict_proba(x0.to_frame().T)[0]

comparison = pd.DataFrame({
    "class": class_names,
    "by hand": probs_by_hand.round(4),
    "sklearn predict_proba": probs_sklearn.round(4),
})
comparison
```

The three quantities for `GaussianNB`, named explicitly (this is the G7 vocabulary):

- **Raw value** — the **joint log-likelihood** `log P(features, class)`. One number per class. The class with the largest value wins. (There is no `decision_function` on `GaussianNB`; this is the equivalent.)
- **Probability** — `predict_proba`. The joint log-likelihoods, exponentiated and normalised across classes to sum to 1.
- **Class label** — `predict`. The argmax of the joint log-likelihoods. No threshold needed because multi-class NB picks the largest of three probabilities, not whether one of them exceeds 0.5.

## §F The single hyperparameter: `var_smoothing`

`GaussianNB` has essentially one knob: `var_smoothing`. It adds a small fraction of the largest feature variance in the training set to every variance — a numerical-stability term that prevents division by near-zero variance. Default is `1e-9`.

We vary it on a log scale on the same fixed split.

```python
var_smoothing_grid = [1e-12, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1]
rows = []
for vs in var_smoothing_grid:
    clf_vs = GaussianNB(var_smoothing=vs)
    clf_vs.fit(X_train, y_train)
    rows.append({
        "var_smoothing": vs,
        "train accuracy": accuracy_score(y_train, clf_vs.predict(X_train)),
        "test accuracy":  accuracy_score(y_test,  clf_vs.predict(X_test)),
    })

vs_results = pd.DataFrame(rows)
vs_results
```

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(vs_results["var_smoothing"], vs_results["train accuracy"], "o-", label="train")
ax.semilogx(vs_results["var_smoothing"], vs_results["test accuracy"], "s-", label="test")
ax.set_xlabel("var_smoothing (log scale)")
ax.set_ylabel("accuracy")
ax.set_title("GaussianNB accuracy vs var_smoothing on iris")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()
```

Reading the curve:

- `var_smoothing` adds a small fraction of the **largest feature variance** in the training set to every variance term in the likelihood. This prevents a zero-or-near-zero variance from making the likelihood blow up.
- On well-behaved data like iris, accuracy is essentially flat across many orders of magnitude. The default `1e-9` is fine.
- On data with **near-zero-variance features** — e.g., a feature that is almost-constant within one class — raising `var_smoothing` stabilises the fit. It is the one situation where this knob actually changes anything.
- Very large values (e.g. `1e-1`) effectively swamp the per-class variance with a global term, blurring the class-specific Gaussian shapes; accuracy starts to drop.

## §G Why does Naive Bayes work despite the independence assumption being wrong?

On iris, petal length and petal width are clearly correlated within each class — yet the test accuracy in §E is high. Why?

> ⏱ **Pause-point — skippable.** This section is intuition, not workflow. If you only need to *use* `GaussianNB`, you have everything from §A–§F. Read on if you want to know *why* a model built on a false assumption still classifies well — and why you should not trust its probabilities.

### Three connected reasons

- **NB does not need calibrated probabilities — it only needs the right argmax.** The classifier picks the class with the highest posterior. As long as the true class has the highest score, it does not matter that the absolute numbers are wrong.
- **Correlated features double-count evidence, but the argmax often lands on the same class.** When petal length is large *and* petal width is large, NB treats them as two independent pieces of evidence for virginica — and adds their log-likelihoods. That double-counts, but it double-counts in the *same direction* for the same class, so the winner usually does not change.
- **NB tends to give overconfident probabilities — close to 0 or 1.** Because correlated features pile evidence onto one class, the posterior collapses toward a single class faster than it should. The `predict_proba` outputs are uncalibrated: a "98 % virginica" from `GaussianNB` does not mean "98 % of the time, observations with this profile turn out to be virginica". Treat NB probabilities as a *ranking* across classes, not as a meaningful frequency. If your downstream code (cost-sensitive decisions, threshold tuning, calibration plots) needs *calibrated* probabilities, NB is usually the wrong choice; see §I.

## §H NB variants for non-Gaussian data

`GaussianNB` is one of several Naive Bayes variants in `sklearn.naive_bayes`. Each variant assumes a different per-feature distribution; you pick the one that matches your feature types. **Theory only** — we do not fit these here.

- **`MultinomialNB`** — feature counts. Each feature `x_i` is a count (e.g. how many times each word appears in a document). The canonical text-classification NB; probably the most-used NB in practice. *Use when*: bag-of-words / TF features, click counts, any non-negative integer feature.
  *See:* scikit-learn user guide on Naive Bayes → "Multinomial Naive Bayes".

- **`BernoulliNB`** — binary features. Each `x_i` is 0/1 (e.g. "does this word appear in the document?"). Differs from `MultinomialNB` by **explicitly modelling the absence of a feature** as evidence — counts vs presence/absence are different signals.
  *See:* scikit-learn user guide → "Bernoulli Naive Bayes".

- **`CategoricalNB`** — small-cardinality categorical features. Each `x_i` takes one of a few discrete values (e.g. `red / green / blue`). Internally fits a separate multinomial per feature.
  *See:* scikit-learn user guide → "Categorical Naive Bayes".

- **`ComplementNB`** — `MultinomialNB`'s sibling, designed for imbalanced text classification. Often beats `MultinomialNB` on real-world (skewed) document corpora.
  *See:* scikit-learn user guide → "Complement Naive Bayes".

**The decision rule in one line:** continuous features → `GaussianNB`; word counts → `MultinomialNB`; binary indicators → `BernoulliNB`; small categorical alphabets → `CategoricalNB`; imbalanced text → `ComplementNB`.

## §I When to reach for Naive Bayes (and when not to)

### Reach for NB when

- **Text classification baseline** — sentiment, spam, topic. `MultinomialNB` on a TF or bag-of-words representation is the standard "number to beat" before going to a transformer.
- **Real-time scoring** — predictions must come back in microseconds. NB's `O(n_features)` per prediction is hard to beat.
- **High-dimensional sparse data** — when there are 50 000 features and 5 000 rows, fitting logistic regression is slow and unstable; NB shrugs and finishes in a fraction of a second.
- **A first-pass baseline on any classification problem** — three lines of code, useful upper bound on how hard the problem is.

### Avoid NB when

- **You need calibrated probabilities.** NB outputs are systematically miscalibrated (§G). If downstream code reads `predict_proba` as a real frequency — for cost-sensitive thresholding, expected-value calculations, or probability plots — use a model whose probabilities are calibrated (logistic regression after `CalibratedClassifierCV`, gradient boosting with `isotonic` calibration, etc.).
- **Feature interactions are the whole story.** NB assumes independence given the class. If the signal is in how features combine (XOR-type structure), NB will miss it entirely. Use a tree-based model or a kernel SVM.
- **Continuous features are wildly non-Gaussian.** `GaussianNB` on heavily skewed or bimodal features will struggle. Either transform the features (`log`, quantile) or switch model family.

## §J Recap

- **`GaussianNB` is the fastest classifier in scikit-learn.** It stores three small arrays — class priors, per-class feature means, per-class feature variances — and predicts via Bayes' rule with the conditional-independence + Gaussian assumptions.
- **The "naive" assumption is almost never true, and the model often works anyway.** What NB needs is the right argmax across classes, not calibrated probabilities — so correlated features double-counting evidence usually does not change the predicted class.
- **Pick the NB variant by feature type.** Continuous → `GaussianNB`; counts → `MultinomialNB`; binary → `BernoulliNB`; small categorical → `CategoricalNB`; imbalanced text → `ComplementNB`. The single hyperparameter `var_smoothing` is a numerical-stability knob, not a quality knob.

**Coming next** — `lec_12f_SVM.ipynb`. Support Vector Machines on a 2D iris slice with decision boundaries, the linear-vs-RBF kernel contrast, and the `C` / `gamma` hyperparameters tuned one-at-a-time. SVM is the geometrically-clean cousin of the probabilistic classifiers in this lecture — same problem, very different machinery.

## Further reading

- [scikit-learn user guide §1.9 — Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) — covers GaussianNB, MultinomialNB, BernoulliNB, and CategoricalNB with worked examples.
- [`GaussianNB` API](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) — the `var_smoothing` hyperparameter walked in §F.
- [Wikipedia — Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) — the derivation of the conditional-independence approximation discussed in §B and §G.
- [Hand & Yu, 2001 — *Idiot's Bayes — not so stupid after all?* (Int. Stat. Review)](https://doi.org/10.1111/j.1751-5823.2001.tb00465.x) — the canonical reference for *why* Naive Bayes works despite the independence assumption being wrong, discussed in §G.

<!-- source: lectures_07_13_pandas_plots_scikit/lecture_10_knn_train_test_split/reading_material/lec_10e_knn_vs_other_classifiers.ipynb @ 0cc874704aaa -->

# Lecture 10e. KNN vs. other classifiers (reference notebook)

This notebook is **reference-only** — no worked code, no exercises depend on it. KNN is the focus of this lecture (`lec_10a`–`lec_10c` and the optional `lec_10d`). The alternatives below are pointers for further reading, useful when KNN's distance-based vote (see `lec_10b_knn_assumptions_caveats.ipynb`) is the wrong fit for your data: when the decision boundary is roughly linear, when interpretability matters, when feature scales differ wildly, or when prediction-time cost on a large training set becomes a problem.

A **full worked side-by-side comparison** of these classifiers — fitted models, coefficient and importance tables, support-vector counts, decision-boundary plots — lives in Lecture 13 (`lec_13` — model pipelines, hyper-parameter grids, model selection). This notebook stays at the conceptual level so you build the decision reflex before the implementation details.

For a visual gallery of how every scikit-learn classifier behaves on the same toy datasets, see scikit-learn's [plot_classifier_comparison example](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html).

## Logistic regression — the linear baseline

Despite the name, **logistic regression is a classifier**, not a regression algorithm. It fits a linear combination of the features and pushes the result through a sigmoid (or softmax for multi-class) to produce class probabilities.

**When to prefer over KNN:**

- The decision boundary in your data is roughly linear — a straight line (or hyperplane) cleanly separates the classes.
- You need **calibrated probabilities** for downstream decisions (e.g., "only flag transactions with > 0.9 fraud probability"), not just hard labels.
- You need **interpretable coefficients** — one per feature, with a sign that tells you which class the feature pushes toward.
- The dataset is large enough that KNN's O(N) prediction-time cost is a problem. Logistic regression's prediction cost is O(d) — proportional to the number of features, independent of training-set size.

**Watch out for:** logistic regression assumes a linear decision boundary; it under-fits curved class shapes where KNN or SVM with an RBF kernel would do better. Always scale features before fitting for numerical stability of the optimiser and for coefficient comparability.

→ [scikit-learn LogisticRegression docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

## Decision trees (and the leap to random forests)

A **decision tree** classifies by asking a sequence of simple yes/no questions about one feature at a time (`petal_length > 2.45?` → go left, else right) until it reaches a leaf with a class label. The result is an interpretable, axis-aligned partition of feature space.

**When to prefer over KNN:**

- You have **mixed feature types** — categorical, ordinal, and numeric in the same dataset. Trees handle them naturally; KNN needs careful encoding and scaling.
- You need **interpretability** at the level of "here are the rules the model follows" — a small tree can be drawn and read by a non-technical stakeholder.
- Feature scales differ wildly (rooms 1–8 vs. listing price 50,000–1,500,000). Trees split on raw values; KNN's distance metric would be dominated by the largest-range feature.

**Watch out for:** a single deep tree over-fits aggressively. The standard fix is **Random Forest** — many trees fit on bootstrap samples of the data with votes averaged. Random Forest trades the interpretability of one tree for a substantial accuracy boost and is the standard tabular-data baseline in 2026. Gradient-boosted trees (XGBoost, LightGBM, CatBoost) take this further and are routinely the best non-deep-learning method on tabular data.

→ [scikit-learn DecisionTreeClassifier docs](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)   ·   [RandomForestClassifier docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## Support Vector Machines (RBF kernel)

A **Support Vector Machine** finds the *widest possible margin* between classes. With a linear kernel it learns a hyperplane; with a non-linear kernel — the most common is the RBF / Gaussian kernel — it implicitly maps the data into a higher-dimensional space where a hyperplane separates classes that were tangled in the original space.

**When to prefer over KNN:**

- The decision boundary is non-linear but smooth — the RBF kernel curves around clusters naturally where KNN would draw jagged piecewise-linear boundaries.
- You have a moderate sample size (a few hundred to a few thousand) and high feature dimensionality. SVM with the kernel trick is one of the few classical algorithms that copes well in this regime.
- You need a small prediction-time memory footprint — SVM only stores the *support vectors* (points on or inside the margin), typically a small fraction of the training set.

**Watch out for:** SVM training cost is between O(N²) and O(N³), so it is impractical above ~100,000 training samples — switch to a linear classifier or a gradient-boosted tree instead. The two key hyperparameters (`C` for regularisation strength, `gamma` for kernel width) interact and usually require a grid search. Always scale features (the RBF kernel is distance-based — same logic as for KNN).

→ [scikit-learn SVC docs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)   ·   [SVM user guide](https://scikit-learn.org/stable/modules/svm.html)

## Further reading on supervised classification more broadly

- [Supervised learning algorithms in scikit-learn](https://scikit-learn.org/stable/supervised_learning.html) — the full menu: linear models, trees, ensembles, SVMs, naive Bayes, neural nets, and more.
- [Comparing different classifiers on toy datasets](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) — the canonical "which classifier picks up which shape" gallery.
- [Choosing the right estimator (scikit-learn flowchart)](https://scikit-learn.org/stable/machine_learning_map.html) — a decision tree (of the human kind) for picking an estimator given your problem.
- [Common pitfalls in scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html) — covers data leakage, scaling discipline, and improperly randomised splits.

**Where this course goes deeper:**

- **Lecture 11** — linear regression, the three-way train/validation/test split, bias-variance tradeoff.
- **Lecture 12** — logistic regression with worked examples, Naive Bayes, and Support Vector Machines.
- **Lecture 13** — model pipelines, hyper-parameter grids, model selection and stacking. The full worked side-by-side comparison of KNN / logistic regression / decision trees / SVM.

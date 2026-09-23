<!-- source: lectures_07_13_pandas_plots_scikit/lecture_09_clustering_deploy_hf_app/reading_material/lec_09f_other_clustering_algos.ipynb @ 0cc874704aaa -->

# Lecture 09f. Other clustering algorithms (reference notebook)

This notebook is **reference-only** — no worked code, no exercises depend on it. KMeans is the focus of this lecture (`lec_09a`–`lec_09c` and the optional `lec_09d`/`lec_09e`). The alternatives below are pointers for further reading, useful when KMeans's spherical-cluster assumption (see `lec_09b_kmeans_assumptions_caveats.ipynb`) is the wrong fit for your data.

For a side-by-side visual comparison of how every clustering algorithm in scikit-learn behaves on the same toy datasets, see scikit-learn's [plot_cluster_comparison example](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html).

## DBSCAN — density-based clustering

**Density-Based Spatial Clustering of Applications with Noise.** Groups points that are densely packed together (many neighbours within a radius `eps`) and labels sparse points as noise (cluster `-1`).

**When to prefer over KMeans:**

- The clusters in your data are non-spherical (rings, moons, elongated blobs).
- You don't know K up front and don't want to choose it — DBSCAN figures the number of clusters out from the density structure.
- You want explicit "doesn't belong anywhere" handling for outliers.

**Watch out for:** DBSCAN is sensitive to its two parameters (`eps`, `min_samples`) and behaves badly when clusters have very different densities. The newer **HDBSCAN** (built into scikit-learn since v1.3) handles varying density well and removes the `eps` choice.

→ [scikit-learn DBSCAN docs](https://scikit-learn.org/stable/modules/clustering.html#dbscan)   ·   [HDBSCAN docs](https://scikit-learn.org/stable/modules/clustering.html#hdbscan)

## Hierarchical / agglomerative clustering

Builds a tree (a *dendrogram*) of nested clusters: every point starts as its own cluster, and the closest pair is merged repeatedly until only one cluster remains. You then "cut" the tree at the level you want to read off K clusters.

**When to prefer over KMeans:**

- You want to *inspect* the clustering structure at multiple Ks at once — the dendrogram shows the whole hierarchy in one plot.
- The data is small enough that the O(n²) memory cost is OK (typically ≤ a few thousand points).
- The cluster sizes are very uneven, or the natural shape is hierarchical (taxonomies, phylogenies, document trees, organisational charts).

**Watch out for:** compute cost grows quickly with `n`. For large data, consider `MiniBatchKMeans` or `BIRCH` instead. The choice of *linkage* (`ward`, `complete`, `average`, `single`) materially changes the result.

→ [scikit-learn AgglomerativeClustering docs](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)

## Gaussian Mixture Models (GMM)

Fits a mixture of Gaussian distributions to the data. Each cluster is a Gaussian with its own mean, covariance, and prior weight. Returns **soft assignments** (probability of belonging to each cluster) instead of hard ones.

**When to prefer over KMeans:**

- The clusters are elliptical or have different orientations / scales — KMeans assumes isotropic spherical clusters; GMM does not (with `covariance_type='full'`).
- You want a per-point uncertainty estimate (`predict_proba`) instead of a single hard label.
- You want the model to be probabilistic so you can plug log-likelihoods into a downstream Bayesian or anomaly-detection pipeline.

**Watch out for:** the EM optimiser can land in a local minimum and is sensitive to the chosen `covariance_type`. Run with multiple `random_state` values and keep the best log-likelihood. Use BIC / AIC to pick the number of components.

→ [scikit-learn GaussianMixture docs](https://scikit-learn.org/stable/modules/mixture.html)

## Further reading on unsupervised learning more broadly

- [Unsupervised learning algorithms in scikit-learn](https://scikit-learn.org/stable/unsupervised_learning.html) — clustering is one slice; manifold learning, density estimation, and matrix decomposition are others.
- [Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) — when you have ground-truth labels (rare in practice), the supervised metrics. When you don't, the silhouette score (used in `lec_09a`) and the Calinski-Harabasz index are the standard tools.
- [Comparing different clustering algorithms on toy datasets](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html) — the canonical "which algorithm picks up which shape" gallery.

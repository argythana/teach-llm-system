<!-- source: lectures_07_13_pandas_plots_scikit/lecture_09_clustering_deploy_hf_app/practice_exercises/lec_09_exercises_solutions.ipynb @ 0cc874704aaa -->

# Lecture 09 — Practice Exercises (Solutions)

Runnable solutions for `lec_09_exercises.ipynb`. Every required exercise (0.1, 1.1–1.2, 2.1–2.2, 3.1, 4.1) and the stretch exercise (5.1) has a working solution. Re-run top-to-bottom under `course_venv`.

The Gradio solution (4.1) builds the `gr.Interface` object but stops short of `.launch()` so the whole notebook executes cleanly under `jupyter nbconvert --execute`. To see the app, uncomment the `.launch()` line at the bottom of cell 4.1's solution.

## Setup

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons, make_blobs

import gradio as gr
```

```python
DATA_PATH = '../reading_material/mall_customers.csv'
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["CustomerID"])
print(f"Mall customers dataset: {df.shape[0]} rows, {df.shape[1]} columns")
df.head(3)
```

---
### 0.1 [G1] — Unsupervised vs supervised

> **Supervised** learning trains on labelled data — every row has an input `X` and a known target `y`, and the model learns the `X → y` mapping (e.g. *spam vs not-spam classification*). **Unsupervised** learning has no `y`; the model finds structure in `X` alone (e.g. *customer segmentation with KMeans*).

---
### 1.1 [G2 + G3] — KMeans with two features and `K = 5`

```python
X_2feat = df[["Annual_Income_(k$)", "Spending_Score"]]

model = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = model.fit_predict(X_2feat)
df["cluster"] = labels

print("cluster_centers_:")
print(model.cluster_centers_)
print()
print("labels_ (first 20):", model.labels_[:20])
print(f"inertia_: {model.inertia_:.2f}")
```

```python
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df, x="Annual_Income_(k$)", y="Spending_Score",
                hue="cluster", palette="deep", ax=ax)
ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
           marker="*", s=300, c="black", label="centroids")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set_title("KMeans, 2 features, K=5");
```

---
### 1.2 [G6] — Predict the cluster of new customers

```python
new_customers = pd.DataFrame(
    [[20, 80], [75, 50], [110, 15]],
    columns=["Annual_Income_(k$)", "Spending_Score"],
    index=["A", "B", "C"],
)
new_customers["predicted_cluster"] = model.predict(new_customers)
new_customers
```

**Why is this called *predict* even though there is no `y` target?** scikit-learn uses one consistent estimator API across supervised and unsupervised models. For KMeans, `predict(X_new)` does not predict a target value — it assigns each new point to the nearest *already-trained* centroid. The name is API-level, not semantic.

---
### 2.1 [G4] — Elbow + silhouette side-by-side

```python
ks_elbow = list(range(1, 11))
wcss = []
for k in ks_elbow:
    m = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_2feat)
    wcss.append(m.inertia_)

ks_sil = list(range(2, 11))
sil = []
for k in ks_sil:
    m = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbls = m.fit_predict(X_2feat)
    sil.append(silhouette_score(X_2feat, lbls))

best_k_sil = ks_sil[int(np.argmax(sil))]
best_k_elbow = 5  # by inspection

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ks_elbow, wcss, marker="o")
axes[0].axvline(best_k_elbow, ls="--", c="red", label=f"elbow ≈ K={best_k_elbow}")
axes[0].set_xlabel("K"); axes[0].set_ylabel("WCSS (inertia_)")
axes[0].set_title("Elbow method"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ks_sil, sil, marker="o", color="green")
axes[1].axvline(best_k_sil, ls="--", c="red", label=f"best K={best_k_sil}")
axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette score"); axes[1].legend(); axes[1].grid(alpha=0.3)

print(f"Elbow suggests K={best_k_elbow}; silhouette suggests K={best_k_sil} (score={max(sil):.3f}).")
```

**Do the two metrics agree?** They typically agree within ±1 on a clean dataset like this — elbow says ≈5, silhouette says ≈5 or 6. **When they disagree, trust silhouette for the *quality* of the clustering** (it has a clear interpretation: how well-separated are the clusters?), but trust the elbow for the *budget* (K beyond the elbow buys you very little extra explanatory power per cluster). The maintainer's call usually goes to whichever K gives clusters that are easiest to *label* downstream.

---
### 2.2 [G5] — Four features, profile, label

```python
X_4feat = df[["Genre", "Age", "Annual_Income_(k$)", "Spending_Score"]].copy()
X_4feat["Gender"] = (X_4feat["Genre"] == "Male").astype(int)
X_4feat = X_4feat.drop(columns="Genre")

model_4 = KMeans(n_clusters=6, random_state=42, n_init=10)
df["cluster"] = model_4.fit_predict(X_4feat)

profile = df.assign(Gender=X_4feat["Gender"]).groupby("cluster")[
    ["Age", "Annual_Income_(k$)", "Spending_Score", "Gender"]].mean().round(1)
print("Per-cluster mean of every numeric feature:")
profile
```

```python
# Human-readable labels — the maintainer's intuitive read of the per-cluster profile
cluster_labels = {
    0: "wise_constrained",
    1: "start_earning_living_it",
    2: "have_not_spend_not",
    3: "save_or_spend_elsewhere",
    4: "young_cautious",
    5: "young_yolos",
}
df["cluster_label"] = df["cluster"].map(cluster_labels)

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="Annual_Income_(k$)", y="Spending_Score",
                hue="cluster_label", palette="deep", ax=ax)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Cluster label")
ax.set_title("Six-cluster KMeans on four features, with human labels");
```

---
### 3.1 [G7] — Two failure modes combined

**Part A — Non-convex shapes (`make_moons`):**

```python
X_moons, y_moons = make_moons(n_samples=400, noise=0.06, random_state=42)
labels_kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_moons)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="coolwarm", s=15)
axes[0].set_title("True moons")
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_kmeans_moons, cmap="coolwarm", s=15)
axes[1].set_title("KMeans (K=2) — cuts across the moons")
for ax in axes: ax.set_aspect("equal"); ax.grid(alpha=0.3)
```

**Part B — Anisotropic blobs:**

```python
X_blobs, y_blobs = make_blobs(n_samples=600, centers=3, cluster_std=0.6, random_state=42)
T = np.array([[0.6, -0.6], [-0.4, 0.8]])
X_aniso = X_blobs @ T

km_aniso = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_aniso = km_aniso.fit_predict(X_aniso)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_blobs, cmap="tab10", s=15)
axes[0].set_title("True labels (anisotropic blobs)")
axes[1].scatter(X_aniso[:, 0], X_aniso[:, 1], c=labels_aniso, cmap="tab10", s=15)
axes[1].scatter(km_aniso.cluster_centers_[:, 0], km_aniso.cluster_centers_[:, 1],
                marker="*", s=200, c="black")
axes[1].set_title("KMeans (K=3) — slices along the wrong axis")
for ax in axes: ax.grid(alpha=0.3)
```

**Why does KMeans fail on each?**

- **Moons (non-convex):** KMeans assigns each point to its nearest centroid by Euclidean distance, producing convex (Voronoi-like) regions. The two moons are non-convex, so any straight-edged partition has to cut across one or both — there is no centroid placement that recovers the moon shape.
- **Anisotropic blobs:** KMeans implicitly assumes isotropic (equally scaled in every direction) spherical clusters. After the diagonal stretch, points within one true cluster are now further from each other than from the *nearest centroid of a different cluster* — so KMeans groups by Euclidean distance, slicing the long axis instead of along it. Gaussian Mixture Models with `covariance_type='full'` handle this case correctly (see `lec_09f`).

---
### 4.1 [G8] — Wrap a trained KMeans in a Gradio app (local)

The interface is built but `launch()` is **not** called here, so this notebook executes cleanly under `nbconvert`. Uncomment the last line and run interactively to see the app.

```python
def predict_segment(genre: str, age: int, income: float, spending: int) -> str:
    """Score one customer with the four-feature KMeans model trained in 2.2."""
    gender = 1 if genre == "Male" else 0
    row = pd.DataFrame(
        [[age, income, spending, gender]],
        columns=["Age", "Annual_Income_(k$)", "Spending_Score", "Gender"],
    )
    cluster_id = int(model_4.predict(row)[0])
    label = cluster_labels.get(cluster_id, f"cluster_{cluster_id}")
    return f"Cluster {cluster_id} — {label}"


demo = gr.Interface(
    fn=predict_segment,
    inputs=[
        gr.Radio(choices=["Male", "Female"], value="Female", label="Genre"),
        gr.Slider(15, 80, value=30, step=1, label="Age"),
        gr.Slider(10, 150, value=60, step=1, label="Annual income (k$)"),
        gr.Slider(0, 100, value=50, step=1, label="Spending score (0–100)"),
    ],
    outputs="text",
    title="Mall customer segment predictor",
    description="Predicts the customer segment using the 6-cluster KMeans from Exercise 2.2.",
)
print("Gradio Interface built — run `demo.launch()` to start the local app.")
# demo.launch()  # uncomment to launch interactively
```

---
### 5.1 [O1, *stretch*] — `init='random'` vs `init='k-means++'`

```python
for n_init in (1, 10):
    rand = KMeans(n_clusters=5, init="random",     n_init=n_init, random_state=0).fit(X_2feat)
    pp   = KMeans(n_clusters=5, init="k-means++",  n_init=n_init, random_state=0).fit(X_2feat)
    gap  = rand.inertia_ - pp.inertia_
    print(f"n_init={n_init:2d}  random inertia={rand.inertia_:>9.1f}  k-means++ inertia={pp.inertia_:>9.1f}  gap={gap:>+8.1f}")
```

**What this teaches:** with `n_init=1`, the random initialisation often produces a worse local optimum (higher inertia) than `k-means++`, which seeds centroids spread out across the data. With `n_init=10`, scikit-learn runs the algorithm ten times from different seeds and keeps the best result, so the gap shrinks. This is exactly why both `n_init` and `'k-means++'` exist as defaults.

---
### 5.2 [O2, *stretch*] — Lloyd's algorithm in two phases

**Phase 1 (assignment):** for each data point, compute its distance to every centroid and assign the point to the cluster of the *nearest* centroid — this partitions the data into K Voronoi-like regions.

**Phase 2 (update):** for each cluster, compute the *mean* of all the points currently assigned to it and move that cluster's centroid to the mean — this is the only step where the centroids actually move.

**Stopping condition:** the loop exits when no point changed its cluster assignment in the last full pass (or equivalently, when the centroids' positions stop moving by more than the `tol` parameter; or when `max_iter` is reached, whichever comes first). KMeans is guaranteed to converge, but only to a *local* minimum of the WCSS — which is why `n_init > 1` exists, to retry from multiple random starts and keep the best.

---
### 5.3 [O3, *stretch*] — When to prefer DBSCAN, hierarchical, GMM over KMeans

- **DBSCAN** — prefer when clusters are **non-convex** (rings, moons, elongated blobs) or when you want explicit "doesn't belong anywhere" handling for outliers (DBSCAN labels noise points as `-1`); also when you do *not* know K up front, since DBSCAN discovers it from the data's density structure.
- **Hierarchical / agglomerative** — prefer when you want to *inspect* the clustering structure at multiple Ks at once via the dendrogram (small-to-medium datasets, ≤ a few thousand points), or when the data has a natural hierarchical structure (taxonomies, phylogenies, document trees).
- **Gaussian Mixture Models** — prefer when clusters are **elliptical or differently scaled** (KMeans assumes isotropic spherical clusters; GMM with `covariance_type='full'` does not), or when you need **soft assignments** (a per-point probability of belonging to each cluster, via `predict_proba`) instead of hard labels for a downstream probabilistic pipeline.

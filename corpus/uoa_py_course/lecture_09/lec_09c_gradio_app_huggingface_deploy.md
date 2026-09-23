<!-- source: lectures_07_13_pandas_plots_scikit/lecture_09_clustering_deploy_hf_app/reading_material/lec_09c_gradio_app_huggingface_deploy.ipynb @ 0cc874704aaa -->

# Lecture 09c. Deploy a Machine Learning App with Gradio on Hugging Face Spaces

In this notebook we build a simple **Gradio web application** that uses our KMeans clustering model, and deploy it on **Hugging Face Spaces** so that anyone with a browser can use it.  

[My clustering Application example](https://huggingface.co/spaces/thanarg/mall-customers-clustering)

---

## What is Hugging Face Spaces?

[Hugging Face](https://huggingface.co/) is a platform for machine learning and AI. It hosts models, datasets, and **Spaces** — free hosted web apps.

A **Space** is a Git repository on Hugging Face that automatically builds and deploys your app. You can use three frameworks:
- **Gradio** (easiest for ML demos)
- Streamlit
- Docker

**Why use Spaces?**
- Free hosting (CPU tier is free forever)
- Share your ML work with a URL
- No server configuration needed
- Automatic HTTPS, auto-restart on push

---

## Step 1: Create a Hugging Face Account

1. Go to [huggingface.co/join](https://huggingface.co/join)
2. Create a free account (email + password or GitHub login)
3. Verify your email

---

## Step 2: Create an Access Token (API Key)

A **token** (API key) lets you push code to your Spaces from your terminal **without entering your password each time**.

**API** stands for **Application Programming Interface** — it is a set of rules that allows different software programs to communicate with each other. An API *key* (or *token*) is a secret string that identifies you when your code talks to an external service.

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Give it a name (e.g., `my-spaces-token`)
4. Select **"Write"** permission
5. Click **"Generate"** and **copy it immediately** (you won't see it again)

**Security:**  
- Never commit your token to a Git repository  
- Never share it in a notebook  
- Store it as an environment variable or use `hf auth login`

```bash
# In your terminal, login once:
pip install huggingface_hub
hf auth login
# Paste your token when prompted. It's stored locally.
```

> **Note:** The old command `huggingface-cli login` is deprecated. The current CLI uses `hf auth login`.

---

## Step 3 (only for Options B/C): Create a Space on the Hugging Face Website

> **Note:** If you use our recommended deployment method (Option A in Step 6), you can skip this step — the Space is created programmatically from Python.

If you prefer to deploy via Git (Option B) or the web interface (Option C), you must first create the Space manually:

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose a name for your Space (e.g., `mall-customers-clustering`)
3. Select **Gradio** as the SDK
4. Choose **Public** (so others can see it)
5. Click **"Create Space"**

Your Space is now a Git repository at:  
`https://huggingface.co/spaces/YOUR_USERNAME/mall-customers-clustering`

## What is Gradio?

[Gradio](https://www.gradio.app/) is a Python library that creates web interfaces for ML models in a few lines of code. Please revise the previous' lecture dedicated notebook.

- No HTML/CSS/JavaScript needed
- Supports inputs: sliders, dropdowns, text, images, audio
- Supports outputs: text, plots, tables, images
- Runs locally for testing, deploys to Hugging Face for sharing

**With activated virtual environemt**

```bash
pip install gradio
```

```python
# Install gradio if not already installed
# !pip install gradio
```

## Step 4: Build the Gradio App Locally

We will create a clustering app with **two tabs**.

**Tab 1 — Predict Cluster.** For a single customer:
1. Loads the mall customers data
2. Encodes the Genre column as a numeric Gender feature (same as in lecture 09a, section 5.1)
3. Trains KMeans with **four features** (Age, Annual Income, Spending Score, Gender)
4. Shows an interactive **3D scatter plot** (Income × Spending Score × Age) colored by cluster
5. Lets the user choose the number of clusters
6. **Predicts the cluster for a new customer** based on their Age, Income, Spending Score, and Gender

**Tab 2 — Cluster Profiles.** Compare what makes each segment different at the chosen K:
- Bar chart of **cluster sizes** (how many customers fall in each segment)
- Grouped bar chart of **mean Age, Income and Spending per cluster**
- Stacked bar chart of the **gender breakdown per cluster**
- A detailed table with counts, means and **% Male** per cluster

Below we first build a simplified single-screen version with `gr.Interface` to introduce the basics, and then build the final two-tab app with `gr.Blocks` that gets deployed in **Step 5**.

```python
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import gradio as gr
```

```python
# Load and prepare data
df = pd.read_csv("mall_customers.csv")
df = df.drop("CustomerID", axis=1)

# Encode Genre as numeric Gender column (Male=1, Female=0)
df["Gender"] = pd.get_dummies(df["Genre"], drop_first=True, prefix="Genre")

df.head(3)
```

```python
def cluster_and_plot(n_clusters, new_age, new_income, new_spending, new_gender):
    """
    Train KMeans with n_clusters using 4 features, show 3D scatter plot,
    and predict cluster for a new customer.
    """
    # Four features for clustering (same as lecture 09a, section 4.4)
    X = df[["Age", "Annual_Income_(k$)", "Spending_Score", "Gender"]]

    # Train KMeans
    model = KMeans(n_clusters=int(n_clusters), random_state=0, n_init=10)
    df["cluster"] = model.fit_predict(X).astype(str)

    # Predict cluster for new customer
    new_customer = [[new_age, new_income, new_spending, new_gender]]
    predicted_cluster = str(model.predict(new_customer)[0])

    # 3D scatter plot: Income x Spending Score x Age, colored by cluster
    fig = px.scatter_3d(
        df,
        x="Annual_Income_(k$)",
        y="Spending_Score",
        z="Age",
        color="cluster",
        height=650,
        width=800,
        title=f"KMeans with {int(n_clusters)} clusters (4 features, 3D view)",
        size="Annual_Income_(k$)",
    )

    # Add new customer as a star marker (use add_scatter3d for 3D plots)
    fig.add_scatter3d(
        x=[new_income],
        y=[new_spending],
        z=[new_age],
        mode="markers",
        marker=dict(size=5, symbol="diamond", color="black", line=dict(width=2, color="white")),
        name=f"New customer → cluster {predicted_cluster}",
    )

    fig.update_layout(margin=dict(t=50, l=40, r=40, b=40))

    # Summary table: mean of each feature per cluster
    summary = (
        df.groupby("cluster")[["Age", "Annual_Income_(k$)", "Spending_Score"]]
        .mean()
        .round(1)
        .reset_index()
    )
    summary_text = summary.to_string(index=False)

    result_text = f"New customer (Age={new_age}, Income={new_income}k$, Spending={new_spending}, Gender={'M' if new_gender==1 else 'F'}) → **Cluster {predicted_cluster}**"

    return fig, summary_text, result_text
```

```python
# Test the function with 6 clusters (as in clustering lecture)
# Test with a new customer: Age 30, Income 80k, Spending 60, Male (Gender=1)
fig, summary, result = cluster_and_plot(6, 30, 80, 60, 1)
print(result)
print("\n" + summary)
fig.show()
```

### Now wrap it in a Gradio interface

```python
demo = gr.Interface(
    fn=cluster_and_plot,
    inputs=[
        gr.Slider(minimum=2, maximum=10, step=1, value=6, label="Number of Clusters (K)"),
        gr.Slider(minimum=18, maximum=70, step=1, value=30, label="New Customer Age"),
        gr.Slider(minimum=15, maximum=137, step=1, value=50, label="New Customer Annual Income (k$)"),
        gr.Slider(minimum=1, maximum=100, step=1, value=50, label="New Customer Spending Score"),
        gr.Radio([0, 1], value=1, label="New Customer Gender (0=Female, 1=Male)"),
    ],
    outputs=[
        gr.Plot(label="3D Cluster Visualization"),
        gr.Textbox(label="Cluster Summary (mean per cluster)", lines=8),
        gr.Textbox(label="Prediction Result"),
    ],
    title="Mall Customers KMeans Clustering (4 features, 3D)",
    description="Choose the number of clusters and enter a new customer's data. "
                "The model uses Age, Annual Income, Spending Score, and Gender. "
                "The 3D plot shows Income × Spending × Age, with your customer marked as a star.",
)

# Uncomment below to launch locally for testing (opens at http://localhost:7860)
demo.launch()
```

### Cluster Interpretations (with 6 clusters)

When using the default of **6 clusters**, each cluster has an intuitive meaning based on customer characteristics (analyzed in lecture 09a):

| Cluster | Label | Profile |
|---------|-------|---------|
| **0** | Wise & Constrained | Age 40+, average income, low-to-moderate spending. Budget-conscious older customers. |
| **1** | Start Earning, Living It | Age 27-40, high-to-very-high income, very high spending. Young professionals enjoying newfound earnings. |
| **2** | Have Not, Spend Not | Low income, low-to-moderate spending. Budget-constrained or infrequent shoppers. |
| **3** | Save or Spend Elsewhere | Mid-to-high income, low spending. Either savers or customers who shop elsewhere. |
| **4** | Young & Cautious | Young (under 35), low income, low spending. Budget-conscious young customers. |
| **5** | Young YOLOs | Young (under 35), low income, high-to-very-high spending. Young uninhibited spenders. |

**Note:** The interpretation section in the app displays these labels only when using 6 clusters (the default). If you change K to another value, the algorithm retrains but interpretations are not available for other K values.

**To test locally**, uncomment `demo.launch()` above and run the cell.  
Open your browser at `http://localhost:7860`.  
Press `Ctrl+C` in the terminal or restart the kernel to stop it.

## Step 5: Prepare Files for Hugging Face Deployment

A Gradio Space needs **at minimum** these files:

| File | Purpose |
|------|--------|
| `app.py` | Your Gradio app script |
| `requirements.txt` | Python packages to install |
| `mall_customers.csv` | Your data file |

Let's create them.

```python
# This cell creates the app.py file that will be deployed.

app_code = r'''
from pathlib import Path
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import gradio as gr

# Load data relative to the script location (works locally and on HF Spaces)
SCRIPT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(SCRIPT_DIR / "mall_customers.csv")
df = df.drop("CustomerID", axis=1)

# Encode Genre as numeric Gender column (Male=1, Female=0)
df["Gender"] = pd.get_dummies(df["Genre"], drop_first=True, prefix="Genre")

# Cluster interpretations (from lecture 09a analysis with 6 clusters)
CLUSTER_LABELS = {
    0: "wise_constrained",
    1: "start_earning_living_it",
    2: "have_not_spend_not",
    3: "save_or_spend_elsewhere",
    4: "young_cautious",
    5: "young_yolos",
}

CLUSTER_DESCRIPTIONS = {
    0: "**Wise & Constrained**: Age 40+, average income, low-to-moderate spending. Spend cautiously within their budget.",
    1: "**Start Earning, Living It**: Age 27-40, high-to-very-high income, very high spending. Young professionals enjoying newfound earnings.",
    2: "**Have Not, Spend Not**: Low income, low-to-moderate spending. Budget-constrained customers or those who don't shop much.",
    3: "**Save or Spend Elsewhere**: Mid-to-high income, low spending. Either savers or customers who shop elsewhere.",
    4: "**Young & Cautious**: Young (under 35), low income, low spending. Budget-conscious young customers.",
    5: "**Young YOLOs**: Young (under 35), low income, high-to-very-high spending. Young, uninhibited spenders (You Only Live Once).",
}

CLUSTER_LEGEND_DESCRIPTIONS = {
    0: "0: wise_constrained",
    1: "1: start_earning_living_it",
    2: "2: have_not_spend_not",
    3: "3: save_or_spend_elsewhere",
    4: "4: young_cautious",
    5: "5: young_yolos",
}

LEGEND_COLOR_MAP = {
    CLUSTER_LEGEND_DESCRIPTIONS[0]: "blue",
    CLUSTER_LEGEND_DESCRIPTIONS[1]: "purple",
    CLUSTER_LEGEND_DESCRIPTIONS[2]: "green",
    CLUSTER_LEGEND_DESCRIPTIONS[3]: "yellow",
    CLUSTER_LEGEND_DESCRIPTIONS[4]: "red",
    CLUSTER_LEGEND_DESCRIPTIONS[5]: "brown",
}

FEATURES_4D = ["Age", "Annual_Income_(k$)", "Spending_Score", "Gender"]


def _fit_kmeans(n_clusters):
    """Fit KMeans on the 4 features and return (model, labels)."""
    X = df[FEATURES_4D]
    model = KMeans(n_clusters=int(n_clusters), random_state=0, n_init=10)
    labels = model.fit_predict(X)
    return model, labels


def _summary_dataframe(labels):
    """Per-cluster size + mean of Age, Income, Spending. Returned as a DataFrame
    so Gradio renders it as a real table (not a misaligned text block)."""
    tmp = df.copy()
    tmp["cluster"] = labels
    return (
        tmp.groupby("cluster")
        .agg(
            Count=("Age", "size"),
            Mean_Age=("Age", "mean"),
            Mean_Income_k=("Annual_Income_(k$)", "mean"),
            Mean_Spending=("Spending_Score", "mean"),
        )
        .round(1)
        .reset_index()
        .rename(columns={
            "cluster": "Cluster",
            "Mean_Age": "Mean Age",
            "Mean_Income_k": "Mean Income (k$)",
            "Mean_Spending": "Mean Spending",
        })
    )


def cluster_and_plot(n_clusters, new_age, new_income, new_spending, new_gender):
    """Train KMeans, show 3D scatter with the new customer marked, return summary
    and predicted-cluster interpretation."""
    model, labels = _fit_kmeans(n_clusters)
    df["cluster"] = labels.astype(str)

    new_customer = [[new_age, new_income, new_spending, new_gender]]
    predicted_cluster = str(model.predict(new_customer)[0])

    df_plot = df.copy()
    color_col = "cluster"
    color_map = None
    legend_kwargs = {
        "x": 0.01,
        "y": 1.08,
        "xanchor": "left",
        "yanchor": "top",
        "bgcolor": "rgba(255,255,255,0.88)",
        "bordercolor": "lightgray",
        "borderwidth": 1,
    }

    # Use full descriptive legend labels for the lecture's default K=6 case.
    if int(n_clusters) == 6:
        df_plot["cluster_description"] = (
            df_plot["cluster"].astype(int).map(CLUSTER_LEGEND_DESCRIPTIONS)
        )
        color_col = "cluster_description"
        color_map = LEGEND_COLOR_MAP
        legend_kwargs["title"] = "Cluster"

    fig = px.scatter_3d(
        df_plot,
        x="Annual_Income_(k$)",
        y="Spending_Score",
        z="Age",
        color=color_col,
        height=650,
        title=f"KMeans with {int(n_clusters)} clusters (4 features, 3D view)",
        size="Annual_Income_(k$)",
        size_max=12,
        color_discrete_map=color_map,
    )

    fig.add_scatter3d(
        x=[new_income],
        y=[new_spending],
        z=[new_age],
        mode="markers",
        marker=dict(size=5, symbol="diamond", color="black",
                    line=dict(width=2, color="white")),
        name=f"New customer → cluster {predicted_cluster}",
    )

    fig.update_layout(
        autosize=True,
        margin=dict(t=50, l=40, r=40, b=40),
        showlegend=True,
        legend=legend_kwargs,
    )
    fig.update_traces(showlegend=True)

    summary = _summary_dataframe(labels)

    result_text = (f"New customer (Age={new_age}, Income={new_income}k$, Spending={new_spending}, "
                   f"Gender={'M' if new_gender==1 else 'F'}) → Cluster {predicted_cluster}")

    if int(n_clusters) == 6 and int(predicted_cluster) in CLUSTER_DESCRIPTIONS:
        interpretation = CLUSTER_DESCRIPTIONS[int(predicted_cluster)]
    else:
        interpretation = "Cluster interpretation available only for 6 clusters (default)."

    return fig, summary, result_text, interpretation


def cluster_profiles(n_clusters):
    """Build comparative profile plots for the chosen K:
    - cluster sizes bar chart
    - mean Age / Income / Spending per cluster (grouped bar)
    - gender breakdown per cluster (stacked bar)
    - detailed table with counts, means, and % male per cluster
    """
    _, labels = _fit_kmeans(n_clusters)
    profile_df = df.copy()
    profile_df["cluster"] = labels.astype(str)

    # 1. Cluster sizes
    sizes = (
        profile_df["cluster"].value_counts().sort_index().reset_index()
    )
    sizes.columns = ["cluster", "count"]
    fig_sizes = px.bar(
        sizes, x="cluster", y="count",
        title="Cluster sizes — how many customers in each segment",
        text="count", color="cluster", height=320,
    )
    fig_sizes.update_traces(textposition="outside")
    fig_sizes.update_layout(showlegend=False, yaxis_title="Number of customers")

    # 2. Mean of each feature per cluster (grouped bar)
    means_long = (
        profile_df.groupby("cluster")[["Age", "Annual_Income_(k$)", "Spending_Score"]]
        .mean()
        .round(1)
        .reset_index()
        .melt(id_vars="cluster", var_name="Feature", value_name="Mean")
    )
    fig_means = px.bar(
        means_long, x="cluster", y="Mean", color="Feature",
        barmode="group", text="Mean", height=420,
        title="Mean values per cluster — what makes each segment different",
    )
    fig_means.update_traces(textposition="outside")
    fig_means.update_layout(yaxis_title="Mean value")

    # 3. Gender breakdown per cluster (stacked bar)
    gender_counts = (
        profile_df.groupby(["cluster", "Genre"]).size().reset_index(name="count")
    )
    fig_gender = px.bar(
        gender_counts, x="cluster", y="count", color="Genre",
        barmode="stack", height=320,
        title="Gender breakdown per cluster",
        color_discrete_map={"Female": "#e377c2", "Male": "#1f77b4"},
    )
    fig_gender.update_layout(yaxis_title="Number of customers")

    # 4. Detailed per-cluster stats table
    table = (
        profile_df.groupby("cluster")
        .agg(
            Count=("Age", "size"),
            Mean_Age=("Age", "mean"),
            Mean_Income_k=("Annual_Income_(k$)", "mean"),
            Mean_Spending=("Spending_Score", "mean"),
            Pct_Male=("Gender", lambda s: round(100 * s.mean(), 1)),
        )
        .round(1)
        .reset_index()
        .rename(columns={
            "cluster": "Cluster",
            "Mean_Age": "Mean Age",
            "Mean_Income_k": "Mean Income (k$)",
            "Mean_Spending": "Mean Spending",
            "Pct_Male": "% Male",
        })
    )

    return fig_sizes, fig_means, fig_gender, table


with gr.Blocks() as demo:
    gr.Markdown(
        "## Mall Customers KMeans Clustering (4 features, 3D)\n"
        "The model uses Age, Annual Income, Spending Score, and Gender. "
        "Use the **Predict Cluster** tab to place a new customer in a segment, "
        "or the **Cluster Profiles** tab to compare what makes each segment different."
    )

    with gr.Tabs():
        # ------------------------------------------------------------------
        # Tab 1: Predict Cluster (existing functionality)
        # ------------------------------------------------------------------
        with gr.Tab("Predict Cluster"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Customer Profile")
                    n_clusters = gr.Slider(minimum=2, maximum=10, step=1, value=6,
                                           label="Number of Clusters (K)")
                    new_age = gr.Slider(minimum=18, maximum=70, step=1, value=30,
                                       label="Age")
                    new_income = gr.Slider(minimum=15, maximum=137, step=1, value=50,
                                          label="Annual Income (k$)")
                    new_spending = gr.Slider(minimum=1, maximum=100, step=1, value=50,
                                            label="Spending Score")
                    new_gender = gr.Radio([0, 1], value=1, label="Gender (0=Female, 1=Male)")

                    submit_btn = gr.Button("Analyze Customer", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### Results")
                    plot_output = gr.Plot(label="3D Cluster Visualization")
                    summary_output = gr.Dataframe(
                        label="Cluster Summary (size + mean per cluster)",
                        interactive=False,
                        wrap=True,
                    )
                    result_output = gr.Textbox(label="Prediction Result")
                    interpretation_output = gr.Textbox(label="Cluster Interpretation")

            predict_inputs = [n_clusters, new_age, new_income, new_spending, new_gender]
            predict_outputs = [plot_output, summary_output, result_output, interpretation_output]

            submit_btn.click(fn=cluster_and_plot, inputs=predict_inputs, outputs=predict_outputs)
            for input_elem in predict_inputs:
                input_elem.change(fn=cluster_and_plot, inputs=predict_inputs, outputs=predict_outputs)

        # ------------------------------------------------------------------
        # Tab 2: Cluster Profiles (new)
        # ------------------------------------------------------------------
        with gr.Tab("Cluster Profiles"):
            gr.Markdown(
                "### Compare what makes each cluster different\n"
                "Pick a number of clusters and see the size, mean characteristics, "
                "and gender breakdown of every segment side by side."
            )
            profiles_k = gr.Slider(minimum=2, maximum=10, step=1, value=6,
                                   label="Number of Clusters (K)")
            profiles_btn = gr.Button("Compute Profiles", variant="primary")
            with gr.Row():
                profile_sizes_plot = gr.Plot(label="Cluster sizes")
                profile_gender_plot = gr.Plot(label="Gender breakdown")
            profile_means_plot = gr.Plot(label="Mean values per cluster")
            profile_table = gr.Dataframe(
                label="Detailed per-cluster statistics",
                interactive=False, wrap=True,
            )

            profile_outputs = [profile_sizes_plot, profile_means_plot,
                               profile_gender_plot, profile_table]
            profiles_btn.click(fn=cluster_profiles, inputs=profiles_k, outputs=profile_outputs)

demo.launch()
'''

# print(app_code)
```

```python
# Save app.py to disk (for local testing before deploying)
with open("app.py", "w") as f:
    f.write(app_code.strip())

print("app.py saved.")
```

```python
# Save requirements.txt (no version pins — HF Spaces manages its own versions)
requirements = """pandas
plotly
scikit-learn
gradio
"""

with open("requirements.txt", "w") as f:
    f.write(requirements.strip())
print("requirements.txt saved.")
```

### Test locally before deploying

In your terminal:
```bash
python app.py
```

This opens the app at `http://localhost:7860`. Verify it works.

## Step 6: Deploy to Hugging Face Spaces

There are three ways to deploy. We recommend **Option A** (Python via the `huggingface_hub` library) because you can do everything — create the Space and upload files — directly from this notebook, without needing to know Git or leave your coding environment.

- **Option A** — Python `huggingface_hub` (recommended). Create the Space and push files in one cell.
- **Option B** — `git clone` + `git push`. Requires you to create the Space on the website first.
- **Option C** — Drag-and-drop upload through the Hugging Face web interface. Also requires you to create the Space first.

### Option A: Using the `huggingface_hub` Python library (recommended)

[`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/) is the official Python library for talking to the Hugging Face Hub (Spaces, models, datasets) from your own code. The cell below uses it to do two things in one go:

1. **`api.create_repo(...)`** — creates the Space if it doesn't exist yet. The `exist_ok=True` flag means the call is safe to re-run: if the Space is already there, the call simply continues instead of raising an error.
2. **`api.upload_file(...)`** — uploads each of the three required files (`app.py`, `requirements.txt`, `mall_customers.csv`). Each call **overwrites** the file on the Space if it already exists, so you can re-run this cell any time you change `app.py` to push an update — no `git add` / `git commit` / `git push` needed.

**Prerequisites (one-time setup):**

- `pip install huggingface_hub` in your virtual environment.
- In a terminal, run `hf auth login` and paste your access token (created in **Step 2**). The token is then cached on your machine, so future deployments require no further authentication.

**Why this is the best option for this course:**

- No Git knowledge required.
- The Space is created programmatically — no need to click through the website first (so **Step 3** is skipped).
- Everything is reproducible and editable from the notebook.
- Updates are a single cell re-run away.

Set your username in the cell below and run it.

```python
# Option A (recommended): Deploy directly from Python using huggingface_hub
# First: pip install huggingface_hub
# Then: hf auth login (do this once in terminal)

from huggingface_hub import HfApi

# Replace with your actual username
USERNAME = "thanarg"  # <-- CHANGE THIS
SPACE_NAME = "mall-customers-clustering"

api = HfApi()

# Step 1: Create the Space programmatically (no need to create it on the website first!)
api.create_repo(
    repo_id=f"{USERNAME}/{SPACE_NAME}",
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,  # won't fail if it already exists
)
print(f"Space created: https://huggingface.co/spaces/{USERNAME}/{SPACE_NAME}")

# Step 2: Upload all three files
for filename in ["app.py", "requirements.txt", "mall_customers.csv"]:
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=f"{USERNAME}/{SPACE_NAME}",
        repo_type="space",
    )
    print(f"Uploaded {filename}")

print(f"\nDone! Visit: https://huggingface.co/spaces/{USERNAME}/{SPACE_NAME}")
```

---

### Option B: Using Git

> **Prerequisite:** You must first create the Space on the Hugging Face website (**Step 3** above) before you can clone it.

```bash
# Clone your Space repository (must already exist on huggingface.co)
git clone https://huggingface.co/spaces/YOUR_USERNAME/mall-customers-clustering
cd mall-customers-clustering

# Copy your files into the repo
cp /path/to/app.py .
cp /path/to/requirements.txt .
cp /path/to/mall_customers.csv .

# Push to Hugging Face
git add .
git commit -m "Initial deployment of clustering app"
git push
```

After pushing, Hugging Face will automatically build and deploy your app.  
Visit: `https://huggingface.co/spaces/YOUR_USERNAME/mall-customers-clustering`

---

### Option C: Upload via the Hugging Face web interface

> **Prerequisite:** You must first create the Space on the Hugging Face website (**Step 3** above).

1. Go to your Space page on huggingface.co
2. Click the **"Files"** tab
3. Click **"Add file" → "Upload files"**
4. Upload `app.py`, `requirements.txt`, and `mall_customers.csv`
5. The Space rebuilds automatically

## Step 7: Verify your deployment

After pushing, wait 1-2 minutes for the Space to build.  
Visit your Space URL. You should see the Gradio interface with the sliders.

If the build fails:
- Check the **"Logs"** tab on your Space page for errors
- Common issues: missing package in `requirements.txt`, wrong file path for the CSV

---

## Summary

| Step | What |
|------|------|
| 1 | Create Hugging Face account |
| 2 | Generate an access token (API key) |
| 3 | Create a new Space (Gradio SDK) |
| 4 | Write your `app.py` with Gradio |
| 5 | Create `requirements.txt` + include data file |
| 6 | Push files to your Space (git, web UI, or Python) |
| 7 | Visit your live app URL |

**Your app is now live on the internet, for free.**

---

## Extra: Useful links

- [Gradio documentation](https://www.gradio.app/docs/)
- [Hugging Face Spaces documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio + Hugging Face quickstart](https://www.gradio.app/guides/sharing-your-app#hosting-on-hf-spaces)
- [Example Spaces](https://huggingface.co/spaces)

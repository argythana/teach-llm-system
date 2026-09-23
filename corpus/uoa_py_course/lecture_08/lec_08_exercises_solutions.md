<!-- source: lectures_07_13_pandas_plots_scikit/lecture_08_EDA_plots/practice_exercises/lec_08_exercises_solutions.ipynb @ 0cc874704aaa -->

# Lecture 08 — Gradio Exercises: Solutions

This notebook contains **verified solutions** for every exercise.  
Each exercise progressively enhances the base Gradio Heart Disease Explorer app.

### Base app recap
The starting `app_gradio.py` has:
- A dropdown to select a numeric feature
- A radio button to choose chart type (Histogram / Box Plot / Violin)
- A single Plot output

The exercises below add new features one at a time.

---
## Exercise 1 — Add descriptive statistics as a second output

Add a text output that shows `df[feature].describe()` alongside the plot.

```python
from pathlib import Path
import gradio as gr
import pandas as pd
import plotly.express as px

SCRIPT_DIR = Path(".").resolve()
DATA_FILE = SCRIPT_DIR / "predict_heart_disease_train.csv"
df = pd.read_csv(DATA_FILE).drop(columns=["id"])
numeric_cols = df.select_dtypes(include="number").columns.tolist()
color_map = {"Presence": "#e74c3c", "Absence": "#2ecc71"}


def explore_with_stats(feature, chart_type):
    if chart_type == "Histogram":
        fig = px.histogram(df, x=feature, color="Heart Disease",
                           color_discrete_map=color_map, barmode="overlay", opacity=0.7)
    elif chart_type == "Box Plot":
        fig = px.box(df, x="Heart Disease", y=feature, color="Heart Disease",
                     color_discrete_map=color_map)
    else:
        fig = px.violin(df, x="Heart Disease", y=feature, color="Heart Disease",
                        color_discrete_map=color_map, box=True)

    stats = df[feature].describe().to_string()
    return fig, stats


demo = gr.Interface(
    fn=explore_with_stats,
    inputs=[
        gr.Dropdown(choices=numeric_cols, value="Age", label="Select Feature"),
        gr.Radio(choices=["Histogram", "Box Plot", "Violin"], value="Histogram", label="Chart Type"),
    ],
    outputs=[gr.Plot(label="Visualization"), gr.Textbox(label="Descriptive Statistics")],
    title="Heart Disease Data Explorer",
    description="Select a feature and chart type to explore the dataset.",
)

demo.launch()
```

---
## Exercise 2 — Add a correlation heatmap tab

Use `gr.TabbedInterface` to add a second tab that shows a correlation heatmap.

```python
import gradio as gr
import pandas as pd
import plotly.express as px
from pathlib import Path

SCRIPT_DIR = Path(".").resolve()
DATA_FILE = SCRIPT_DIR / "predict_heart_disease_train.csv"
df = pd.read_csv(DATA_FILE).drop(columns=["id"])
numeric_cols = df.select_dtypes(include="number").columns.tolist()
color_map = {"Presence": "#e74c3c", "Absence": "#2ecc71"}


# --- Tab 1: Feature explorer (same as base app) ---
def explore_data(feature, chart_type):
    if chart_type == "Histogram":
        fig = px.histogram(df, x=feature, color="Heart Disease",
                           color_discrete_map=color_map, barmode="overlay", opacity=0.7)
    elif chart_type == "Box Plot":
        fig = px.box(df, x="Heart Disease", y=feature, color="Heart Disease",
                     color_discrete_map=color_map)
    else:
        fig = px.violin(df, x="Heart Disease", y=feature, color="Heart Disease",
                        color_discrete_map=color_map, box=True)
    return fig

tab1 = gr.Interface(
    fn=explore_data,
    inputs=[
        gr.Dropdown(choices=numeric_cols, value="Age", label="Select Feature"),
        gr.Radio(choices=["Histogram", "Box Plot", "Violin"], value="Histogram", label="Chart Type"),
    ],
    outputs=gr.Plot(label="Visualization"),
)


# --- Tab 2: Correlation heatmap ---
def show_heatmap():
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Feature Correlation Heatmap")
    return fig

tab2 = gr.Interface(
    fn=show_heatmap,
    inputs=[],
    outputs=gr.Plot(label="Correlation Heatmap"),
)


demo = gr.TabbedInterface([tab1, tab2], ["Feature Explorer", "Correlations"])
demo.launch()
```

---
## Exercise 3 — Add a 3D scatter plot tab

Add a third tab where the user picks X, Y, and Z axes for a `px.scatter_3d()` plot.

```python
import gradio as gr
import pandas as pd
import plotly.express as px
from pathlib import Path

SCRIPT_DIR = Path(".").resolve()
DATA_FILE = SCRIPT_DIR / "predict_heart_disease_train.csv"
df = pd.read_csv(DATA_FILE).drop(columns=["id"])
numeric_cols = df.select_dtypes(include="number").columns.tolist()
color_map = {"Presence": "#e74c3c", "Absence": "#2ecc71"}


# Tab 1: Feature explorer
def explore_data(feature, chart_type):
    if chart_type == "Histogram":
        fig = px.histogram(df, x=feature, color="Heart Disease",
                           color_discrete_map=color_map, barmode="overlay", opacity=0.7)
    elif chart_type == "Box Plot":
        fig = px.box(df, x="Heart Disease", y=feature, color="Heart Disease",
                     color_discrete_map=color_map)
    else:
        fig = px.violin(df, x="Heart Disease", y=feature, color="Heart Disease",
                        color_discrete_map=color_map, box=True)
    return fig

tab1 = gr.Interface(
    fn=explore_data,
    inputs=[
        gr.Dropdown(choices=numeric_cols, value="Age", label="Select Feature"),
        gr.Radio(choices=["Histogram", "Box Plot", "Violin"], value="Histogram", label="Chart Type"),
    ],
    outputs=gr.Plot(label="Visualization"),
)


# Tab 2: Correlation heatmap
def show_heatmap():
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Feature Correlation Heatmap")
    return fig

tab2 = gr.Interface(fn=show_heatmap, inputs=[], outputs=gr.Plot(label="Correlation Heatmap"))


# Tab 3: 3D scatter
def scatter_3d(x_col, y_col, z_col):
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                        color="Heart Disease", color_discrete_map=color_map,
                        opacity=0.5, title=f"{x_col} vs {y_col} vs {z_col}")
    return fig

tab3 = gr.Interface(
    fn=scatter_3d,
    inputs=[
        gr.Dropdown(choices=numeric_cols, value="Age", label="X axis"),
        gr.Dropdown(choices=numeric_cols, value="Max HR", label="Y axis"),
        gr.Dropdown(choices=numeric_cols, value="ST depression", label="Z axis"),
    ],
    outputs=gr.Plot(label="3D Scatter"),
)


demo = gr.TabbedInterface([tab1, tab2, tab3], ["Feature Explorer", "Correlations", "3D Scatter"])
demo.launch()
```

---
## Exercise 4 — Add age-group prevalence chart tab

Add a fourth tab that shows heart disease prevalence (%) by age group,
with a slider to control the bin width.

```python
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

SCRIPT_DIR = Path(".").resolve()
DATA_FILE = SCRIPT_DIR / "predict_heart_disease_train.csv"
df = pd.read_csv(DATA_FILE).drop(columns=["id"])
numeric_cols = df.select_dtypes(include="number").columns.tolist()
color_map = {"Presence": "#e74c3c", "Absence": "#2ecc71"}


# Tab 1: Feature explorer
def explore_data(feature, chart_type):
    if chart_type == "Histogram":
        fig = px.histogram(df, x=feature, color="Heart Disease",
                           color_discrete_map=color_map, barmode="overlay", opacity=0.7)
    elif chart_type == "Box Plot":
        fig = px.box(df, x="Heart Disease", y=feature, color="Heart Disease",
                     color_discrete_map=color_map)
    else:
        fig = px.violin(df, x="Heart Disease", y=feature, color="Heart Disease",
                        color_discrete_map=color_map, box=True)
    return fig

tab1 = gr.Interface(
    fn=explore_data,
    inputs=[
        gr.Dropdown(choices=numeric_cols, value="Age", label="Select Feature"),
        gr.Radio(choices=["Histogram", "Box Plot", "Violin"], value="Histogram", label="Chart Type"),
    ],
    outputs=gr.Plot(label="Visualization"),
)


# Tab 2: Correlation heatmap
def show_heatmap():
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Feature Correlation Heatmap")
    return fig

tab2 = gr.Interface(fn=show_heatmap, inputs=[], outputs=gr.Plot(label="Correlation Heatmap"))


# Tab 3: 3D scatter
def scatter_3d(x_col, y_col, z_col):
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                        color="Heart Disease", color_discrete_map=color_map,
                        opacity=0.5, title=f"{x_col} vs {y_col} vs {z_col}")
    return fig

tab3 = gr.Interface(
    fn=scatter_3d,
    inputs=[
        gr.Dropdown(choices=numeric_cols, value="Age", label="X axis"),
        gr.Dropdown(choices=numeric_cols, value="Max HR", label="Y axis"),
        gr.Dropdown(choices=numeric_cols, value="ST depression", label="Z axis"),
    ],
    outputs=gr.Plot(label="3D Scatter"),
)


# Tab 4: Prevalence by age group
def prevalence_by_age(bin_width):
    bin_width = int(bin_width)
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    bins = list(range(age_min, age_max + bin_width, bin_width))
    tmp = df.copy()
    tmp["Age_group"] = pd.cut(tmp["Age"], bins=bins)
    prev = (tmp.groupby("Age_group", observed=True)["Heart Disease"]
            .apply(lambda s: (s == "Presence").mean() * 100))
    prev_df = prev.reset_index()
    prev_df.columns = ["Age Group", "Prevalence (%)"]
    prev_df["Age Group"] = prev_df["Age Group"].astype(str)
    fig = px.bar(prev_df, x="Age Group", y="Prevalence (%)",
                 title="Heart Disease Prevalence by Age Group",
                 color="Prevalence (%)", color_continuous_scale="OrRd")
    return fig

tab4 = gr.Interface(
    fn=prevalence_by_age,
    inputs=gr.Slider(minimum=5, maximum=20, step=5, value=10, label="Age bin width"),
    outputs=gr.Plot(label="Prevalence Chart"),
)


demo = gr.TabbedInterface(
    [tab1, tab2, tab3, tab4],
    ["Feature Explorer", "Correlations", "3D Scatter", "Prevalence"]
)
demo.launch()
```

---
## Exercise 5 — Add faceted histograms by Chest Pain Type

Add a fifth tab that shows a faceted histogram of any selected feature,
with one subplot per `Chest pain type`.

```python
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

SCRIPT_DIR = Path(".").resolve()
DATA_FILE = SCRIPT_DIR / "predict_heart_disease_train.csv"
df = pd.read_csv(DATA_FILE).drop(columns=["id"])
numeric_cols = df.select_dtypes(include="number").columns.tolist()
color_map = {"Presence": "#e74c3c", "Absence": "#2ecc71"}


# Tab 1: Feature explorer with stats
def explore_with_stats(feature, chart_type):
    if chart_type == "Histogram":
        fig = px.histogram(df, x=feature, color="Heart Disease",
                           color_discrete_map=color_map, barmode="overlay", opacity=0.7)
    elif chart_type == "Box Plot":
        fig = px.box(df, x="Heart Disease", y=feature, color="Heart Disease",
                     color_discrete_map=color_map)
    else:
        fig = px.violin(df, x="Heart Disease", y=feature, color="Heart Disease",
                        color_discrete_map=color_map, box=True)
    stats = df[feature].describe().to_string()
    return fig, stats

tab1 = gr.Interface(
    fn=explore_with_stats,
    inputs=[
        gr.Dropdown(choices=numeric_cols, value="Age", label="Select Feature"),
        gr.Radio(choices=["Histogram", "Box Plot", "Violin"], value="Histogram", label="Chart Type"),
    ],
    outputs=[gr.Plot(label="Visualization"), gr.Textbox(label="Descriptive Statistics")],
)


# Tab 2: Correlation heatmap
def show_heatmap():
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Feature Correlation Heatmap")
    return fig

tab2 = gr.Interface(fn=show_heatmap, inputs=[], outputs=gr.Plot(label="Correlation Heatmap"))


# Tab 3: 3D scatter
def scatter_3d(x_col, y_col, z_col):
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                        color="Heart Disease", color_discrete_map=color_map,
                        opacity=0.5, title=f"{x_col} vs {y_col} vs {z_col}")
    return fig

tab3 = gr.Interface(
    fn=scatter_3d,
    inputs=[
        gr.Dropdown(choices=numeric_cols, value="Age", label="X axis"),
        gr.Dropdown(choices=numeric_cols, value="Max HR", label="Y axis"),
        gr.Dropdown(choices=numeric_cols, value="ST depression", label="Z axis"),
    ],
    outputs=gr.Plot(label="3D Scatter"),
)


# Tab 4: Prevalence by age group
def prevalence_by_age(bin_width):
    bin_width = int(bin_width)
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    bins = list(range(age_min, age_max + bin_width, bin_width))
    tmp = df.copy()
    tmp["Age_group"] = pd.cut(tmp["Age"], bins=bins)
    prev = (tmp.groupby("Age_group", observed=True)["Heart Disease"]
            .apply(lambda s: (s == "Presence").mean() * 100))
    prev_df = prev.reset_index()
    prev_df.columns = ["Age Group", "Prevalence (%)"]
    prev_df["Age Group"] = prev_df["Age Group"].astype(str)
    fig = px.bar(prev_df, x="Age Group", y="Prevalence (%)",
                 title="Heart Disease Prevalence by Age Group",
                 color="Prevalence (%)", color_continuous_scale="OrRd")
    return fig

tab4 = gr.Interface(
    fn=prevalence_by_age,
    inputs=gr.Slider(minimum=5, maximum=20, step=5, value=10, label="Age bin width"),
    outputs=gr.Plot(label="Prevalence Chart"),
)


# Tab 5: Faceted histograms by Chest Pain Type
def faceted_hist(feature):
    fig = px.histogram(df, x=feature, color="Heart Disease",
                       facet_col="Chest pain type", facet_col_wrap=2,
                       color_discrete_map=color_map, barmode="overlay", opacity=0.7,
                       title=f"{feature} by Chest Pain Type")
    fig.update_layout(height=500)
    return fig

tab5 = gr.Interface(
    fn=faceted_hist,
    inputs=gr.Dropdown(choices=numeric_cols, value="Age", label="Select Feature"),
    outputs=gr.Plot(label="Faceted Histogram"),
)


demo = gr.TabbedInterface(
    [tab1, tab2, tab3, tab4, tab5],
    ["Explorer + Stats", "Correlations", "3D Scatter", "Prevalence", "Faceted Histograms"]
)
demo.launch()
```

---
## Verification — Run the final app

Save the final solution (Exercise 5 cell) as `app_gradio_complete.py`  
and run it to verify all five tabs work.

```python
# Write the complete solution to a .py file for testing
complete_app = '''
from pathlib import Path
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "predict_heart_disease_train.csv"
df = pd.read_csv(DATA_FILE).drop(columns=["id"])
numeric_cols = df.select_dtypes(include="number").columns.tolist()
color_map = {"Presence": "#e74c3c", "Absence": "#2ecc71"}


# Tab 1: Feature explorer with stats
def explore_with_stats(feature, chart_type):
    if chart_type == "Histogram":
        fig = px.histogram(df, x=feature, color="Heart Disease",
                           color_discrete_map=color_map, barmode="overlay", opacity=0.7)
    elif chart_type == "Box Plot":
        fig = px.box(df, x="Heart Disease", y=feature, color="Heart Disease",
                     color_discrete_map=color_map)
    else:
        fig = px.violin(df, x="Heart Disease", y=feature, color="Heart Disease",
                        color_discrete_map=color_map, box=True)
    stats = df[feature].describe().to_string()
    return fig, stats

tab1 = gr.Interface(
    fn=explore_with_stats,
    inputs=[
        gr.Dropdown(choices=numeric_cols, value="Age", label="Select Feature"),
        gr.Radio(choices=["Histogram", "Box Plot", "Violin"], value="Histogram", label="Chart Type"),
    ],
    outputs=[gr.Plot(label="Visualization"), gr.Textbox(label="Descriptive Statistics")],
)


# Tab 2: Correlation heatmap
def show_heatmap():
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Feature Correlation Heatmap")
    return fig

tab2 = gr.Interface(fn=show_heatmap, inputs=[], outputs=gr.Plot(label="Correlation Heatmap"))


# Tab 3: 3D scatter
def scatter_3d(x_col, y_col, z_col):
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                        color="Heart Disease", color_discrete_map=color_map,
                        opacity=0.5, title=f"{x_col} vs {y_col} vs {z_col}")
    return fig

tab3 = gr.Interface(
    fn=scatter_3d,
    inputs=[
        gr.Dropdown(choices=numeric_cols, value="Age", label="X axis"),
        gr.Dropdown(choices=numeric_cols, value="Max HR", label="Y axis"),
        gr.Dropdown(choices=numeric_cols, value="ST depression", label="Z axis"),
    ],
    outputs=gr.Plot(label="3D Scatter"),
)


# Tab 4: Prevalence by age group
def prevalence_by_age(bin_width):
    bin_width = int(bin_width)
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    bins = list(range(age_min, age_max + bin_width, bin_width))
    tmp = df.copy()
    tmp["Age_group"] = pd.cut(tmp["Age"], bins=bins)
    prev = (tmp.groupby("Age_group", observed=True)["Heart Disease"]
            .apply(lambda s: (s == "Presence").mean() * 100))
    prev_df = prev.reset_index()
    prev_df.columns = ["Age Group", "Prevalence (%)"]
    prev_df["Age Group"] = prev_df["Age Group"].astype(str)
    fig = px.bar(prev_df, x="Age Group", y="Prevalence (%)",
                 title="Heart Disease Prevalence by Age Group",
                 color="Prevalence (%)", color_continuous_scale="OrRd")
    return fig

tab4 = gr.Interface(
    fn=prevalence_by_age,
    inputs=gr.Slider(minimum=5, maximum=20, step=5, value=10, label="Age bin width"),
    outputs=gr.Plot(label="Prevalence Chart"),
)


# Tab 5: Faceted histograms by Chest Pain Type
def faceted_hist(feature):
    fig = px.histogram(df, x=feature, color="Heart Disease",
                       facet_col="Chest pain type", facet_col_wrap=2,
                       color_discrete_map=color_map, barmode="overlay", opacity=0.7,
                       title=f"{feature} by Chest Pain Type")
    fig.update_layout(height=500)
    return fig

tab5 = gr.Interface(
    fn=faceted_hist,
    inputs=gr.Dropdown(choices=numeric_cols, value="Age", label="Select Feature"),
    outputs=gr.Plot(label="Faceted Histogram"),
)


demo = gr.TabbedInterface(
    [tab1, tab2, tab3, tab4, tab5],
    ["Explorer + Stats", "Correlations", "3D Scatter", "Prevalence", "Faceted Histograms"]
)
demo.launch()
'''

with open("app_gradio_complete.py", "w") as f:
    f.write(complete_app)

print("Saved to app_gradio_complete.py")
print("Run:  python lectures_07_13_pandas_plots_scikit/lecture_08_EDA_plots/app_gradio_complete.py")
```

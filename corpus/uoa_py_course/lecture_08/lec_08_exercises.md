<!-- source: lectures_07_13_pandas_plots_scikit/lecture_08_EDA_plots/practice_exercises/lec_08_exercises.ipynb @ 0cc874704aaa -->

# Lecture 08 — Gradio App Exercises

Each exercise asks you to **add a new feature** to the base `app_gradio.py` Heart Disease Explorer.  
The exercises are cumulative — each one builds on the previous.

### Starting point (`app_gradio.py`)
The base app has:
- A **Dropdown** to select a numeric feature
- A **Radio** button to choose chart type (Histogram / Box Plot / Violin)
- A single **Plot** output
- Uses `gr.Interface`

### How to work
1. Copy the base app into each exercise cell
2. Modify it to add the requested feature
3. Run the cell — the Gradio app will launch inside the notebook
4. Stop the app before moving on (`demo.close()` or restart kernel)

### Reference
- [Gradio Quickstart](https://www.gradio.app/guides/quickstart)
- [Gradio Interface docs](https://www.gradio.app/docs/gradio/interface)
- [gr.TabbedInterface](https://www.gradio.app/docs/gradio/tabbedinterface)

---
## Exercise 1 — Add descriptive statistics as a second output

**Goal:** Return both the plot **and** a text summary of the selected feature.

**Steps:**
1. Make `explore_data()` return **two** values: the plotly figure and a string with `df[feature].describe()`
2. Add a second output to `gr.Interface`: a `gr.Textbox(label="Descriptive Statistics")`

**Hint:** `outputs` accepts a list: `outputs=[gr.Plot(...), gr.Textbox(...)]`

```python
# Exercise 1 — your code here
```

---
## Exercise 2 — Add a correlation heatmap tab

**Goal:** Create a second tab that displays a correlation heatmap of all numeric features.

**Steps:**
1. Keep the existing `gr.Interface` as `tab1`
2. Create a new function `show_heatmap()` (no inputs) that:
   - Computes `df[numeric_cols].corr()`
   - Returns `px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)`
3. Wrap it in a second `gr.Interface` as `tab2`
4. Combine both with `gr.TabbedInterface([tab1, tab2], ["Feature Explorer", "Correlations"])`

**Hint:** A function with no inputs uses `inputs=[]`

```python
# Exercise 2 — your code here
```

---
## Exercise 3 — Add a 3D scatter plot tab

**Goal:** Add a third tab where the user picks **X, Y, and Z axes** for a 3D scatter plot.

**Steps:**
1. Create a function `scatter_3d(x_col, y_col, z_col)` that returns `px.scatter_3d(...)`
2. Use three `gr.Dropdown` inputs (one per axis), each with `choices=numeric_cols`
3. Use good default values (e.g. Age, Max HR, ST depression)
4. Color by `"Heart Disease"` with the same `color_map`
5. Add it as `tab3` to the `TabbedInterface`

**Hint:** `px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color="Heart Disease", opacity=0.5)`

```python
# Exercise 3 — your code here
```

---
## Exercise 4 — Add age-group prevalence chart with a slider

**Goal:** Add a fourth tab showing heart disease prevalence (%) by age group,  
with a **slider** to control the age bin width.

**Steps:**
1. Create `prevalence_by_age(bin_width)` that:
   - Uses `pd.cut()` to bin the `"Age"` column
   - Groups by the age bins and calculates `(Heart Disease == "Presence").mean() * 100`
   - Returns a `px.bar()` with color mapped to prevalence
2. Use `gr.Slider(minimum=5, maximum=20, step=5, value=10)` as input
3. Add as `tab4`

**Hint:**
```python
bins = list(range(age_min, age_max + bin_width, bin_width))
tmp["Age_group"] = pd.cut(tmp["Age"], bins=bins)
prev = tmp.groupby("Age_group", observed=True)["Heart Disease"].apply(
    lambda s: (s == "Presence").mean() * 100
)
```

```python
# Exercise 4 — your code here
```

---
## Exercise 5 — Add faceted histograms by Chest Pain Type

**Goal:** Add a fifth tab with histograms faceted by `"Chest pain type"`.

**Steps:**
1. Create `faceted_hist(feature)` that returns a `px.histogram()` with:
   - `facet_col="Chest pain type"` and `facet_col_wrap=2`
   - Colored by `"Heart Disease"` with `barmode="overlay"`
2. Input: one `gr.Dropdown` for feature selection
3. Use `fig.update_layout(height=500)` so the facets have enough space
4. Add as `tab5`

**Hint:** `px.histogram(df, x=feature, color="Heart Disease", facet_col="Chest pain type", facet_col_wrap=2)`

```python
# Exercise 5 — your code here
```

---
## Summary

After completing all 5 exercises you will have built a **5-tab Gradio app** with:

| Tab | Feature | Gradio concepts used |
|-----|---------|---------------------|
| 1 | Feature explorer + statistics | Multiple outputs (`Plot` + `Textbox`) |
| 2 | Correlation heatmap | No-input interface, `px.imshow` |
| 3 | 3D scatter | Multiple `Dropdown` inputs, `px.scatter_3d` |
| 4 | Prevalence by age | `Slider` input, `pd.cut`, `px.bar` |
| 5 | Faceted histograms | Plotly facets, `facet_col` |

**Key Gradio patterns practiced:**
- `gr.Interface` with single and multiple inputs/outputs
- `gr.TabbedInterface` for multi-tab apps
- Input widgets: `gr.Dropdown`, `gr.Radio`, `gr.Slider`
- Output widgets: `gr.Plot`, `gr.Textbox`

# Lecture 08 — Practice Exercises

All exercises from the Lecture 08 notebooks collected in one place.  
Work through them after studying the corresponding lecture material.

| Section | Source Notebook | Topic |
|---|---|---|
| 1 | `lec_08a` | Interactive Plots (Plotly, Gapminder) |
| 2 | `lec_08e` | EDA on Heart Disease Dataset |
| 3 | `lec_08g` | Web App Frameworks |

---
## Setup — Load Libraries and Data

Run this cell first to have everything ready for the exercises below.

```python
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Heart disease dataset (used in sections 2, 3, and 4)
DATA_DIR = Path(".")  # same folder as this notebook
heart = pd.read_csv(DATA_DIR / "predict_heart_disease_train.csv").drop(columns=["id"])

# Gapminder dataset (used in section 1) — built into Plotly
gapminder = px.data.gapminder()

print(f"Heart disease dataset: {heart.shape[0]} rows, {heart.shape[1]} columns")
print(f"Gapminder dataset:     {gapminder.shape[0]} rows, {gapminder.shape[1]} columns")
heart.head(3)
```

---
## 1. Interactive Plots — Gapminder Faceted Plot

*From `lec_08a_interactive_plots.ipynb`*

### Exercise 1.1 — Make a useful and beautiful faceted plot

Using the **Gapminder** dataset and `plotly.express`, create a **faceted plot**
that tells a compelling story. Choose your own approach, for example:

- Choose only some time periods
- Compare two continents (e.g., Asia vs Africa, or Europe vs Americas)
- Pick specific countries from different continents to compare
- Create custom groups of countries and add a new "group" column  
  (e.g., EU, Eurozone, Balkans, OECD, Northern America)

**Requirements:**
- Use `px.scatter()` with `facet_col` or `facet_row`
- Make it visually appealing (colors, labels, title)
- The plot should communicate something meaningful about the data

```python
# Exercise 1.1 — Your code here
```

---
## 2. EDA on Heart Disease Dataset

*From `lec_08e_example_EDA_heart_disease.ipynb`*

These exercises build on the full EDA walkthrough of the heart disease dataset.  
Use either Plotly Express or Seaborn — whichever you prefer.

### Exercise 2.1 — Violin plot: Cholesterol by Sex and Heart Disease

Create a **violin plot** for `Cholesterol` grouped by both `Sex` and `Heart Disease`.

**Hint:** In Plotly Express, use `px.violin()` with `x`, `y`, and `color` parameters.

```python
# Exercise 2.1 — Your code here
```

### Exercise 2.2 — Faceted histogram: Age by Chest Pain Type

Make a **faceted histogram** of `Age` with one subplot per `Chest pain type`.

**Hint:** In Plotly Express, use `px.histogram()` with the `facet_col` parameter.

```python
# Exercise 2.2 — Your code here
```

### Exercise 2.3 — 3D scatter plot

Build a `px.scatter_3d()` plot with:
- **x** = Age
- **y** = Max HR
- **z** = ST depression
- **color** = Heart Disease

```python
# Exercise 2.3 — Your code here
```

### Exercise 2.4 — Heart disease prevalence by age group

Calculate and plot the **heart disease prevalence rate** by age group  
(e.g., 30–39, 40–49, 50–59, etc.).

**Steps:**
1. Create an `Age_group` column using `pd.cut()`
2. Group by `Age_group` and calculate the percentage of `"Presence"`
3. Plot the result as a bar chart

```python
# Exercise 2.4 — Your code here
```

---
## 3. Web App Frameworks

*From `lec_08g_web_app_frameworks.ipynb`*

These exercises require running `.py` files from the terminal.  
The apps were generated by the code cells in `lec_08g`.

### Exercise 3.1 — Run and interact with each web app

Run each of the four apps and interact with them in your browser:

```bash
streamlit run app_streamlit.py
python app_gradio.py
python app_dash.py
python app_taipy.py
```

**Write down** (in the cell below) one thing you liked and one limitation you noticed for each framework.

*Your observations here:*

| Framework | What I liked | Limitation I noticed |
|---|---|---|
| Streamlit | | |
| Gradio | | |
| Dash | | |
| Taipy | | |

### Exercise 3.2 — Add a correlation heatmap to the Streamlit app

Open `app_streamlit.py` and add a **correlation heatmap** as a third chart.

**Hints:**
- Use `px.imshow(filtered.corr(numeric_only=True), text_auto=True)`
- Add it below the existing two columns
- Run with `streamlit run app_streamlit.py` to see the result

```python
# Exercise 3.2 — Paste your modified Streamlit code here for reference
```

### Exercise 3.3 — Add descriptive statistics to the Gradio app

Modify `app_gradio.py` so that the `explore_data()` function returns
**two outputs**: the plot AND a text string with descriptive statistics
for the selected feature.

**Hints:**
- Change `outputs` to a list: `[gr.Plot(...), gr.Textbox(...)]`
- Return a tuple: `return fig, df[feature].describe().to_string()`
- Run with `python app_gradio.py`

```python
# Exercise 3.3 — Paste your modified Gradio code here for reference
```

### Exercise 3.4 — Add a second chart to the Dash app

Open `app_dash.py` and add a **histogram** that updates alongside the
existing scatter plot when the user changes the X-axis dropdown.

**Hints:**
- Add a second `dcc.Graph(id="histogram")` to the layout
- Modify the callback to return two figures (add a second `Output`)
- Run with `python app_dash.py`

```python
# Exercise 3.4 — Paste your modified Dash code here for reference
```

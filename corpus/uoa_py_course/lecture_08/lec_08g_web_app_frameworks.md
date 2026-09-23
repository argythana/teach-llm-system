<!-- source: lectures_07_13_pandas_plots_scikit/lecture_08_EDA_plots/reading_material/lec_08g_web_app_frameworks.ipynb @ 0cc874704aaa -->

# Lecture 08g: Python Web App Frameworks for Data Science

Once you've done your EDA and built a model, you often want to **share results**
with non-technical stakeholders. Web app frameworks let you create interactive
dashboards and demos without learning HTML/CSS/JavaScript.

In this notebook we introduce four frameworks:

| Framework | Best for | Complexity |
|---|---|---|
| **Streamlit** | Quick data apps and dashboards | Very easy |
| **Gradio** | ML model demos with inputs/outputs | Very easy |
| **Dash** | Production dashboards with callbacks | Medium |
| **Taipy** | Full data pipelines + UI | Medium |

**Important**: These frameworks run as **standalone web servers**, not inside Jupyter.
You write a `.py` file and run it from the terminal.

### Install
```bash
pip install streamlit gradio dash taipy
```

---
## 0. Core Concepts: How Python Web Apps Work

Before we look at specific frameworks, let's understand what happens when you
run a Python web app. These concepts apply to **all four frameworks** in this notebook.

### 0.1 The client-server model

Remember: when you run a Python script normally (`python my_script.py`), the script
runs once, prints its output, and exits.

A **web app** is different — it starts a **server**: a program that keeps running
and **waits for requests**.

| Concept | What it means | Everyday analogy |
|---|---|---|
| **Server** | Your Python script, running and waiting | A waiter standing ready to take orders |
| **Client** | The web browser that asks for pages | A customer placing an order |
| **Request** | The browser asking "show me this page" | The customer saying "I'd like a coffee" |
| **Response** | The server sending back HTML/charts/data | The waiter bringing the coffee |

When you run `python app_gradio.py`, your script becomes a server that listens
on a **port** (e.g., `localhost:7860`). Opening that URL in a browser sends a request,
and your Python code builds the page as a response.

> **localhost** = your own computer. **:7860** = the port number (like a door number
> in a building — different apps use different ports so they don't conflict).

### 0.2 State: what the app "remembers"

A key difference between a regular script and a web app:

* **Regular script**: variables exist while the script runs, then disappear.
* **Web app**: the server keeps running, so variables **persist** between user interactions.

This is called **state** — the current values of all variables the app is tracking.

```
User selects "Age" → state: selected_feature = "Age"
User changes to "Cholesterol" → state: selected_feature = "Cholesterol"
```

Each framework handles state differently:

| Framework | How state works |
|---|---|
| **Streamlit** | Re-runs your **entire script** on every interaction. Uses `st.session_state` to remember things between re-runs. |
| **Gradio** | Calls your **function** with the current input values. The function returns new outputs. No persistent state needed for simple apps. |
| **Dash** | Uses **callbacks** — functions that fire when specific inputs change. The app only re-runs the relevant callback, not the whole script. |
| **Taipy** | Uses a **state object** — when you change `state.variable`, the UI updates automatically (reactive binding). |

### 0.3 Reactivity: how user actions trigger code

In a Jupyter notebook, **you** decide when to run a cell.
In a web app, **the user** triggers code execution by interacting with widgets
(clicking buttons, moving sliders, selecting dropdowns).

The mechanism connecting a user action to your Python code differs by framework:

| Pattern | Used by | How it works |
|---|---|---|
| **Full re-run** | Streamlit | Every interaction re-executes the script from top to bottom |
| **Function call** | Gradio | Each interaction calls your function with the current inputs and displays the outputs |
| **Callback decorator** | Dash | You write `@app.callback(Output, Input)` — Dash calls your function only when those inputs change |
| **Reactive binding** | Taipy | You modify `state.variable` in a handler function — the UI auto-updates wherever that variable is used |

### 0.4 The three ingredients of every web app

Regardless of the framework, every Python web app has three parts:

1. **Data / Logic** — your Python code (pandas, plotly, sklearn, etc.)
2. **Layout / UI** — what the user sees (dropdowns, charts, text)
3. **Interactivity** — how user actions trigger your code and update the UI

```text
┌──────────────────────────────────────────────────┐
│                    Web App                       │
│                                                  │
│   ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
│   │   Data   │──→│  Layout  │──→│Interactivity│  │
│   │  /Logic  │   │   /UI    │   │             │  │
│   │          │←──│          │←──│  user acts  │  │
│   └──────────┘   └──────────┘   └─────────────┘  │
│                                                  │
└──────────────────────────────────────────────────┘
```

Each framework just gives you a different way to define these three parts.

**Summary:**
* A web app is a Python script that runs as a **server** and responds to **requests** from a browser.
* **State** is what the app remembers between user interactions.
* **Reactivity** is how user actions trigger your Python code.
* Every web app has three parts: **data/logic**, **layout/UI**, and **interactivity**.

**Shortest Summary:**
* **Server** = Python script that keeps running and waits.
* **State** = what the app remembers.
* **Reactivity** = user action → code runs → UI updates.

---
## 1. Streamlit — The Fastest Way to Build Data Apps

[Documentation](https://docs.streamlit.io/)  
[Gallery](https://streamlit.io/gallery)

Streamlit is the most popular framework for quick data apps.
It re-runs your entire script on every user interaction (simple but effective).

### How to run
1. Save the code below as a `.py` file (e.g., `app_streamlit.py`)
2. In the terminal: `streamlit run app_streamlit.py`
3. Your browser opens automatically at `http://localhost:8501`

```python
# This cell writes a Streamlit app to a .py file.
# NOTE: Do NOT run with "python ..." — Streamlit needs its own server.
# Run from the project root (uoa_py_course/):
#   streamlit run lectures_07_13_pandas_plots_scikit/lecture_08_EDA_plots/reading_material/app_streamlit.py

streamlit_code = '''
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Resolve data path relative to this script's location (works from any CWD)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "predict_heart_disease_train.csv"

# Page configuration
st.set_page_config(page_title="Heart Disease EDA", layout="wide")

# Title and description
st.title("Heart Disease EDA Dashboard")
st.markdown("Interactive exploration of the Heart Disease dataset.")

# Load data (cached so it only loads once)
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df.drop(columns=["id"])

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
age_range = st.sidebar.slider("Age Range", int(df.Age.min()), int(df.Age.max()), (30, 70))
sex_filter = st.sidebar.multiselect("Sex", options=[0, 1], default=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

# Apply filters
filtered = df[(df.Age.between(*age_range)) & (df.Sex.isin(sex_filter))]
st.sidebar.metric("Filtered patients", f"{len(filtered):,}")

# Layout: two columns
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(filtered, x="Age", color="Heart Disease",
                       color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"},
                       title="Age Distribution")
    st.plotly_chart(fig, width="stretch")

with col2:
    fig = px.scatter(filtered, x="Age", y="Max HR", color="Heart Disease",
                     color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"},
                     opacity=0.5, title="Age vs Max HR")
    st.plotly_chart(fig, width="stretch")

# Show raw data
if st.checkbox("Show raw data"):
    st.dataframe(filtered.head(100))
'''

with open("app_streamlit.py", "w") as f:
    f.write(streamlit_code)

print("Streamlit app saved to app_streamlit.py")
print("Run:  streamlit run lectures_07_13_pandas_plots_scikit/lecture_08_EDA_plots/reading_material/app_streamlit.py")
```

---
## 2. Gradio — ML Model Demos Made Easy

[Documentation](https://www.gradio.app/docs)  
[Gallery](https://www.gradio.app/demos)

Gradio is designed for **ML model demos**. You define inputs and outputs,
and Gradio builds the interface automatically.

Here we show a **local demo** — you can deploy this to
[Hugging Face Spaces](https://huggingface.co/spaces) for free sharing
(we'll cover deployment in a later lecture with a trained model).

```python
gradio_code = '''
from pathlib import Path
import gradio as gr
import pandas as pd
import plotly.express as px

# Resolve data path relative to this script's location (works from any CWD)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "predict_heart_disease_train.csv"

# Load data
df = pd.read_csv(DATA_FILE).drop(columns=["id"])

def explore_data(feature, chart_type):
    """Generate a plot based on user selections."""
    if chart_type == "Histogram":
        fig = px.histogram(df, x=feature, color="Heart Disease",
                           color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"},
                           barmode="overlay", opacity=0.7)
    elif chart_type == "Box Plot":
        fig = px.box(df, x="Heart Disease", y=feature,
                     color="Heart Disease",
                     color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"})
    else:  # Violin
        fig = px.violin(df, x="Heart Disease", y=feature,
                        color="Heart Disease",
                        color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"},
                        box=True)
    return fig

# Numeric columns for the dropdown
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Build the Gradio interface
demo = gr.Interface(
    fn=explore_data,
    inputs=[
        gr.Dropdown(choices=numeric_cols, value="Age", label="Select Feature"),
        gr.Radio(choices=["Histogram", "Box Plot", "Violin"], value="Histogram", label="Chart Type")
    ],
    outputs=gr.Plot(label="Visualization"),
    title="Heart Disease Data Explorer",
    description="Select a feature and chart type to explore the dataset."
)

demo.launch()  # Opens at http://localhost:7860
'''

with open("app_gradio.py", "w") as f:
    f.write(gradio_code)

print("Gradio app saved to app_gradio.py")
print("Run:  python lectures_07_13_pandas_plots_scikit/lecture_08_EDA_plots/reading_material/app_gradio.py")
```

---
## 3. Dash — Production-Ready Dashboards

[Documentation](https://dash.plotly.com/)  

Dash (by Plotly) is more powerful than Streamlit but requires more code.
It uses **callbacks** — functions that run when a user interacts with a widget.

Dash is the right choice when you need:
- Fine-grained control over layout
- Complex multi-page apps
- Production deployment with authentication

```python
dash_code = '''
from pathlib import Path
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Resolve data path relative to this script's location (works from any CWD)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "predict_heart_disease_train.csv"

# Load data
df = pd.read_csv(DATA_FILE).drop(columns=["id"])
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Create the Dash app
app = Dash(__name__)

# Layout: what the user sees
app.layout = html.Div([
    html.H1("Heart Disease EDA Dashboard"),
    html.Div([
        html.Label("Select X-axis feature:"),
        dcc.Dropdown(id="x-feature", options=numeric_cols, value="Age"),
    ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
    html.Div([
        html.Label("Select Y-axis feature:"),
        dcc.Dropdown(id="y-feature", options=numeric_cols, value="Max HR"),
    ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
    dcc.Graph(id="scatter-plot"),
])

# Callback: runs every time the user changes a dropdown
@app.callback(
    Output("scatter-plot", "figure"),
    Input("x-feature", "value"),
    Input("y-feature", "value")
)
def update_plot(x_feat, y_feat):
    fig = px.scatter(df, x=x_feat, y=y_feat, color="Heart Disease",
                     color_discrete_map={"Presence": "#e74c3c", "Absence": "#2ecc71"},
                     opacity=0.4, title=f"{x_feat} vs {y_feat}")
    return fig

if __name__ == "__main__":
    app.run(debug=True, port=8060)  # Opens at http://localhost:8060
'''

with open("app_dash.py", "w") as f:
    f.write(dash_code)

print("Dash app saved to app_dash.py")
print("Run:  python lectures_07_13_pandas_plots_scikit/lecture_08_EDA_plots/reading_material/app_dash.py")
```

---
## 4. Taipy — Data Pipelines + UI

[Documentation](https://docs.taipy.io/)  
[GitHub](https://github.com/Avaiga/taipy)

Taipy is a newer framework that combines:
- **Taipy GUI**: Build web UIs with a Markdown-like syntax
- **Taipy Core**: Define data pipelines and scenarios

It's particularly good for data science workflows where you want to
chain data loading → processing → modeling → visualization.

**Note**: For complex charts (color grouping, overlays), use the `figure` property
with Plotly figures instead of Taipy's native chart syntax. This gives you the
full power of `plotly.express` while Taipy handles the interactivity.

```python
taipy_code = '''
from pathlib import Path
from taipy.gui import Gui
import pandas as pd
import plotly.express as px

# Resolve data path relative to this script's location (works from any CWD)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "predict_heart_disease_train.csv"

# Load data
df = pd.read_csv(DATA_FILE).drop(columns=["id"])

# Default feature for the selector
selected_feature = "Age"
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Color map used by both charts
color_map = {"Presence": "#e74c3c", "Absence": "#2ecc71"}

# Build initial Plotly figures
hist_fig = px.histogram(df, x=selected_feature, color="Heart Disease",
                        color_discrete_map=color_map, barmode="overlay",
                        opacity=0.7, title=f"Distribution of {selected_feature}")

scatter_fig = px.scatter(df, x="Age", y="Max HR", color="Heart Disease",
                         color_discrete_map=color_map, opacity=0.5,
                         title="Age vs Max HR")

def on_feature_change(state):
    """Update the histogram when the user selects a different feature."""
    state.hist_fig = px.histogram(
        state.df, x=state.selected_feature, color="Heart Disease",
        color_discrete_map=color_map, barmode="overlay",
        opacity=0.7, title=f"Distribution of {state.selected_feature}"
    )

# Taipy page — uses the figure property for Plotly figures.
page = """
# Heart Disease Data Explorer

Select a feature: <|{selected_feature}|selector|lov={numeric_cols}|on_change=on_feature_change|>

<|chart|figure={hist_fig}|>

<|chart|figure={scatter_fig}|>
"""

if __name__ == "__main__":
    Gui(page=page).run(dark_mode=False, host="0.0.0.0", port=5000)
'''

with open("app_taipy.py", "w") as f:
    f.write(taipy_code)

print("Taipy app saved to app_taipy.py")
print("Run:  python lectures_07_13_pandas_plots_scikit/lecture_08_EDA_plots/reading_material/app_taipy.py")
print("Open: http://localhost:5000")
```

---
## 5. Comparison & Choosing the Right Framework

| Criteria | Streamlit | Gradio | Dash | Taipy |
|---|---|---|---|---|
| **Learning curve** | Very easy | Very easy | Medium | Medium |
| **Best for** | Dashboards, data apps | ML demos | Production apps | Data pipelines + UI |
| **Interactivity** | Re-runs script | Function-based | Callbacks | Reactive bindings |
| **Deployment** | Streamlit Cloud | Hugging Face Spaces | Any server | Taipy Cloud |
| **Customization** | Limited | Limited | High | High |
| **Popularity (2025)** | Very high | High | High | Growing |

### Practical advice
- **Start with Streamlit** — it's the fastest path from idea to working app.
- **Use Gradio** when demoing an ML model with clear inputs/outputs.
- **Switch to Dash** when you need production-grade features.
- **Try Taipy** when your app involves multi-step data pipelines.

### Practice Exercises
1. Run each `.py` file created above and interact with the web apps.
2. Add a new chart type to the Streamlit app (e.g., a correlation heatmap).
3. Modify the Gradio app to also show descriptive statistics as text output.
4. Add a second chart to the Dash app that updates alongside the scatter plot.

---
## 6. Managing Running Apps from the Terminal

Each framework starts a **web server** that keeps running until you stop it.
If you forget to stop one, the port stays occupied and you'll get an error next time.

### Stop a running app
In the terminal where the app is running, press:
```
Ctrl + C
```

### Find running apps (by port)
If you closed the terminal or forgot which apps are still running,
use `lsof` or `ss` to check the ports:

```bash
# Show which process is using a specific port (e.g., Streamlit on 8501)
lsof -i :8501

# Check multiple ports at once (all four frameworks)
lsof -i :8501 -i :7860 -i :8060 -i :5000

# Alternative using ss (no sudo needed)
ss -tlnp | grep -E '8501|7860|8060|5000'
```

### Kill a running app by port
```bash
# Kill whatever is running on port 8501 (Streamlit)
kill $(lsof -t -i :8501)

# Force kill if it doesn't stop
kill -9 $(lsof -t -i :8501)

# Kill all four framework ports at once
kill $(lsof -t -i :8501 -i :7860 -i :8060 -i :5000)
```

### Kill by process name
```bash
# Find and kill a Streamlit process
pkill -f "streamlit run"

# Find and kill a specific app file
pkill -f "app_dash.py"
```

### Quick reference

| Framework | Default Port | Find Command | Kill Command |
|---|---|---|---|
| **Streamlit** | 8501 | `lsof -i :8501` | `kill $(lsof -t -i :8501)` |
| **Gradio** | 7860 | `lsof -i :7860` | `kill $(lsof -t -i :7860)` |
| **Dash** | 8060 | `lsof -i :8060` | `kill $(lsof -t -i :8060)` |
| **Taipy** | 5000 | `lsof -i :5000` | `kill $(lsof -t -i :5000)` |

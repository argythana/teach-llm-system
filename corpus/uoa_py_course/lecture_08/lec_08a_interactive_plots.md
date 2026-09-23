<!-- source: lectures_07_13_pandas_plots_scikit/lecture_08_EDA_plots/reading_material/lec_08a_interactive_plots.ipynb @ 0cc874704aaa -->

# Lecture 08a: Interactive Plots — A Demo of Python Visualization with `Plotly Express`

In this notebook, we explore **interactive data visualization** using [Plotly Express](https://plotly.com/python/plotly-express/).

**Recommended video**: Watch the first 15 minutes of this SciPy 2021 [presentation](https://www.youtube.com/watch?v=FpCgG85g2Hw).

Run the notebook cell by cell while watching the presentation and:
* read the code,
* read about the use of each plot,
* read the info of the data presented in each plot.

The contents of the tutorial have been curated and edited by Thanasis Argyriou for the UOA BIS Postgrad python course.

The original tutorial code is here:
https://github.com/nicolaskruchten/scipy2021/blob/main/styled.ipynb

```python
# Before running this notebook, make sure you have plotly installed.
# With your virtual environment activated, run in the terminal:
# python -m pip install plotly
```

```python
# plotly.express (aliased as px) provides one-line functions for interactive plots.
import plotly.express as px

# Suppress FutureWarnings so our output stays clean.
# In real projects, you'd eventually address these warnings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

## 1. The GAPMINDER dataset.   

Each row represents:
* a country,
* its life expectancy,
* population,
* GDP per capita,
* on given year.  

We have: 1074 rows in the data.

```python
# Plotly comes with built-in demo datasets. px.data.gapminder() loads the
# famous Gapminder dataset as a pandas DataFrame — handy for learning!
df = px.data.gapminder()
type(df)  # Confirm it's a DataFrame
```

```python
df.sample(5, random_state=1) # randomness always the same for all, everytime.
```

```python
len(df)
```

### pandas query method for year 2007: 142 rows, meaning 142 countries.

```python
# Use the pandas .query() method to filter rows.
# We keep only year 2007 so each country appears once (one row per country).
df = df.query("year == 2007")
```

```python
df.sample(5, random_state=1)
```

```python
len(df)
```

```python
# df.country.unique()
```

```python
len(df.country.unique())
```

## 2. Basic interactive plots.

### Strip  plot, also known as dot plot.  

A scatterplot where one variable is categorical: https://en.wikipedia.org/wiki/Dot_plot_(statistics).

```python
# Simplest strip plot: each dot is a country, positioned by life expectancy.
# A strip plot is a scatter plot where one axis is categorical.
fig = px.strip(
    df,
    x="lifeExp"
)  # Assign the plot to a variable called 'fig' (a Plotly Figure object)
```

```python

fig.show()  # Call .show() explicitly to render the plot
```

```python
# px.strip?
```

```python
# Adding hover_name makes the country name appear prominently when you
# hover over a dot. This is one of the biggest advantages of interactive plots!
px.strip(
    df,
    x="lifeExp",
    hover_name="country"
)
```

```python
px.strip(
    df,
    x="lifeExp", 
    hover_name="continent", 
    color="country"
)
```

```python
px.strip(
    df,
    x="lifeExp", 
    hover_name="country", 
    color="continent"
        )
```

### Histogram  
https://en.wikipedia.org/wiki/Histogram

```python
# A histogram groups data into "bins" (ranges) and counts how many values
# fall in each bin. The 'marginal="rug"' adds tick marks along the x-axis
# showing individual data points — great for seeing the actual distribution.
px.histogram(
    df,
    x="lifeExp",
    hover_name="country",
    color="continent",
    marginal="rug"
)
```

```python
px.histogram(df, 
             x="lifeExp", 
             y="pop",
             hover_name="country",
             color= "continent",
             marginal="rug"
            )
```

```python
px.histogram(df, 
             x="lifeExp", 
             y="pop",
             hover_name="country", 
             color="continent",
             marginal="rug",
             facet_col="continent"
            )
```

### Bar chart and stacked bar chart 
https://en.wikipedia.org/wiki/Bar_chart

```python
# stacked bar chart
px.bar(
    df, 
    x="pop", 
    y="continent", 
    color="lifeExp", 
    hover_name="country"
)
```

### Sunburst diagramm
Used to represent "part-to-whole-relationships", nested structures, tree-like structures.  

Also known as ```multi-level pie chart```, or ```ring chart```.  
https://en.wikipedia.org/wiki/Pie_chart#Ring

```python
df.head()
```

```python
px.sunburst(
    df,
    values="pop",
    path=["continent", "country"],
    color="lifeExp",bar
    hover_name="country",
    height=600
           )
```

### Treemap diagram
Used to show hierachical data (tree-like) structures in nested boxes.  
https://en.wikipedia.org/wiki/Treemapping

```python
help(px.treemap)
```

```python
# Use to show tree like structures in nested boxes.
px.treemap(
        df,
        values="pop",
        path=["continent", "country"],
        color= "lifeExp",
        hover_name="country",
        custom_data=["gdpPercap"],  # add GDP per capita as custom data
        hover_data=["gdpPercap"],  # display GDP per capita in hover data
        labels={"gdpPercap": "GDP per Capita"},  # display GDP per capita in the box
        color_continuous_scale='RdYlGn',
        height=600,{"gdpPercap": "GDP per Capita"},
        width=1000
          )
```

### Choropleth diagram   
https://en.wikipedia.org/wiki/Choropleth_map  
Use intensity of color to map values of data to geolocation.

```python
px.choropleth(
    df,
    locations="iso_alpha",
    color= "lifeExp",
    hover_name="country"
             )
```

### Scatter plot
https://en.wikipedia.org/wiki/Scatter_plot  

Bi-variate relationship representation.

```python
# px.scatter?
```

```python
# A scatter plot shows the relationship between two numeric variables.
# Here we explore: does higher GDP per capita relate to higher life expectancy?
px.scatter(
    data_frame=df,
    x="gdpPercap",
    y="lifeExp",
    hover_name="country",
    height=400
)
```

```python
# Enhanced scatter: color by continent, bubble size by population.
# log_x=True uses a logarithmic x-axis — useful when values span several
# orders of magnitude (e.g., GDP ranges from ~300 to ~50,000).
fig = px.scatter(
    df,
    x="gdpPercap",
    y="lifeExp",
    hover_name="country",
    color="continent",
    size="pop",         # bubble size proportional to population
    size_max=60,        # cap the maximum bubble diameter
    log_x=True,         # logarithmic scale on x-axis
    height=400
)

fig.show()
```

### Check attributes of plotly "figure objects."  
They are dictionaries of attributes. Update attributes to control the figure.

```python
# try the filter search bar at the right of this cell's output.
# fig.show("json")
```

## 2. Detailed plot of significant dimensions of the GAPMINDER dataset.   

Each row represents a country, its fife expectancy, population and GDP per capita on a given year.  
142 rows for year 2007 => 142 countries.

```python
df = px.data.gapminder().query("year == 2007")  # Reassign because I may have changed something.
```

```python
# Create plot object and assign to a value
fig = px.scatter(
    df, 
    y="lifeExp",
    x="gdpPercap", 
    color="continent", 
    log_x=True, 
    size="pop", 
    size_max=60,
    hover_name="country", 
    height=600,
    width=1000, 
    template="simple_white",     df,    df,
    x="gdpPercap",
    y="lifeExp",
    hover_name="country",
    x="gdpPercap",
    y="lifeExp",
    hover_name="country",
    color_discrete_sequence=px.colors.qualitative.G10,
    title="Health vs Wealth 2007",
    labels=dict(
            continent="Continent",
            pop="Population",
            gdpPercap="GDP per Capita (US$, price-adjusted)", 
            lifeExp="Life Expectancy (years)")
)    df,
    x="gdpPercap",
    y="lifeExp",
    hover_name="country",

# Update layout
fig.update_layout(
    font_family="Rockwell",
    legend=dict(
                orientation="h", 
                title="Continent", 
                y=1.1,     df,
    x="gdpPercap",
    y="lifeExp",
    hover_name="country",
                x=1, 
                xanchor="right", 
                yanchor="bottom"
    )
)

# Update x and y axes
fig.update_xaxes(tickprefix="$", range=[2,5], dtick=1)
fig.update_yaxes(range=[30,90])

fig.add_hline((df["lifeExp"]*df["pop"]).sum()/df["pop"].sum(), line_width=1, line_dash="dot")
fig.add_vline((df["gdpPercap"]*df["pop"]).sum()/df["pop"].sum(), line_width=1, line_dash="dot")

fig.show()
```

## 4. Curated, extra stylized version of plot.

```python
df = px.data.gapminder().query("year == 2007")

fig = px.scatter(
    df, 
    y="lifeExp", 
    x="gdpPercap", 
    color="continent",
    log_x=True, 
    size="pop", 
    size_max=60, 
    hover_name="country",
    height=600, 
    width=900, 
    template="simple_white", 
    color_discrete_sequence=px.colors.qualitative.G10,
    title="Health vs Wealth 2007",
    # this dictionary syntax is more readable.
    labels=dict(
            continent="Continent", 
        pop="Population",
            gdpPercap="GDP per Capita (US$, price-adjusted)", 
            lifeExp="Life Expectancy (years)")
)


fig.update_layout(
        font_family="Rockwell",
        legend=dict(
            orientation="h", title="", y=1.1, x=1,
            xanchor="right", yanchor="bottom")
)

fig.update_xaxes(tickprefix="$", range=[2,5], dtick=1)
fig.update_yaxes(range=[35,90])

fig.add_hline((df["lifeExp"]*df["pop"]).sum()/df["pop"].sum(), line_width=1, line_dash="dot")
fig.add_vline((df["gdpPercap"]*df["pop"]).sum()/df["pop"].sum(), line_width=1, line_dash="dot")

fig.show()

## Uncomment any line below to save to selected file version.

fig.write_html("gapminder_2007.html") # interactive file type for websites.
# fig.write_json("gapminder_2007.json") # serialized export
```

```python
### To save as .svg you need kaleido.
## Image export using the "kaleido" engine requires the kaleido package,
## which can be installed using pip: $ pip install -U kaleido

# fig.write_image("gapminder_2007.svg") # static, non-interactive export.
```

## 5. Expanded Gapminder data examples by Thanasis Argyriou

### Faceted scatterplot. Using data for more years.

```python
df_all_years = px.data.gapminder()
df_all_years.sample(5, random_state=3)
```

```python
df_all_years.columns
```

```python
# 12 yearly periods data
df_all_years.year.unique()
```

```python
# 12 yearly periods data
df_all_years.year.unique().size
# len(df_all_years.year.unique()) # same result as above.
```

```python
# 1700 rows
len(df_all_years)
```

#### The plot below is a typical example of the very bad practive of "overplotting".  
This is just to demonstrate the code.

```python
# add facets by year in 4 columns on chart
fig = px.scatter(
    df_all_years, 
    x='gdpPercap', 
    y='lifeExp', 
    color='continent',
    size='pop', 
    hover_name="country",
    facet_col='year', 
    facet_col_wrap=4, # add facets by year in 4 columns on chart
    log_x=True, 
    height=800, width=1100
)

fig.update_xaxes(tickprefix="$", dtick=1)

fig.show()
```

#### Homework assignment: Make a useful and beautiful faceted plot.  
Do what you think would make this date useful and beautiful in a faceted plot.    
E.g.:    
Chose only some periods.  
Chose two continents to compare such as Asia VS Africa, or Europe VS Americas.   
Or chose some countries or a single country from a continent to compare with another group of countries.   
You may create groups of countries and add a new dimension "groups", e.g. EU, EZ, Balkans, OECD, Northern America.    
For country groups you may check World Bank groups.

### Animated plot with slider and play button.

```python
df = px.data.gapminder()

fig = px.scatter(
    df,
    x="gdpPercap",
    y="lifeExp",
    color="continent",
    size="pop", 
    size_max=50,
    hover_name="country",
    animation_frame="year", 
    animation_group="country",
    height=600, width=950, 
    template="simple_white", 
    color_discrete_sequence=px.colors.qualitative.G10,
    log_x=True, 
    range_x=[100,100000], 
    range_y=[25,90],
    title="Health vs Wealth, 1952-2007",
    
    ##this dictionary syntax below is more readable.
    labels=dict(
        continent="Continent", pop="Population",
        gdpPercap="GDP per Capita (US$, price-adjusted)", 
        lifeExp="Life Expectancy (years)")
    )

fig.update_layout(
    font_family="Rockwell",
    legend=dict(
        orientation="h", title="", y=1, x=0.9,
        xanchor="right", yanchor="bottom")
)
    labels=dict(
        continent="Continent", pop="Population",
        gdpPercap="GDP per Capita (US$, price-adjusted)", 
        lifeExp="Life Expectancy (years)")
    )
fig.update_xaxes(tickprefix="$", dtick=1)
# fig.update_yaxes(range=[35,90])

# comment this line to remove play, pause buttons
#fig["layout"].pop("updatemenus")
fig.show()

## Uncomment line below to save to necessary file version.
fig.write_html("gapminder_1952-2007.html")
```

### Animated plot with slider.

```python
# Basic example below from https://plotly.com/python/animations/

df = px.data.gapminder()

fig = px.scatter(
    df, 
    x="gdpPercap", 
    y="lifeExp",
    animation_frame="year", 
    animation_group="country",
    size="pop", 
    color="continent", 
    hover_name="country",
    height=500, width=900,
    log_x=True, 
    size_max=55, 
    range_x=[100,100000], 
    range_y=[25,90]
                )

# comment out this line to bring back the play, pause buttons.
fig["layout"].pop("updatemenus") # optional, drop animation buttons

fig.show()
```

### Annotate a selected country
Example A

```python
fig = px.scatter(
    df, 
    y="lifeExp", 
    x="gdpPercap", 
    color="continent",
    
    # add text over selected country:
    #text = df.country, This works and shows all countries
    #text = (df.country == "Greece"), # this results to "true" and works.

    text = (df.country == "Greece").replace(True, "GR"),  
    size="pop", 
    size_max=50, 
    hover_name="country",
    animation_frame="year", 
    animation_group="country",
    height=600,
    width=1000, 
    template="simple_white", 
    color_discrete_sequence=px.colors.qualitative.G10,
    log_x=True, 
    range_x=[100,100000], 
    range_y=[25,90],
    title="Health vs Wealth, 1952-2007",
    # This dictionary syntax below is more readable.
    labels=dict(
        continent="Continent", pop="Population",
        gdpPercap="GDP per Capita (US$, price-adjusted)", 
        lifeExp="Life Expectancy (years)")
)

fig.update_traces(textposition='top center')

fig.update_layout(
    font_family="Rockwell", 
    legend=dict(
        orientation="h", title="", y=1, x=0.9,
        xanchor="right", yanchor="bottom")
)

fig.update_xaxes(tickprefix="$", dtick=1)
# fig.update_yaxes(range=[35,90])

# comment this line to remove play, pause buttons
#fig["layout"].pop("updatemenus")
fig.show()

## Uncomment line below to save to necessary file version.
# fig.write_html("gapminder_1952-2007.html")
```

#### Annotate selected country solution B

```python
fig = px.scatter(
    df, y="lifeExp", x="gdpPercap", color="continent",
    
    # add text over selected country:
    #text = ((df.country == "Greece")).replace(True, "GR"),  # super fast hacky solution
    
    # this works too and is syntatically correct
    text = ["GR" if country_name == True else "" for country_name in (df.country == "Greece")], 
    size="pop",
    size_max=50,
    hover_name="country",
    animation_frame="year",
    animation_group="country",
    height=600, width=1000,
    template="simple_white", 
    color_discrete_sequence=px.colors.qualitative.G10,
    log_x=True,
    range_x=[100, 100000],
    range_y=[25, 90],
    title="Health vs Wealth, 1952-2007",
    ##this dictionary syntax below is more readable.
    labels=dict(
        continent="Continent", pop="Population",
        gdpPercap="GDP per Capita (US$, price-adjusted)", 
        lifeExp="Life Expectancy (years)"))

fig.update_traces(textposition='top center')

fig.update_layout(
    font_family="Rockwell",
    legend=dict(
        orientation="h", title="", y=1, x=0.9,
        xanchor="right", yanchor="bottom")
)

fig.update_xaxes(tickprefix="$", dtick=1)
# fig.update_yaxes(range=[35,90])

# comment this line to remove play, pause buttons
#fig["layout"].pop("updatemenus")
fig.show()

## Uncomment line below to save to necessary file version.
# fig.write_html("gapminder_1952-2007.html")
```

### Year slider + clickable continent legend
Combine two controls that actually **work together**:
- **Year slider** (bottom): scrub through time — powered by `animation_frame`.
- **Continent legend** (right): Plotly legends are clickable by default!
  - **Single-click** a continent name → hide/show that continent.
  - **Double-click** a continent name → isolate it (hide all others).
  - **Double-click again** → "Show All" (bring everything back).

We also add a small **"Show All"** button for convenience.

Unlike stacking multiple independent sliders (where only the last-selected one is active),
this approach lets the slider and the legend filter affect the same plot simultaneously.

```python
# Strategy:
#   1. px.scatter with animation_frame="year" auto-creates play/pause buttons + year slider.
#   2. color="continent" creates one trace per continent → each appears in the legend.
#   3. Plotly legends are CLICKABLE by default:
#        - single-click a continent → hide/show it
#        - double-clic,
    template="simple_white",
    color_discrete_sequence=px.colors.qualitative.G10,
    title="Life Expectancy vs GDP per capita",
    labels=dict(k a continent → isolate it (hide others)
#        - double-click again → show all
#   4. We add one small "Show All" button to make the reset action obvious.
#   5. We label two countries (USA, China) on every bubble, like we did for "GR" above.

df = px.data.gapminder()

# Build a per-row text label: short code if the country is one we want to highlight,
# empty string otherwise. This list aligns 1:1 with df rows (and with each animation frame).
country_labels = {"United States": "USA", "China": "CN"}
labels_per_row = [country_labels.get(c, "") for c in df["country"]]

fig = px.scatter(
    df, x="gdpPercap", y="lifeExp",
    animation_frame="year",
    animation_group="country",
    color="continent",
    size="pop",
    size_max=50,
    hover_name="country",
    text=labels_per_row,         # <-- short text shown next to each bubble
    log_x=True,
    range_x=[100, 100000],
    range_y=[25, 90],
    template="simple_white",
    color_discrete_sequence=px.colors.qualitative.G10,
    title="Life Expectancy vs GDP per capita",
    labels=dict(
        continent="Continent", pop="Population",
        gdpPercap="GDP per Capita (US$, price-adjusted)",
        lifeExp="Life Expectancy (years)")
)

# Position the text just above each bubble
fig.update_traces(textposition="top center", textfont=dict(size=11))

# --- "Show All" button: makes every continent trace visible again ---
n_traces = len(fig.data)
show_all_button = dict(
    type="buttons",
    direction="left",
    x=1.02,
    y=0.5,
    xanchor="left",
    yanchor="middle",
    showactive=False,
    bgcolor="white",
    bordercolor="#999",
    font=dict(size=11),
    pad=dict(r=5, t=5),
    buttons=[dict(
        label="Show All",
        method="update",
        args=[{"visible": [True] * n_traces}]
    )]
)``

# Plotly's animation_frame auto-creates play/pause in updatemenus[0].
# We must keep t
)hose and APPEND our button, not replace them.
existing_menus = list(fig.layout.updatemenus)
existing_menus.append(show_all_button)

fig.update_layout(
    font_family="Rockwell",
    height=600,
    width=1000,
    updatemenus=existing_menus,
    # Vertical legend on the right — easier to click each continent
    legend=dict(
        title="Continent<br><sub>(click to filter out)</sub>",
        orientation="v",
        x=1.02, y=1,
        xanchor="left", yanchor="top",
    ),
    margin=dict(r=180),  # extra right margin for legend + button
)

fig.update_xaxes(tickprefix="$", dtick=1)
fig.show()
```

Detailed configuration arguments for animations:    
https://plotly.com/python/v3/gapminder-example/

## 6. Extra: bokeh, a library similar to plotly.

[bokeh demo](https://demo.bokeh.org/)

A similar tutorial for bokeh libary using the gapminder dataset.    
https://demo.bokeh.org/gapminder   
https://www.kaggle.com/code/parulpandey/recreating-gapminder-visualisation-with-bokeh/notebook

The original retro presentation of the GAPMINDER data:   
https://www.youtube.com/watch?v=hVimVzgtD6w&t=275s

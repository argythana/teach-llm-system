<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_06_define_functions/reading_material/lec_06a_custom_functions_demo.ipynb @ 0cc874704aaa -->

# Lecture 06a. Basic examples of user defined functions

In this notebook, we will learn how to define and use custom functions in Python using a practical example.

We combine user defined functions with functions from the `pandas` library to create a report on a dataset.

In the next notebook of this lecture, we will discuss more about topics in functions, in a more abstract way, using more examples.

Learning Goals:
* syntax for defining a function.
* `docstrings` (documentation strings) for functions.
* syntax for `calling` a function.
* the difference between `parameters` and `arguments`.
* the difference between `positional` and `keyword` arguments.
* the difference between `required` and `optional` arguments.
* the `return` statement.


<mark>Necessary homework reading:</mark>

The notes from the 2nd lecture on `built-in` functions, `parameters`, `arguments`, `docstrings` from the file:

`function_parameters_arguments.ipynb`.

 You may find the file on eclass (lecture_02) or in the [GitHub Course Repo](https://github.com/argythana/uoa_py_course/blob/main/lectures_01_06_fundamentals_for_data_science/lecture_06_define_functions/function_parameters_arguments.ipynb).

**First: activate venv and install the following 3 python packages**
```bash
pip install pandas matplotlib openpyxl
```

```python
import pandas as pd  # Pandas is the next lecture material
```

```python
# This is the Present Working Directory (pwd) of this notebook in your PC.pwd
#! pwd
```

```python
# # if your data file is NOT in the same directory as your notebook, you need to specify the path.
# data_dir = "./../../data/"
# data = pd.read_excel(data_dir + "grades_factors.xlsx")
```

```python
# if the data file is in the same directory as your notebook, you can just use the file name.

# read the data from the Excel file and load the data into the PC memory
data = pd.read_excel("grades_factors.xlsx")
```

```python
# Uncomment to see the data
data
```

```python
# length of the data
len(data)
```

```python
data
```

```python
type(data)
```

This is the dataset from your assignment in another course.
Factors that affect the grade in Calculus.

## 1. Some `pandas` methods to get info about the data.
We will learn about `pandas` in the next class. For now, just a few useful methods to work with columns.

```python
type(data)
```

```python
# The columns in the data. Notice the notation: Index(['...', '...'])
data.columns
```

```python
# Using indexing to get the first item of the columns
data.columns[0]
```

```python
data.dtypes
```

```python
# a view of the columns as a list
data.columns.to_list()
```

```python
# Assign the column headers list to a variable name
column_headers = data.columns.to_list()
```

```python
type(column_headers)
```

```python
column_headers
```

```python
len(data.columns)
```

```python
for header in column_headers:
    header.replace(" ", "_")
```

```python
column_headers
```

```python
# Replace white spaces inside the column names with underscores using a loop
column_headers = [header.replace(" ", "_") for header in column_headers]
column_headers
```

```python
# Convert all capital letters to lowercase in the column names using a loop
column_headers = [i.lower() for i in column_headers]
column_headers
```

```python
# a pandas syntax to replace white spaces inside the column names with underscores
data.columns.str.replace(" ", "_")
```

```python
# pandas syntax to convert all capital letters to lowercase in the column names
data.columns.str.lower()
```

```python
data.columns
```

## 2. Define a simple function to modify the headers of the data.

```python
# function definition: name, parameters, docstring, body
def fix_headers(headers_list):
    """
    This function takes a list of headers and returns a list of headers in which the
    white spaces are replaced by underscores, and all letters converted to lowercase.
    Params:
        headers_list: list of strings.
    Returns:
        headers_list_fixed: list of strings fixed with underscores and lowercase letters.
    """
    # Replace white spaces inside the column names with underscores
    headers_list_with_underscores = headers_list.str.replace(" ", "_")  # Does not word in-place, need to reassign.

    # Convert all capital letters to lowercase in the column names
    headers_list_fixed = headers_list_with_underscores.str.lower()

    return headers_list_fixed
```

```python
fix_headers?
```

```python
help(fix_headers)
```

```python
# The column names
data.columns
```

```python
# An interactive view of how fix_headers() works on the data.columns object.
fix_headers(data.columns)
```

```python
# The headers did not change because? What do you think?
data.columns
```

```python
# To modify the headers, you need to assign the returned VALUE of the function to data columns
data.columns = fix_headers(headers_list=data.columns)
```

```python
data.columns
```

```python
data
```

## 3. Pandas custom functions to calculate statistics of data features.

```python
# Get average of a column
data["calc_hs"].mean()
```

```python
# Get standard deviation of a column
data["calc_hs"].std()
```

```python
# Get average descriptive stats from all columns.
data.describe()
```

```python
# Slice the data to get descriptive stats from some columns.
data[["calc_hs", "act_math", "alg2_grade"]].describe()
```

## 4. Define a function to create a report on the data.

```python
def report_on_data(data_file, columns=None, save_to_file=False):
    """
    Take an Excel file and selected columns and return a report on the data as DataFrame.
    Params:
        data_file: An Excel filename as string.
        columns: A list of columns to include in the report. If None, all columns are included.
        save_to_file: A boolean value to save the report to a file.
    Returns:
        descr_stats_report: A pandas DataFrame with descriptive stats of the data.
    """

    # Load the data from the Excel file into memory
    my_data = pd.read_excel(data_file)

    # this block of code does the same as the next one, using the opposite logic.
    # if columns == None:
    #     my_data = my_data.describe()

    # if user provides a list of columns, slice the data to get descriptive stats from those columns.
    if columns is not None:
        my_data = my_data[columns]

    # Fix the column headers
    my_data.columns = fix_headers(my_data.columns)

    # Create report
    descr_stats_report = my_data.describe()

    # if user wants to save the report to a file, ask for the filename and save it.
    if save_to_file:
        new_filename = input("Enter the name of the file to save the report: ")
        descr_stats_report.to_excel(f"{new_filename}.xlsx", index=True)
        print("Report saved to file.")
    
    return descr_stats_report
```

```python
report_on_data.__doc__
```

```python
help(report_on_data)
```

## 5. Call the function using only 'positional' arguments
**When we use positional arguments, we need to use the same order for arguments as in the function definition.**

```python
# Call the function with only the filename.
# This means that the function will use all columns and the report will not be saved to a file.
report_on_data("grades_factors.xlsx")
```

```python
# call the function with the filename and a list of columns
# report only for 2 columns
# report will not be saved to a file
report_on_data(
    data_file="grades_factors.xlsx",
    columns=['Calc HS', 'ACT Math'],
    save_to_file=False,
)
```

```python
# When we use positional arguments, we need the same order for arguments as in the definition.
# THIS WILL NOT WORK.
report_on_data(
    columns=['Calc HS', 'ACT Math'],
    data_file="grades_factors.xlsx",
    verbose=False,
)
```

## 6. Call the function using 'keyword' arguments
This will change nothing in the function, but it will **make the call to the function more readable**.

```python
# Call the function with the filename and a list of columns
# Report only for 2 columns
# Report will not be saved to a file
report_on_data(
    data_file="grades_factors.xlsx",
    columns=['Calc HS', 'ACT Math'],
    save_to_file=False,
)
```

```python
# When we use keyword arguments, we can change the order of the arguments.
# Changing keyword arguments orders is not recommended, but works.
report_on_data(
    data_file="grades_factors.xlsx",
    save_to_file=True,
)
```

## 7. Call the function and assign the returned value to a variable
**The `return` statement returns the output of the function to the caller for further processing.**

Actually, functions should be defined to do just one thing and do it well.

So, it is not optimal to define a function that does two things: 1. create a report and 2. save it to a file.

It is better to define two functions: one for creating the report and another for saving it to a file. And then just call the functions one after another in the main program.

```python
# call the function and assign the returned value to a variable
example_small_report = report_on_data(
    data_file="grades_factors.xlsx",
    columns=['Calc HS', 'ACT Math'],
    save_to_file=True, # Do not save to a file.
)
```

```python
# show the value of the variable "example_small_report"
example_small_report
```

```python
# the type of the returned value
type(example_small_report)
```

## 8. Save the report to a file using the pandas method `to_excel()`

```python
# Since it is a pandas DataFrame, we can use a pandas method to
# save the report to an Excel file.
example_small_report.to_excel(
    "my_first_python_report_to_excel.xlsx",
    index=True, # comment out to see what happens
    # header=False # uncomment to see what happens
)
```

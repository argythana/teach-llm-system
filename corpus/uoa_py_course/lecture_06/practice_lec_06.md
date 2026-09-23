<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_06_define_functions/practice_exercises/practice_lec_06.ipynb @ 0cc874704aaa -->

# Lecture 06 Practice: Define Functions

**Topics:** Defining functions, parameters & arguments, return values, default parameters, scope.

---

## Exercise 1: Temperature converter with validation

Write a function `convert_temp(value, from_unit)` that:  
- Converts Celsius to Fahrenheit or vice versa depending on `from_unit` ("C" or "F")  
- Returns the converted value rounded to 1 decimal place  
- Returns `None` if `from_unit` is invalid  

Test with: `convert_temp(100, "C")` → 212.0, `convert_temp(32, "F")` → 0.0

```python
# Your code here
```

## Exercise 2: Statistics from scratch

Without using the `statistics` module, write three functions:  
- `my_mean(numbers)` — returns the arithmetic mean  
- `my_median(numbers)` — returns the median (handle both odd and even length lists)  
- `my_mode(numbers)` — returns the most frequent value  

Test with: `sample = [4, 7, 2, 7, 9, 1, 7, 3, 5]`

```python
sample = [4, 7, 2, 7, 9, 1, 7, 3, 5]

# Your code here
```

## Exercise 3: Password strength checker

Write a function `check_password(pwd)` that returns a strength score (0-4) based on:  
- +1 if length >= 8  
- +1 if contains at least one uppercase letter  
- +1 if contains at least one digit  
- +1 if contains at least one special character (`!@#$%^&*`)  

Also print a message: 0-1 "Weak", 2 "Fair", 3 "Good", 4 "Strong".  
Test with several passwords of varying complexity.

```python
# Your code here
```

## Exercise 4: Function composition

Write three small functions that each do one transformation on a string:  
- `clean(text)` — removes leading/trailing whitespace and converts to lowercase  
- `remove_punctuation(text)` — removes all non-alphanumeric characters (keep spaces)  
- `word_count(text)` — returns a dictionary with each word and its count  

Then write a main function `analyze(text)` that chains all three and returns the word count dict.  
Test with: `"  Hello, World! Hello, Python!  "`

```python
# Your code here
```

## Exercise 5: Default parameters and flexible arguments

Write a function `format_name(first, last, title="Mr.", reverse=False)` that:  
- By default returns: `"Mr. John Smith"`  
- If `reverse=True`, returns: `"Smith, Mr. John"`  
- The `title` parameter allows: `"Dr."`, `"Prof."`, etc.  

Test it with different combinations of arguments. Try calling with keyword arguments in different orders.

```python
# Your code here
```

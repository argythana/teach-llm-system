<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_05_boolean_conditions_control_statements/practice_exercises/practice_lec_05a_boolean.ipynb @ 0cc874704aaa -->

# Lecture 05a Practice: Boolean Operations & Comparisons

**Topics:** Comparison operators, `is` vs `==`, membership tests (`in`), boolean logic (`and`, `or`, `not`), `bool()`.

---

## Exercise 1: Debug the comparisons

The code below uses `==` and `is`. Predict the output for each line, then run to verify.  
Write a comment explaining **why** `is` behaves differently from `==`.

```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

# Predict True or False for each. Run to verify.
print(a == b)    # ?
print(a is b)    # ?
print(a is c)    # ?
print(a == c)    # ?
print(b is c)    # ?

# Explain: why does `is` give different results than `==`?
```

## Exercise 2: Short-circuit evaluation

Python's `and`/`or` use short-circuit logic and return the *actual value*, not just True/False.  
Predict what each expression returns, then uncomment and verify.

```python
# Predict the returned value for each, then uncomment and run:

# 5 and 3
# 0 and 3
# "" or "default"
# "hello" or "world"
# None or 0 or [] or "finally"
# 1 and 2 and 3

# Question: How could you use `or` to set a default value for a variable that might be empty?
```

## Exercise 3: Membership tests & combined logic

Given student data, write boolean expressions (no loops needed) to answer each question.

```python
students = ['Maria Papadopoulou', 'Nikos Georgiou', 'Elena Konstantinou', 'Dimitris Alexiou']
passing_grades = {'Maria Papadopoulou': 7, 'Nikos Georgiou': 4, 'Elena Konstantinou': 9}

# 1. Is 'Alexiou' in students? Why does this return False even though Dimitris Alexiou is in the list?

# 2. Check if 'Nikos Georgiou' is in students AND his grade is >= 5

# 3. Check if 'Dimitris Alexiou' is in students but NOT in passing_grades

# 4. Write an expression: is Elena's grade above 8 OR Maria's grade above 8?
```

## Exercise 4: Lexicographic comparisons (tricky!)

Strings and lists are compared element by element (lexicographically), **not** by length.  
Predict each result and write a brief explanation. Then run to verify.

```python
# Predict True or False and explain why:
print("apple" < "banana")       # ? 
print("9" > "10")               # ? (tricky!)
print("19999" < "2")            # ? 
print([1, 30] > [1, 2, 3, 4])   # ?
print([1, 2] < [1, 2, 0])       # ?

# When would this matter in real code? Write a brief comment.
```

## Exercise 5: Truthiness and defaults

Use `bool()` and short-circuit evaluation to solve a practical problem.

```python
# A web form returns these values (empty string means user left the field blank):
user_name = ""
user_email = ""
user_phone = "6912345678"

# 1. Using `or`, write an expression that returns the first non-empty contact info
#    (check email first, then phone). Assign to `contact`.

# 2. Write a boolean expression that checks: user filled in their name AND
#    at least one of (email or phone) is provided.

# 3. What does bool([]) return? What about bool([0])? Why?
```

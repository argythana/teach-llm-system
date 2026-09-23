<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_05_boolean_conditions_control_statements/practice_exercises/practice_lec_05c_if_condition.ipynb @ 0cc874704aaa -->

# Lecture 05c Practice: `if` / `elif` / `else` Conditions

**Topics:** Conditional logic, nested conditions, pseudocoding, combining conditions.

---

## Exercise 1: Grade classifier with edge cases

Ask the user for a grade (0-10).  
- Validate that the input is a number and within range. If not, print an error message.  
- Classify: 9-10 "Excellent", 7-8 "Very Good", 5-6 "Pass", below 5 "Fail".  
- If the grade is exactly 4.5-4.9, add: "So close! Consider asking for a re-examination."

```python
# Your code here
```

## Exercise 2: Shipping cost calculator

A store calculates shipping based on order total and destination:  
- Domestic orders over 50€: free shipping  
- Domestic orders under 50€: 5€ shipping  
- International orders over 100€: 10€ shipping  
- International orders under 100€: 20€ shipping  
- If the item is fragile, add 3€ to any shipping cost  

Write pseudocode first, then implement. Print the total cost (order + shipping).

```python
order_total = 75.0
destination = "international"  # "domestic" or "international"
is_fragile = True

# Pseudocode:

# Code:
```

## Exercise 3: Leap year checker

A year is a leap year if:  
- Divisible by 4, **except**  
- Years divisible by 100 are **not** leap years, **except**  
- Years divisible by 400 **are** leap years.  

Test with: 2000 (leap), 1900 (not), 2024 (leap), 2023 (not).

```python
year = int(input("Enter a year: "))

# Your code here
```

## Exercise 4: Rock, Paper, Scissors

Write pseudocode, then implement:  
- Computer picks randomly from `['rock', 'paper', 'scissors']`  
- Ask user for their choice  
- Determine winner and print result  
- Handle invalid input from user

```python
import random

# Pseudocode:

# Code:
```

## Exercise 5: Day planner

Given the weather, day of week, and energy level, suggest an activity:  
- If it's a weekend AND sunny AND energy is high: "Go hiking"  
- If it's a weekend AND sunny AND energy is low: "Sit at a café"  
- If it's a weekend AND rainy: "Watch a movie"  
- If it's a weekday AND energy is high: "Study at the library"  
- If it's a weekday AND energy is low: "Rest, study tomorrow"  

Use nested `if` statements (not a single giant `if/elif` chain).

```python
weather = "sunny"      # "sunny" or "rainy"
day_type = "weekend"   # "weekend" or "weekday"
energy = "high"        # "high" or "low"

# Your code here (use nested if, not flat elif)
```

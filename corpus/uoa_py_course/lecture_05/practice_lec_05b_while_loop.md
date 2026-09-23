<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_05_boolean_conditions_control_statements/practice_exercises/practice_lec_05b_while_loop.ipynb @ 0cc874704aaa -->

# Lecture 05b Practice: `while` Loops

**Topics:** `while` syntax, `break`, `continue`, counters, infinite loops with exit conditions.

---

## Exercise 1: Collatz conjecture

The Collatz sequence starts with any positive integer and applies these rules repeatedly:
- If the number is even, divide by 2
- If the number is odd, multiply by 3 and add 1

The conjecture says this always reaches 1.  
Write a `while` loop that takes a starting number and prints each step until reaching 1. Also count the steps.

```python
number = 27  # Try different starting values

# Your code here
```

## Exercise 2: Input validation loop

Write a program that keeps asking the user to enter a number between 1 and 10.  
- If they enter something that's not a number, print a warning and ask again.  
- If the number is out of range, tell them whether it's too high or too low.  
- When valid, print the number and exit.  

Hint: `str.isdigit()` checks if a string contains only digits.

```python
# Your code here
```

## Exercise 3: Accumulator with a stop condition

A user enters expenses one by one.  
- Keep a running total.  
- After each entry, show the current total.  
- If the total exceeds a budget of 100, warn them and stop.  
- If they type `done`, stop and show the final total.

```python
budget = 100

# Your code here
```

## Exercise 4: Fibonacci with a twist

Use a `while` loop to generate Fibonacci numbers. But instead of printing them all:  
- Skip (don't print) any Fibonacci number that is even (use `continue`).  
- Stop when you've printed 10 odd Fibonacci numbers.  
- Count how many total Fibonacci numbers you generated to find those 10.

```python
# Your code here
```

## Exercise 5: PIN lockout system

Simulate a PIN entry system:  
- Correct PIN is `"1234"`.  
- User gets 3 attempts.  
- After each wrong attempt, tell them how many tries remain.  
- After 3 wrong attempts, print "Card blocked" and exit.  
- If correct, print "Access granted" and show which attempt succeeded.  

Write pseudocode as comments first, then implement.

```python
# Pseudocode:

# Code:
```

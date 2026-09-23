<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_05_boolean_conditions_control_statements/practice_exercises/practice_lec_05d_for_loop.ipynb @ 0cc874704aaa -->

# Lecture 05d Practice: `for` Loops

**Topics:** `for` with `range()`, iterating sequences, `break`/`continue`, nested loops, accumulation patterns.

---

## Exercise 1: Word analysis

Given a sentence, use a `for` loop to:  
1. Count the number of vowels (a, e, i, o, u — case-insensitive).  
2. Count the number of consonants.  
3. Find the longest word (split by spaces).  

Print all three results.

```python
sentence = "The University of the Aegean is located on several Greek islands"

# Your code here
```

## Exercise 2: FizzBuzz

Classic programming exercise. For numbers 1 to 30:  
- Divisible by both 3 and 5: print `"FizzBuzz"`  
- Divisible by 3 only: print `"Fizz"`  
- Divisible by 5 only: print `"Buzz"`  
- Otherwise: print the number  

**Important:** The order of your conditions matters. Think about why.

```python
# Your code here
```

## Exercise 3: Filtering and transforming data

Given a list of temperatures in Fahrenheit, use a `for` loop to:  
1. Convert each to Celsius: `C = (F - 32) * 5/9`  
2. Store only temperatures above 20°C in a new list.  
3. Print the filtered list and the average of the filtered temperatures.

```python
temps_f = [32, 50, 68, 77, 86, 95, 59, 41, 104, 72]

# Your code here
```

## Exercise 4: Pattern with nested loops

Print this number pyramid (5 rows):  
```
    1
   1 2
  1 2 3
 1 2 3 4
1 2 3 4 5
```
Hint: For each row, print leading spaces then the numbers.

```python
rows = 5

# Your code here
```

## Exercise 5: Prime finder

Find all prime numbers between 2 and 50.  
- A number is prime if it's only divisible by 1 and itself.  
- Use a `for` loop with a nested `for` loop to check divisibility.  
- Use `break` to stop checking once you find a divisor.  
- Collect primes in a list and print it at the end.

```python
# Your code here
```

## Exercise 6: Student grade report

Given a dictionary of student grades, use a `for` loop to:  
1. Print each student and whether they passed (grade >= 5) or failed.  
2. Calculate and print the class average.  
3. Print the names of students who scored above the average.

```python
grades = {
    'Maria': 8.5,
    'Nikos': 4.0,
    'Elena': 9.2,
    'Dimitris': 5.5,
    'Anna': 3.8,
    'Kostas': 7.0
}

# Your code here
```

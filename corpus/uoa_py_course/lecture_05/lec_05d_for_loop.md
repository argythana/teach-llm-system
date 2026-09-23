<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_05_boolean_conditions_control_statements/reading_material/lec_05d_for_loop.ipynb @ 0cc874704aaa -->

# Lecture 05d: Control Flow Structures: loops, conditions. The `for` loop.

[Intro to `for` loop statements](https://docs.python.org/3/tutorial/controlflow.html#for-statements)


Basic Control Flow statements (control structures): while, if, for, try.   
Used to: make decisions and direct the order of program execution.   
Loops could be defined as conditional iterations. Loops are iterations that are bounded by some specified conditions.


**Topics**:   

a) Syntax notation:  
    
* indented blocks.  
    
* `:` sign at end of statement line.  

b) Conditions: Use for all kind of boolean operations such as comparisons, membership tests.   

c) Within structures ["inner" statements](https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops): `break`, `continue`.   

d) The `else`, `elif` "*clauses*".  

e) The [`pass` statements.](https://docs.python.org/3/tutorial/controlflow.html#pass-statements)

f) Nested control flow statements: `nested loops`, `nested conditions`.  


More reading about 

**Most common control flow errors are logical errors. Be mindful:**  

        a) Desciptive control flow assignment.  
        b) Controling logically for all the possible cases when dealing with multiple conditions.  
        c) Position of statements.

THE MOST IMPORTANT ISSUE TO REMEMBER: **position and order of statements in structures.**

**Advanced reading** --> Click below to become python Ninja.
1. `match` [statements](https://docs.python.org/3/tutorial/controlflow.html#match-statements)
2. `exception` [handling](https://docs.python.org/3/tutorial/errors.html)
3. `compound` [statements, try, with](https://docs.python.org/3/reference/compound_stmts.html)

Extra concept in this lecture: **Pseudocoding**:  
Always has been a necessary prerequisite skill for optimal coding.  
Even more important skill in the era of AI coding assistants.

---

## `for` loops

Reminder:  
The range function paremeters: range(start, stop, step)

Be careful and use:  
* ```range()``` to iterate N times.    
* ```for``` to iterate over all objects, all items of a sequence.  


To iterate:  
>  don't use reserved names,  
> use intuitive, descriptive names.

```python
# consice help in interactive python notebooks.
range?
```

```python
user_name = input('Enter your name:\n\n')

# range start from 0, stop at 4, default step=1
for i in range(4):  # range is NOT inclusive TOO !!!
    # print(i)
    print(user_name)
```

```python
# range start from 0, stop at 4, default step=2
for i in range(0, 4, 2):
    print(user_name)
```

#### Use the word "iterations" instead of "i"

```python
user_name = input('Enter your name: ')

# start from 0, stop at 4, default step=1
for iterations in range(4):
    print(f'{user_name}')
```

#### Assign `for` loop repetitions to a descriptive name

```python
example_string = 'carefull with spaces'

for character in example_string:  # use letter instead of i
    print(character)
```

```python
# Assign `for` loop iterations to a descriptive name
example_list = ['a' , 4, 'john', 'jill']

for item in example_list:
    print(item * 2)
```

```python
# Find even or odd numbers
for number in range(2, 10):
    print(number)
    if number % 2 == 0:
        print("Found an even number", number)
    else:
        print("Found an odd number", number)
```

```python
# Find even or odd numbers
for number in range(2, 10):
    print(number)
```

```python
number
```

```python
if number % 2 == 0:
    print("Found an even number", number)
else:
    print("Found an odd number", number)
```

```python
random_mame = "Jonh"

for character in random_mame:
    print(character)
```

```python
if "o" in random_mame:
    print("Found an o in the name")
```

```python
while True:
    user_input = input("Enter a name: ")
    if "o" in user_input:
        print("Found an o in the name")
        break
    else:
        print("No o found in the name")
        break
```

### Continue statement   
[continue](https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops)

Continue with the next cycle of the nearest enclosing loop.

```python
# Find even or odd numbers
# Check the difference wth the script above
for num in range(2, 10):  # 2
    if num % 2 == 0:   # True
        print("Found an even number", num)
        # continue passes control to the beginning of the loop that it belongs to.
        continue  # the nearest enclosing loop is the if:
    print("Found an odd number", num)
```

```python
# Find even or odd numbers
# if continue is in the wrong block:
# This is a total ABSOLUTE Mess

for num in range(2, 10):  # 2
    if num % 2 == 0:   # True
        print("Found an even number", num)
        # continue passes control to the beginning of the loop that it belongs to.
    continue  # the nearest enclosing loop is the for:
    # when the flow ot the script comes to continue, it passes control to the for and iteration continues.
    # the print and odd is never executed, and the loop finishes always ignoring it
    print("Found an odd number", num)
```

```python
# Find even or odd numbers. WRONG.
# Without continue, the loop is a mess.
for num in range(2, 10):
    if num % 2 == 0:
        print("Found an even number", num)
        continue  # commented "continue" to see difference
        # without continue the inner (nested) loop finishes,
        # then goes to the next statement of the outer loop
        # then executes the last print everytime.
        # then goes to beggining to iterate over the next number.
    print("Found an odd number", num)
```

## Assignment, Practice Exercise

**Create a guessing game. Will be graded with 10 points.**  
**Goal: Show pseudo coding and test learning so far.**


Create a random integer number from 1 to 100. (Hint: try with a range from 1 to 20 first)  
Ask user to guess the random number that was created.  
Store the number of users' tries in a counter.  

If user guess < number: inform users that their guess is lower than the number and ask user again.  
If user guess > number: inform users that their guess is higher than the number and ask user again.   
If user guess is equal to the number, inform user about the number of tries it took and then congratulate user.   

Important:   
Try to make different versions using AI assistants.  
If you have more than one solutions, include your pseudocoding variations in the solutions file.

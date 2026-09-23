<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_05_boolean_conditions_control_statements/reading_material/lec_05b_while_loop.ipynb @ 0cc874704aaa -->

# Lecture 05b: Control Flow Structures: loops, conditions. The `while` loop.

[Intro to `while` loop statements.](https://docs.python.org/3/tutorial/introduction.html#first-steps-towards-programming)


Basic Control Flow statements (control structures): while, if, for.
Used to: make decisions and direct the order of program execution.   
Loops could be defined as conditional iterations. Loops are iterations that are bounded by some specified conditions.


**Topics**:   

a) Syntax notation:  
    
* indented blocks.  
    
* `:` sign at end of statement line.  

b) Conditions: Use for all kind of boolean operations such as comparisons, membership tests.   

c) Within structures ["inner" statements](https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops): `break`, `continue`.   

d) The [`pass` statements.](https://docs.python.org/3/tutorial/controlflow.html#pass-statements)

r) Nested control flow statements: `nested loops`, `nested conditions`.


More reading about 

**Most common control flow errors are logical errors. Be mindful:**  

        a) Descriptive control flow assignment.  
        b) Controling logically for all the possible cases when dealing with multiple conditions.  
        c) Position of statements.

THE MOST IMPORTANT ISSUE TO REMEMBER: **position and order of statements in structures.**

**Advanced reading** --> Click below to become python Ninja.
1. `match` [statements](https://docs.python.org/3/tutorial/controlflow.html#match-statements)
2. `exception` [handling](https://docs.python.org/3/tutorial/errors.html)
3. `compound` [statements, try, with](https://docs.python.org/3/reference/compound_stmts.html)

---

## `while` loops.

`while`: Repeat execution (loop) **as long as** an expression is true.  

`break`: terminates enclosing loop, control value keeps current value.  
`continue`: continue enclosing loop => go to start of loop.  
`pass`: do nothing.   
Stop infinite loops with "Interrupt kernel" in notebook, or "Ctrl + c" in python.

#### Example with print inside the loop, but in *wrong* position .  
In the example below, the print statement does not print the final values.  
It prints the values before the operations in each iteration of the loop.

```python
# Fibonacci series: the sum of two elements defines the next.
a = 0
b = 1

while a < 20:
    print(a, b)   # Print before the operations.
    a = b  # a becomes 1
    b = a + b  # becomes 2 
    #a, b = b, a + b
    # break  # Uncomment to break the loop after the first iteration.
```

```python
# The final value of a.
a
```

```python
# The final value of b.
b
```

#### Example with print inside the loop, but in the *correct* position.  
In the example below, the print statement prints the final values.  
It also prints the value of a and of b after each iteration of the loop.

```python
a = 0
b = 1

while a < 20:
    a = b  # a becomes 1
    b = a + b  # becomes 2 
    print(a, b)   # Print after the operation
    #a, b = b, a + b
    # break  # Uncomment to break the loop
```

#### Example with print outside of the loop.  
In the example below, the print statement prints the final values.  
It prints the value of a and of b only once, outside of the loop.

```python
a = 0
b = 1

while a < 20:
    a = b  # a becomes 1
    b = a + b  # becomes 2

print(a, b)
```

#### Example of infinite while loop.
Stop using jupyter GUI square button: -> "interrupt the kernel".  
Or use Tab Menu: Kernel -> Interrupt kernel.  
Stop with "Ctrl + c".  

If you uncomment the code below and run it, you will enter an infinite loop until all your RAM is utilised. Then your PC will crash.

```python
a = 1

# while a < 2:
#     print(a)
```

#### Example of print position inside and outside of a loop.

```python
a = 1

while a <= 4:
    print("In the loop, before the operation: ", a)
    #break
    
    a = a + 1   #  a += 1
    
    print("Still in the loop, after the operation:", a)
    #break


print("This is out of the loop: ", a)
```

#### Trivial example of a nested loop.   
In this case, the nested loop finishes before the outer loop.    
The outer loop continues its operations until its condition is met.  
If you "comment out" the `break` statement in the nested loop, the nested loop will become infinite because there is no operation in it to increase the value of a.

```python
a = 1


while a <= 9:
    print(a, ": in outer loop. Before the addition operation.")
    a += 1 # notation same as: a = a + 1
    # print(a) # Position returns different results !
    while a <= 3:
        print(a, ": in Nested loop. After the addition operation.")
        break # Comment to have infinite loop. Stop with "Ctrl + c"
        

print(f"Out of loop {a}")
```

#### Boring example of a nested loop.   
In this case, the nested loop finishes after the outer loop.    
The nested loop continues its operations until its condition is met.

```python
a = 1


while a <= 10:
    print(a, ": in outer loop. Before the addition operation.")
    a += 1  # notation same as: a = a + 1
    # print(a) # Position returns different results !
    while a <= 50:
        print(a, ": in nested loop. Before the multiplication operation.")
        a *= 2  # notation same as: a = a * 2
        print(a, ": in Nested loop. After the multiplication operation.")
        

print(f"Out of loop {a}")
```

#### `pass` statement example

```python
a, b = 0, 1  # Remember: you can assign multiple variables in one line.


while a < 20:
    a, b = b, a + b
    print(a, end=' ')
    if a == 5:
        print("five",  end=' ')
        pass  # Placeholder for future code. Do nothing.
```

#### Password example with infinite `while` loop until correct input.

```python
correct_pwd = "date"
user_pwd = None  # None is used as a placeholder for "no value". It is not the same as an empty string "".
# user_pwd = input("Enter password: ")

while correct_pwd != user_pwd:
    user_pwd = input("Enter correct password: ")

print("Succesfully logged in.")
```

```python
# What will be the value of this variable after we run the script above?
# Think, then uncomment the line below and run this cell.
# user_pwd
```

```python
correct_pwd = "date"
user_pwd = None
# user_pwd = input("Enter password: ")
counter = 0

while correct_pwd != user_pwd:
    user_pwd = input("Enter correct password: ")
    counter = counter + 1
    if user_pwd != correct_pwd:
        print(f"Attempt {counter}: Incorrect password.")
    if counter == 3:
        print("Too many attempts. Try again later.")
        break

print("This is where break statement takes us.")

if user_pwd == correct_pwd:
    print("Succesfully logged in.")
```

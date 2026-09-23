<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_06_define_functions/reading_material/lec_06b_define_functions.ipynb @ 0cc874704aaa -->

# Lecture 6b: User-defined functions.
 
> Purpose: decomposition, abstraction, reusability.  
> Modularity, modular development.  
> Syntax: `def`, :, indented blocks, `docstrings`, `return` statement.  
> Parameters  
> Arguments  

* When something is repeated in the code -> define a function. Don't Repeat Yourself (DRY).
* When a task needs to be repeated -> define a function. DRY.


### Suggested (suggested means absolutely necessary) videos to watch after class:

#### A. When and how to write functions: [The Mental Game of Python - Raymond Hettinger 2019](https://www.youtube.com/watch?v=UANN2Eu6ZnM&t=298s)

My favourite quote on what functions are: *"new words to make computers easier to use."*

Make sure you understand the stragegies presented right from the start (from 2:50 onwards).  
For `inheritance`, see from 36:00 to 46:00.
Quote: "build classes independently and let inheritance discover itself. Should not plan too many steps in advance: "Combinatory explosion"

Watch this part at least twice to think when to define functions: from 47:45 to 01:01:05.
Quote:
> "Repeat tasks manually until patterns emerge. Then, move to a function.
> If `def` is the first word, you make your life worse."

**Tips to get you started when defining a function:**
* First write the code line by line, as a script, when it works, define is as a function.
* Use debugging statements e.g. print(), within the function, to test and inspect the intermediate results.

#### B. Simple tips to properly define functions: [The ultimate guide to writing functions.](https://www.youtube.com/watch?v=yatgY4NpZXE)

### Basic syntax of definition and calling
1. Define a function:
* `def function_name(parameters):`
* `docstrings` = documentation string = output of help(function_name)
* Function `body` as indented block: always 4 blank spaces.
    In function body -> state what to do.
* Return the output: a `return` statement. Return just the value (not the name) to the caller.

2. Call a function:
* `function_name(arguments)`
* `Parameters` = "names" in definition.
* `Arguments` = "values" in call.

3. Assign the result of the function to a variable:
* `value = function_name(arguments)`

## 1. Simple function without parameters

```python
# a very useless function.

def teach_user_defined_docstrings():
    """
    This is a useless function. Its purpose is to waste your time.
    """
    print("The most useless function ever, was written by Thanasis Argyriou.")

    teacher_character = "boring and not funny"

    return teacher_character
```

```python
print(teach_user_defined_docstrings.__doc__)
```

```python
teach_user_defined_docstrings()
```

```python
# CAREFUL: help() expects the function OBJECT, not the function CALL.
# help(teach_user_defined_docstrings())  # Wrong! Calls the function first, then runs help() on the returned string.
help(teach_user_defined_docstrings)       # Correct! Passes the function object to help().
```

## 2. Simple function with single parameter without default value

```python
def simple_msg_without_default_value(message):
    message = message.upper()

    # print(message)
    return message
```

```python
# Call the function — the returned value is displayed but not saved.
simple_msg_without_default_value("python makes everything uppercase")
```

```python
# Call the function and save the returned value to a variable.
modified_msg = simple_msg_without_default_value("save this for later")
```

```python
modified_msg
```

## 3. Simple function with single parameter with default value

```python
# Recommended style: no blank space between argument and value in definition.
def simple_msg(message="Python is great."):
    """Print the message.
    default value=Python is great."""
    print(message)
```

```python
# call with default value
simple_msg()
```

```python
# Call using a new value, not the default value.
simple_msg(message='Python is boring!')
```

```python
# function doc
help(simple_msg)
```

## 4. Simple function with multiple parameters and one default value

```python
# 3 args, 2 non-default, 1 default value
# assign value to positional args before keyword (else => SyntaxError)
# output  order not the same thing as input order!
def message_from_to(sender, receiver, message='Please study'):
    """
    Print the message from the sender to the receiver.
    """
    print(sender, message, receiver)  # Very Bad order, still works.
```

```python
message_from_to('Boring Teacher:', 'dear Students,')
```

```python
message_from_to('Thanasis', 'students', 'kindly asks you to study, dear')
```

```python
# Both work syntatically. Which is "wrong", why ?
message_from_to('Students:', 'Thanasis', 'No way, this is so boring!')
```

```python
message_from_to('Students:', 'No way this is so boring!', 'Thanasis')
```

## 5 . Simple function with multiple parameters, one default value and conditions

```python
# 3 args, 2 non-default, 1 default value
# assign value to positional args before keyword (else => SyntaxError)
# output  order not the same thing as input order!
def message_from_to(sender, receiver, message='Please study'):
    """
    Print the message from the sender to the receiver.
    Deliver different messages to different receivers.
    """
    if receiver == 'students':
        # message = message + " ,please, please, please."
        message = "Do what you really want,"

    elif receiver == 'Kostas':
        message = "Please ask more questions,"

    print(sender, message, receiver)  # Very Bad order, still works.
```

```python
message_from_to('Thanasis:', 'students')
```

```python
message_from_to('Thanasis:', 'students')
```

## 6. Simple function to calculate order cost
This function works but is defined in a bad way.

```python
def calc_order(q, items, p):
    """
    Calculate the total cost of an order.
    
    Parameters:
    q (int): Quantity of items.
    items (str): Name of the items.
    p (float): Price per item.
    
    Returns:
    None
    """
    total = q * p
    print(f'{q} {items}, cost {total} \n')  # Match definition order.
```

```python
# Order of arguments value assignemnt matches definition,
calc_order(3, 'books', 10)
```

```python
# Order of values passed in function, call in this case DOES NOT match definition, flexible order.
calc_order(items='books', p=10, q=3)
```

#### How this function could be defined for better readability
**Recommendation:** Provide general purpose, descriptive and intuitive names for functions and variables.

```python
def calculate_order_cost(quantity, item_name):  # no object type specified
    """
    Calculate the cost from an order of N items.
    Prices are determined from a menu list, not from the user.
    Arguments:
        quantity: number of items
        item_name: name of the item
    Returns:
        total_cost: total cost of the order,
    """
    prices = {'books': 10, 'pens': 2, 'pencils': 1}  # In practice this is from a database, not inside the function.

    total_cost = quantity * prices[item_name]
    print(f'{quantity} {item_name} cost {total_cost} euros.\n')

    return total_cost
```

```python
calculate_order_cost(5, 'books')
```

## 7. Modular development of functions.

Reminders:

* Functions are used to decompose a problem into smaller parts.
* Each function should do one thing only and do it well.
* When facing a complex problem, we split it in smaller tasks, we define functions for each small task.
* We can call these smaller functions from another function.

```python
# no reference to math module when sqrt() is called.
from math import sqrt

# This way to use sqrt you need: math.sqrt()
# import math


def calculate_hypot(side_a, side_b):
    """
    Calculate the hypotenuse.
    Params:
        side_a: length of side a
        side_b: length of side b
    Returns:
        hyp: length of hypotenuse
    """

    hyp = sqrt(side_a*side_a + side_b*side_b)  # hyp = local variable

    return hyp  # Only the result is returned by the function back to the caller. Not the "name=value" pair.
```

```python
# pass two positional arguments in the function call.
calculate_hypot(6, 8)
```

```python
# pass two keyword arguments in the function call.
calculate_hypot(side_a=6, side_b=8)
```

```python
# Define a second function that uses this function:
def hypot_calculator_with_user_input():
    """
    Ask user for triangle sides and print hypotenuse. Input should be numbers.
    """
    a = float(input("Insert a number for side a: "))
    b = float(input("Insert a number for side b: "))

    # call function calculate_hypot() inside this function.
    hypotenuse = calculate_hypot(a, b)

    print ("Hypotenuse = ", hypotenuse)
```

```python
hypot_calculator_with_user_input()  # No arguments needed, user input is used.
```

```python
# TypeError: hypot_calculator() takes 0 positional arguments but 2 were given
hypot_calculator_with_user_input(6, 8)
```

## 8. Advanced Reading: Functions with arbitrary number of args *

```python
def multiply(*my_arguments, end="hi!"):
    """Multiply all positional arguments together, printing each running product."""
    s = 1  # neutral factor in multiplication (identity element).
    for number in my_arguments:
        s *= number  # this syntax means s = s * number
        print(s)
    print(end)

multiply(1, 2, 3, 4, end="This is the end of the function")
```

```python
# Take your time to see why we get this result!
# Hint: the list object [1, 2, 3, 4, 5, 6] is treated as a SINGLE argument.
# Python multiplies the list by 2 (list * int = list repeated), then by 3, etc.
my_example_list = [1, 2, 3, 4, 5, 6]
multiply(my_example_list, 2, 3, 4, 5)
```

```python
# A string is also a valid argument — Python will try to multiply it.
# string * int = string repeated, so s *= "brick" repeats the string.
my_example_string = "my_imagination_is_that_of_a_brick"

multiply(my_example_string)
```

```python
# A different approach: iterate INSIDE each argument (which must be iterable).
# Bug: the counter 'i' is initialized OUTSIDE the outer loop, so it keeps counting
# across all arguments instead of resetting for each one.
def multiply_iterable_objects(*my_object):
    i = 1  # counter initialized ONCE — this is the bug.
    print(my_object)
    for iterable_object in my_object:
        for item in iterable_object:
            i = i + 1
            item = i * item
            print(item)
```

```python
my_example_string = "brick"

multiply_iterable_objects(my_example_string)
```

```python
my_example_list = [1, 2, 3, 4, 5, 6]

multiply_iterable_objects(my_example_list)
```

```python
# Can you spot the mistake of the function above in this example?
# Hint: the counter i does NOT reset between the string and the list.
# So the list items get multiplied by the WRONG values (i continues from where it left off).
my_example_string = "brick"
my_example_list = [1, 2, 3, 4, 5, 6]

multiply_iterable_objects(my_example_string, my_example_list)
```

```python
# Attempt 2: move the counter increment to the OUTER loop.
# Can you spot the remaining problem?
# Hint: now i increments once per iterable_object, not once per item.
# All items within the same object get the same multiplier.
def multiply_iterable_objects(*my_object):
    i = 1
    print(my_object)
    for iterable_object in my_object:
        i = i + 1
        for item in iterable_object:
            item = i * item
            print(item)


multiply_iterable_objects(my_example_string, my_example_list)
```

```python
# Correct version: reset the counter for each iterable object.
# Now i counts the position of each item WITHIN its own iterable.
def multiply_iterable_objects(*my_object):
    print(my_object)
    for iterable_object in my_object:
        i = 0  # reset counter for each iterable.
        for item in iterable_object:
            i = i + 1
            item = i * item
            print(item)

multiply_iterable_objects(my_example_string, my_example_list)
```

## 9. Advanced Reading: Namespace, global and local scope in variables

[Python scopes and namespaces](https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces).

* **Local scope:** a variable defined inside a function is available only within that function.
* **Global scope:** a variable defined at the top level of the module is available everywhere in the module.

A function can **read** global variables, but cannot **modify** them (unless you use the `global` keyword, which is generally discouraged).

```python
# Example: local vs global scope

greeting = "Hello"  # global variable

def make_farewell():
    farewell = "Goodbye"  # local variable — exists only inside this function
    print(farewell)

make_farewell()

# This will work — greeting is global, visible everywhere:
print(greeting)
```

```python

# This will raise a NameError — farewell is local to make_farewell():
print(farewell)
```

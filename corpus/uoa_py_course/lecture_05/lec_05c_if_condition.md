<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_05_boolean_conditions_control_statements/reading_material/lec_05c_if_condition.ipynb @ 0cc874704aaa -->

# Lecture 05c: Control Flow Structures: loops, conditions. The `if` statement.


[Intro to `if` statements](https://docs.python.org/3/tutorial/controlflow.html#if-statements)






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

e) The `pass` [statement](https://docs.python.org/3/tutorial/controlflow.html#pass-statements)

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

## `if` conditional statements
There can be zero or more `elif` parts, and the `else` part is optional.   
The keyword `elif` is short for ‘else if’, and is useful to avoid excessive indentation.

```python
my_shopping_list = [
    'apples', 'coffee', 'pasta', 'juice', 'chocolate', 'yoghourt'
    ]

num_of_items = len(my_shopping_list)
num_of_items
```

### Single condition.
Comment or comment out the `num_of_missing_items` to see the difference.

```python
# num_of_missing_items = 2 in my refrigerator.
# num_of_missing_items = 5

# If, else syntax, multiple statements, single condition.
num_of_missing_items = int(input("How many items are missing from the fridge? "))

if num_of_missing_items > 3:
    print(f"Go to Supermarket. It is worth the extra time for items = {num_of_missing_items}")
else:
    print(f"Go to Minimarket for {num_of_missing_items} missing items.")  # 1st else statment
```

```python
budget = 100
oil_price = 50

if oil_price < budget:
    print(f"{oil_price}")
    buy_amount = budget // oil_price
    print(f"Buy {buy_amount} liters of oil.")
else:
    print(f"We are missing {oil_price - budget} dollars to buy oil.")
```

```python
if
```

### Multiple Conditions

```python
## Multiple Conditions. Use `elif`
a = 10
guess = int(input('Guess number: '))  # convert to int

if guess == a:
    print('Got it.')
elif guess == a - 1 or guess == a + 1:
    print('Missed by 1.')
elif (a - 5) == guess or guess == (a + 5):
    print('Missed by 5.')
#elif guess > a - 5 and guess < a + 5:  # guess > 5 and guess < 15
elif 5 < guess < 15:  # This has the same result as the line of code above.
    print('Missed by less than 5.')
else:  # use `else` always in last statement
    print('Missed by more than 5.')
```

### Examples of nested conditions.
In these examples the nested conditions are "evaluated" only if the outer condition is met.

```python
# Nested conditions execution
my_shopping_list = ['apples', 'coffee', 'pasta', 'juice']

# my_shopping_list.append('bananas')   # Remember in-place operations.

my_shopping_list
```

```python
# if bananas not in list, will do nothing
if 'bananas' in my_shopping_list:
    print('Dont forget to buy bananas too')
    # nested condition evaluated only if outer condition is True
    if 'apples' in my_shopping_list: 
        print('Dont forget to buy apples too.')
    else:
        print(f"Buy {my_shopping_list}")
```

```python
my_shopping_list = ['coffee', 'pasta', 'juice']
```

```python
if 'bananas' in my_shopping_list:
    print('Dont forget the bananas')
elif "bananas" not in my_shopping_list:
    my_shopping_list.append('bananas')
    print('Dont forget the bananas')
    # nested condition evaluated only if outer condition is True
    if 'apples' in my_shopping_list: 
        print('Dont forget the apples too')
    else:
        print(f"Buy {my_shopping_list}")
```

### Pseudo coding: Best Practice to make scripts and functions

Set a pwd value.  
Ask user for pwd.  
Count number of tries.  
If pwd is not correct ask again for three times.  
If number of wrong tries is 3 stop execution.  
If correct pwd is entered, print a success message and show number of tries.

### Combine `while` and `if` (a pwd example script).

```python
pwd = "date"
enter_pwd = input("Enter password: ")
number_of_tries = 1

# loop 3 times only, as long as user does not enter the correct pwd.
while pwd != enter_pwd:
    enter_pwd = input("Enter correct password: ")
    number_of_tries += 1

    if number_of_tries == 3 and pwd != enter_pwd:
        print("Your account is locked.")
        break
    
if pwd == enter_pwd:
    print(f"Succesfully logged in after {number_of_tries} tries.")
```

#### The same enter password example. Different coding by GitHub AI copilot.  
The prompt was the pseudocoding lines above.

```python
# Set a password value
password = "secret"

# Initialize a counter for the number of tries
tries = 0

# Loop until the user enters the correct password or has tried 3 times
while tries < 3:
    # Ask the user for the password
    user_password = input("Enter the password: ")

    # Check if the entered password is correct
    if user_password == password:
        print("Success! You entered the correct password.")
        print("Number of tries:", tries + 1)
        break
    else:
        print("Incorrect password.")
        tries += 1

# If the user failed to enter the correct password after 3 tries, stop execution
if tries == 3:
    print("You have entered the wrong password 3 times. Execution stopped.")
```

#### The same enter password example. Different coding by GitHub AI copilot.  
The prompt was: rewrite the code with fewer lines of code.

```python
password = "secret"

for tries in range(3):
    if input("Enter the password: ") == password:
        print(f"Success! You entered the correct password in {tries + 1} tries.")
        break
else:
    print("You have entered the wrong password 3 times. Execution stopped.")
```

### Another pwd example  
Similar as above but instead of breaking the script after 3 wrong tries:  
Endless loop until correct password.  
Give a hint and ask user to try again after 3 wrong tries.    
Use three counters:    

    a) number of tries is reset to zero after 3 tries.  
    b) number of warnings. We warn the user once, after three tries.  
    c) the total number of tries.

```python
pwd = "date"
enter_pwd = input("Enter password: ")
number_of_tries = 1  # This is after the input() for a reason.
number_of_warnings = 0  # Counters are helfpuf and bug-friendly!
total_tries = 0

# endless loop until correctb pwd
while pwd != enter_pwd:
    enter_pwd = input("Enter correct password: ")
    number_of_tries += 1
    if number_of_tries == 3:
        number_of_warnings += 1
        total_tries = number_of_tries*number_of_warnings
        print(f"You tried {total_tries} times. Are you Ok? The pwd is written some lines above.")
        number_of_tries = 0    


print(f"Succesfully logged in after {total_tries+number_of_tries} tries. You got {number_of_warnings} warnings.")
```

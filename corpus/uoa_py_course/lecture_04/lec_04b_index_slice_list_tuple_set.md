<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_04_index_slice_str_list_dict/reading_material/lec_04b_index_slice_list_tuple_set.ipynb @ 0cc874704aaa -->

# Lecture 4b: Indexing, Slicing, Data structures: list, tuple, sets.

### Note: Slicing and Indexing works on **iterable** objects.
New words: indexing, slicing, lists,  **Mutable**, tuples. sets.
**SOS: in-place operations on mutable objets**


1. Lists. A data 'container'.
Iterable **Mutable** sequences, typically used to store collections of **homogeneous** items  
New word: Data containers.

> [built-in data type](https://docs.python.org/3/library/stdtypes.html#typesseq-list)   
>  [sequence data-type](https://docs.python.org/3/library/stdtypes.html#typesseq)  
> list(): [Built-in function](https://docs.python.org/3/library/functions.html#func-list)  

 
> Operations: [common for all sequence types](https://docs.python.org/3/library/stdtypes.html#typesseq-common) and [for mutable sequence types](https://docs.python.org/3/library/stdtypes.html#mutable-sequence-types).
> Indexing.  
> Slicing.  
> Basic functions on lists: len(), [sorted()](https://docs.python.org/3/howto/sorting.html#sortinghowto).  
> [List methods](https://docs.python.org/3/tutorial/datastructures.html) examples.  
  

2. Tuples. Data container.
**Immutable** sequences, typically used to store collections of **heterogeneous** data.

>[built-in data type](https://docs.python.org/3/library/stdtypes.html#tuple)  
> tuple() [built-in function](https://docs.python.org/3/library/functions.html#func-tuple)  
> Indexing.  
> Slicing.  
> Basic functions on Tuples [common for all sequence types](https://docs.python.org/3/library/stdtypes.html#typesseq-common).  


3. Sets. Data container.
>[built-in data type](https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset)  
----

## 1. Lists.
Denoted with square brackets: [ ]  
Iterable **Mutable** sequences, typically used to store collections of **homogeneous** items.

List **items** are separated by commas.  
May containt mixed data types (integers, strings, lists, objects, ...).
Not recommended for mixed data types.  
Ordered and indexex => allow for duplicates and item access.  
Mutable.  
Lists can be multiplied, added. The result is a list.
 
> [built-in data types](https://docs.python.org/3/library/stdtypes.html#typesseq-list)   
>  [sequence data-type](https://docs.python.org/3/library/stdtypes.html#typesseq)  
> list(): [Built-in function](https://docs.python.org/3/library/functions.html#func-list)  

 
> operations: [common for all sequence types](https://docs.python.org/3/library/stdtypes.html#typesseq-common) and [for mutable sequence types](https://docs.python.org/3/library/stdtypes.html#mutable-sequence-types).

```python
a = []  # Intialize and empty list and assign name 'a' to it.
```

```python
type(a)
```

```python
## Notated with square brackets, items separated by comma
strings_list = ['nikos', 'has', 'a', 'lovely', 'cat']
```

```python
type(strings_list)
```

```python
strings_list  # Can you see a list of strings?
```

#### Note: Lists may contain mixed data types. NOT RECOMMENDED.

Better use lists for **homogeneous** items!

```python
# A mixed list
mixed_list = ['nikos has', 3, 'cats but Schrödinger has two pieces of', 0.5, 'cats']

print(mixed_list, '\n')

print("Lists are NOT recommended for Mixed data types!")
```

```python
len(mixed_list)
```

```python
new_list = ['text', 2, 'makaronia']
```

```python
# List may contain other lists as an item. See Note above.
list_with_list = ['nikos', 'has', 3, 'cats', strings_list]

list_with_list
```

```python
list_with_list[-1][3][2]
```

```python
# len(list) to get the number of items
lenght_list_of_list = len(list_with_list)

print('length =', lenght_list_of_list, '\n')
```

```python
strings_list
```

#### Ask Copilot the following prompt:
> Our teacher recommends not to use lists for heterogenous items. But he also recommends validating his recommendations and searching more and questioning everything. You are an expert in teaching Python. Is 'not to use lists for heterogenous items' a solid recommendation? And what should I use for heterogenous itmes instead?

### 1.1 List Indexing.

```python
# List items are ordered, indexed, retain their  type
first_item = strings_list[0]  #! index from zero
first_item
```

```python
second_item = strings_list[1]  #! index from zero
second_item
```

```python
strings_list[0][2]
```

```python
first_item[2]
```

```python
## Since strings are ordered sequences as well:
third_letter_first_item = strings_list[0][2]

print(f'3rd letter of 1st item: {third_letter_first_item} \n')
```

```python
# Lists can be multiplied, added, result is a new list
strings_list*3
```

```python
# When adding lists, items from both lists are ordered
added_list = strings_list + ['and wants', 2, 'dogs']

added_list
```

```python
second_list = ['and wants', 2, 'dogs']
# strings_list.insert(1, second_list)
```

```python
strings_list
```

```python
result = strings_list[:2] + second_list + strings_list[2:]
print(result)
```


### 1.2. List Slicing.

```python
added_list[-4:-2]
```

## Mutability

```python
# Lists are mutable
# Make a copy to mutate it if you need to keep the  original list
to_mutate_list = strings_list.copy()
```

```python
to_mutate_list[0] = 'john'  # assing new value to first item

print(f'New value of 1st item):\n {to_mutate_list}\n')
```

```python
# Mutate list using slicing
to_mutate_list[0:2] = 'virus' # assing a letter to each item place, ! letters > items

print(f'All items can change:\n {to_mutate_list}\n')
```

#### Note: Only with practice you may identify which operations mutate a list.
List concatenation does not work **in-place**.

```python
added_list
```

```python
added_list + ["help"]  # not in place, just another view of operation result
```

```python
added_list
```

```python
## TypeError: can only concatenate list (not "str") to list

## CONCATENATE is a word you should know in programming

# added_list + "help"
```

#### NOTE: Slicing DOES NOT include the last index (Not inclusive operation).

### 1.3 List functions and methods.
> Basic functions on lists: len(), [sorted()](https://docs.python.org/3/howto/sorting.html#sortinghowto).  
> [List methods](https://docs.python.org/3/tutorial/datastructures.html) examples.   
> Mind the different syntax of object methods.

```python
strings_list = ['nikos', 'has a', 'lovely', 'cat']
```

### Note: append works in place!
**In place** means that the operation modifies the object itself without reassignment to the same name, without creating a new object. The object retains its identity (i.e., its memory address does not change).
This is called **mutating** the object.

### 1.4 Operations that modify a list "in-place".
Accurately speaking: operations that **"Mutate"** a list.

```python
# Initialize a list
my_list = [3243, 23, 2, 3]
your_list = [9, 10, 19]
```

```python
my_list + your_list
```

```python
my_list
```

```python
my_list + [45, 56, 45]
```

```python
my_list
```

```python
my_list = [3243, 23, 2, 3]
```

```python
# Sort the list (in-place)
my_list.sort()
```

```python
my_list
```

```python
# append a list to a list
my_list.append(your_list)
```

```python
my_list
```

```python
# Append an element (in-place)
my_list.append(4)
my_list
```

```python
# Extend the list with another list (in-place)
my_list.extend([5, 6])
print(my_list)  # Output: [1, 2, 3, 4, 5, 6]
```

```python
# my_list = my_list.extend([5, 6])
```

```python
type(my_list)
```

```python
print(my_list)  # Output: [1, 2, 3, 4, 5, 6]
```

```python
# Reverse the list (in-place)
my_list.reverse()
print(my_list)  # Output: [6, 5, 4, 3, 2, 1]
```

```python
# Remove an element by value (in-place)
my_list.remove(4)
print(my_list)  # Output: [6, 5, 3, 2, 1]
```

```python
# Pop an element by index (in-place)
my_list.pop(2)  # pop out the 3rd element, which is the list [9, 10, 19]
print(my_list)
```

```python
# Clear all elements from the list (in-place)
my_list.clear()
print(my_list)  # Output: []
```

#### Methods to split an object as alist and join a list as a string

```python
# Convert characters of string to list
string_example = "This is pythonic"
list_of_characters = list(string_example) # converts every charachter to list item
list_of_characters
```

```python
print('Convert characters to list:')
print(' ', string_example)
print(' ', list_of_characters)
```

```python
print('\nJoin items of list:')

# Join items of a list
joined_items_list = ''.join(list_of_characters) #join and use blank space as separator
```

```python
joined_items_list
```

```python
#joined_items_list = '?'.join(list_of_characters) #join and use blank space as separator
#print(' ', joined_items_list)
```

```python
# List methods overview
# print('\nTo get help for method:\n help(function.method)\n')
# help(list)
# help(list.append)
```

#### Note: From python [FAQ on list and mutability](https://docs.python.org/3/faq/programming.html#why-did-changing-list-y-also-change-list-x)

## 2. Tuples.
Denoted with ( )  
Iterable **Immutable** sequences, *typically* used to store collections of **heterogeneous** [data](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences).

>[built-in data types](https://docs.python.org/3/library/stdtypes.html#tuple)  
> tuple() [built-in function](https://docs.python.org/3/library/functions.html#func-tuple)  
> Indexing.  
> Slicing.  
> Basic functions on Tuples [common for all sequence types](https://docs.python.org/3/library/stdtypes.html#typesseq-common).

```python
name_of_teacher_that_talks_too_fast = "Thanasis"
```

The tuple below contains a string, an integer, a list and a variable.

```python
a_boring_tuple = (
    "hi", 3, ["an item of a list", "another item in that list"], name_of_teacher_that_talks_too_fast
    )
```

```python
type(a_boring_tuple)
```

```python
a_boring_tuple
```

### 2.1 Indexing a tuple

```python
# The second item of a borin tuple:
a_boring_tuple[1]
```

## 2.2 Slicing a tuple

```python
# The second, third and fourth item of that very very boring tuple.
a_boring_tuple[1:4]
```

Function outputs are oftenly returned as a tuple.  
Example with divmod function.

```python
result_and_remainder_of_integer_division = divmod(9, 6)  # (9 divided by 6) = 1 . The remainder = 3
```

```python
result_and_remainder_of_integer_division
```

```python
type(result_and_remainder_of_integer_division)
```

## 3. Sets.
For those weird people like 'you know who' that always want extra reading.

>[built-in data type](https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset)     
Unordered collection of distinct hashable objects.  

Common uses include:   
* membership testing,
* removing duplicates from a sequence,
* computing mathematical operations such as:
    * intersection,
    * union,
    * difference,
    * symmetric difference.

Notation: `{ }` or `set()` function.

### 3.1 Create sets and remove duplicates

```python
# A list with duplicate items
pets_list = ['cat', 'dog', 'cat', 'parrot', 'dog']

# Convert list to set: duplicates are removed
pets_set = set(pets_list)

print('original list :', pets_list)
print('as set        :', pets_set)
print('type          :', type(pets_set))
```

### 3.2 Membership testing (`in`)

```python
print('cat' in pets_set)
print('turtle' in pets_set)
```

### 3.3 Sets are mutable: add, update, remove, discard

```python
class_pets = {'cat', 'dog'}
print('start         :', class_pets)

class_pets.add('hamster')
print('after add     :', class_pets)

class_pets.update(['parrot', 'dog'])
print('after update  :', class_pets)

class_pets.remove('cat')
print('after remove  :', class_pets)

class_pets.discard('turtle')  # no error if item is missing
print('after discard :', class_pets)
```

### 3.4 Set operations: union, intersection, difference, symmetric difference

```python
set_a = {'cat', 'dog', 'parrot'}
set_b = {'dog', 'fish', 'parrot'}

print('set_a:', set_a)
print('set_b:', set_b)
print('union                :', set_a | set_b)
print('intersection         :', set_a & set_b)
print('difference (a - b)   :', set_a - set_b)
print('symmetric difference :', set_a ^ set_b)
```

### 3.5 Notes: no indexing, and how to create an empty set

```python
empty_dict = {}
empty_set = set()

print('type({})     =', type(empty_dict))
print('type(set())  =', type(empty_set))

# Sets are unordered, so convert to sorted list if you want stable display
print('sorted set_a =', sorted(set_a))
```

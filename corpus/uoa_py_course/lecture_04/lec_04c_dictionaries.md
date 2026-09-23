<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_04_index_slice_str_list_dict/reading_material/lec_04c_dictionaries.ipynb @ 0cc874704aaa -->

# Lecture 04b: Dictionaries.


Used for storing unordered key-value pairs. **Mutable** data container.   
Usage:  
When a unique "name" (eg index or ID) has to be **associated** with some constant, or changing "values".

> [Dicts are a built-in "Mapping" data type.](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)  
> Create dictionary, dict() constructor and other syntax.    
> [Dictionary methods](https://docs.python.org/3/library/stdtypes.html#typesmapping)  

Denoted with ```{ }```. These characters are called "braces".  
Iterable, Mutable.  
```{key: value}``` pairs. Standard syntax uses `:` to separate a key from its values.   


Dictionaries are like a data table without numeric index.  
A basic example of indexing by a "name" which is called "key".

Unordered does not mean random order.  
> Keys and values are listed in an arbitrary order which is non-random. Order **varies across Python versions**.
Order depends on the dictionary’s history of insertions and deletions.  

**Mind the data type of keys and of valeus**.   
Keys require immutable data types. Such as str, int, float.    
Keys are unique, the same key cannot appear twice in a dictionary.

## 1. Create a dictionary. Various ways, flexibility for different available data.

```python
# help(dict)
```

```python
# dict?
```

```python
empty_dict = {}

empty_dict
```

```python
# 
type(empty_dict)
```

```python
dictionary_example = {'one': 1, 'two': 2, 'three': 3}
```

#### When the keys are simple strings, it is sometimes easier to specify pairs using keyword arguments in the dict function.

```python
# keys are interpreted as strings. Can you understand why?  
dict_a = dict(one=1, two=2, three=3)  # dict() and assign key=value.
dict_a
```

```python
# numbers may be used as keys, although
dict_with_number_keys = {1: 1, 2: 2, 3: 3}
dict_with_number_keys
```

```python

# dict_a = dict('one'=1, 'two'=2, 'three'=3)  # Show this does not work.

# The reason is: Cannot assign a "value" to be equal to another "value".
# "one" is a string value and 1 is a integer value.
# "one" = 1
```

```python
# Show this does not work.
# "one" = 1
```

```python
# similarly, this does not work either, inside a dict.
#dict_with_number_keys_using_assignment = {1 = 1, 2 = 2, 3 = 3}
```

### Alternative syntax to create a dict.

```python
dict_b = {'one': 1, 'two': 2, 'three': 3}  # dict literal notation.

dict_c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))  # dict(), zip() functions. Combine two lists. help(zip) for more.

dict_d = dict([('two', 2), ('one', 1), ('three', 3)])  # dict() and list of pairs.

dict_e = dict({'three': 3, 'one': 1, 'two': 2})  # dict() and pairs.

dict_f = dict({'one': 1, 'three': 3}, two=2)  # combinations


dict_a == dict_b == dict_c == dict_d == dict_e == dict_f
```

```python
# Values may be a lists. E.g. tel_numbers_dict_numbers_dictnubmer and office number
dict_with_lists = {
    'jack': [4098, 4], 'jill': [4139, 8], 'jane': [4333, 3]}
dict_with_lists
```

#### Mind the formatting style below

```python
dict_with_lists_and_dict = {
    'jack': {"tel_numbers_dict_num": 4098, "office_num": 4},
    'jill': {"tel_numbers_dict_num": 4139, "office_num": 8},
    'jane': {"tel_numbers_dict_num": 4333, "office_num": 3},
}

dict_with_lists_and_dict
```

## 2. Access, inspect dictionary: items, keys, values.

```python
# Create a dictionary with phone numbers.
tel_numbers_dict = {'jack': 4098, 'jill': 4139, 'jane': 1234}
tel_numbers_dict
```

```python
type(tel_numbers_dict)
```

```python
# Inspect {key:value} pairs, called items, use items() method.
tel_numbers_dict.items()
```

```python
help(tel_numbers_dict.items)
```

```python
# keys() to inspect the keys
tel_numbers_dict.keys()
```

```python
tel_numbers_dict.values()
```

## 3. Indexing a Dict by key

```python
# Return the value of key.
tel_numbers_dict.get('jack')
```

```python
# Return the value of key. Different syntax, same result.
tel_numbers_dict['jack']
```

```python
# Same result.
tel_numbers_dict.get('jack') == tel_numbers_dict['jack']
```

```python
# tel_numbers_dict['james']  # key error
```

### Get key from a value. Note: values may not be unique.
Notice that in this [question](https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary), the reply with the most votes is WRONG and the accepted answer (second in votes) is incomplete).

```python
mydict = {'george': 16, 'amber': 19, 'jim': 16}
```

```python
list(mydict.keys())[list(mydict.values()).index(16)]
```

```python
# Returns only first instance.
print(list(mydict.keys())[list(mydict.values()).index(16)])
```

```python
# Return all instances.
search_age = 16

for key, value in mydict.items():
    if value == search_age:
        print(key)
```

```python
# My recommended way to return all instances in a list.
[name for name, age in mydict.items() if age == search_age]
```

```python
# This is the same as above but with different names.
# This works ok, python 3+. Called list comprehension.
[k for k, v in mydict.items() if v == search_age]  # k is for key and v for value
```

```python
# Reverses the dictionaty key value pairs. Keys become values.
# If values are not unique, this loses the 2d time a value appears because keys should be unique.
reversed_mydict = dict((v,k) for k,v in mydict.items())
reversed_mydict
```

```python
reversed_mydict[16]
```

```python
# Alternative, more explicit (verbose) syntax of list comprehension.
search_age = 16
name = [k for k in mydict.keys() if mydict[k] == search_age]; name
```

### Modify a dictionary. clear, add, remove, update.   
[Extensive recommended answer.](https://stackoverflow.com/a/8381589)

```python
# Remove all items.
tel_numbers_dict.clear()
tel_numbers_dict
```

```python
tel_numbers_dict = {'jack': 4098, 'jill': 4139, 'jane': 1234}
```

```python
# add a new item at the end. Works "in place" => without new assigment.
tel_numbers_dict['guido'] = 4127
tel_numbers_dict
```

```python
# remove an item. Works "in place" => without new assigment
del tel_numbers_dict['jill']
tel_numbers_dict
```

```python
# Remove key and return its value.
guido_tel_numbers_dict_number = tel_numbers_dict.pop('guido')
guido_tel_numbers_dict_number
```

```python
jack_tel_numbers_dict_number = tel_numbers_dict.pop('jack')
```

```python
jack_tel_numbers_dict_number
```

```python
tel_numbers_dict
```

```python
tel_numbers_dict['toni'] = guido_tel_numbers_dict_number
```

```python
tel_numbers_dict
```

```python
# modify a value
tel_numbers_dict['jane'] = 1000
tel_numbers_dict
```

```python
# Add new items at once
tel_numbers_dict.update(jim=2000, ann=3001)
tel_numbers_dict
```

```python
# Modify items' values
tel_numbers_dict.update(jack=4000, ann=3009)
tel_numbers_dict
```

```python
# insert new key without default value.
tel_numbers_dict.setdefault("jameson")
tel_numbers_dict
```

```python
# insert new key with default value, if no value exised
tel_numbers_dict.setdefault("dan", 1111) 
tel_numbers_dict
```

```python
# if value exists this is not the way to change it.
tel_numbers_dict.setdefault("dan", 3333) 
tel_numbers_dict
```

```python
# if value exists this is not the way to change it.
tel_numbers_dict.update({"dan": 3333}) 
tel_numbers_dict
```

### Miscelanous functions and operations on dictionaries

```python
len(tel_numbers_dict)  # N of items (N of pairs)
```

```python
list(tel_numbers_dict)  # convert keys to list
```

```python
sorted(tel_numbers_dict)  # convert keys to sorted list
```

```python
'thanasis' in tel_numbers_dict
```

```python
"ann" in tel_numbers_dict
```

```python
'jack' not in tel_numbers_dict
```

## 4. Dictionary comprehension.

Meaning of comprehension:
to make new sequence where each element is the result of some operations applied to each member of another sequence or iterable, or to create a subsequence of those elements that satisfy a certain condition.

Efficient method for:
* creating a dict from an iterable, or
* transforming one dictionary into another

```python
# an operation that is applied on all items of a set.
add_two_dict = {x: x+2 for x in {2, 4, 6, 100}}
add_two_dict
```

The code above creates a dictionary with the same keys as the set {2, 4, 6, 100} and the values are the keys plus 2.

## 5. Iterate over a Dictionary: Getting ready for next lecture!
In the next lecture we will see how to use loops.

```python
for employee in tel_numbers_dict:
    print(employee)
```

```python
for name in tel_numbers_dict.keys():
    print(name)
```

```python
for i in tel_numbers_dict.values():
    print(i)
```

```python
for key, value in tel_numbers_dict.items():
    print(key, value)  #, sep=" tel_numbers_dict. number ")
```

```python
for key, value in sorted(tel_numbers_dict.items()):
    print(key, value)
```

```python
dict_with_lists = tel_numbers_dict_numbers_dict= {'jack': [4098, 4097], 'jill': [4139, 4138], 'jane': [1234, 1233]}
dict_with_lists
```

```python
for key, value in dict_with_lists.items():
    print(key, value[0])
```

```python
for key, value in dict_with_lists.items():
    print(key, value[1])
```

## 5. Dictionarios Extra Reading (advanced):  
[subclasses](https://docs.python.org/3/library/collections.html)

Subclasses in Python are defined in a way that inherit properties of the parent classes but have some modified attributes.  
[OrderedDict](https://docs.python.org/3/library/collections.html#collections.OrderedDict) is an example of Dictionary that also has an index. It returns an instance of a dictionary subclass that has its own methods specialized for rearranging dictionary order.

```python
from collections import OrderedDict
```

#### Example of iterating over a dict and getting and index for items  
using the built-in [enumerate() function: ](https://docs.python.org/3/library/functions.html#enumerate)
This creates a new object called enumerate which works liked an explicitly indexed list.

```python
classic_dict_without_index = {"a": 1, "b": 2, "c": 3, "d": 4}

for index,  (key, value) in enumerate(classic_dict_without_index.items()):
    print(index, key, value)
```

```python
type(enumerate(classic_dict_without_index))
```

```python
enumerate?
```

```python
# show that slicing a dictionary does not work
#classic_dict_without_index.values()[:2]
```

```python
# convert to a list to get a slice of the values or keys
list(classic_dict_without_index.keys())[1:3]
```

```python
list(classic_dict_without_index.values())[1:3]
```

```python
list(classic_dict_without_index.items())[1:3]
```

#### Example of slicing an `OrderedDict`
using the [built-in itertools module](https://docs.python.org/3/library/itertools.html)

```python
from itertools import islice
```

```python
# mind the parentheses. A function is used along the dict notation to create an OrderedDict.
ordered_dict = OrderedDict({"a": 1, "b": 2, "c": 3, "d": 4})
ordered_dict
# and the output looks like a list doesn;t it?
```

```python
sliced_part = islice(ordered_dict.items(), 1, 3)

OrderedDict(sliced_part)
```

```python
# Of course, converting to a list is simpler.
list(ordered_dict.items())[1:3]
```

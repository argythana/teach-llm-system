<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_04_index_slice_str_list_dict/reading_material/lec_04a_index_slice_str.ipynb @ 0cc874704aaa -->

# Lecture 4a: Indexing and Slicing strings.

### Note: Slicing and Indexing works on **iterable** objects.
New words: indexing, slicing.

## 1. Strings.
[Iterable](https://docs.python.org/3/glossary.html#term-iterable), [**Immutable**](https://docs.python.org/3/glossary.html#term-immutable) Ordered sequences.  

> [srt: a **text sequence** type](https://docs.python.org/3/library/stdtypes.html#textseq)  

> [String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods)  

> [string constnts module](https://docs.python.org/3/library/string.html#module-string)

> Indexing  
> Slicing.  
> Basic functions on strings, len(), replace.  
> Basic string methods examples.  
> Extra.

Denoted with double or single quotes: " " or ' '.
Immutable: can't be changed directly "in place".
Iterable, Ordered sequences, can be indexed.

> [srt: text sequence type](https://docs.python.org/3/library/stdtypes.html#textseq)

> [String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods)

> [string constants module](https://docs.python.org/3/library/string.html#module-string)

----

### 1.1 String Indexing.

```python
# assign the string "nikos" to the variable onoma
onoma = "nikos"
```

```python
# Count Number of characters with len() function
onoma_length = len(onoma)
```

```python
onoma
```

```python
onoma_length
```

```python
onoma_length
```

```python
# indexing: use the index of ordered characters
first_letter = onoma[0]  # Count from 0
first_letter
```

```python
# indexing: use the index of ordered characters
third_letter = onoma[2] #count from 0
```

```python
third_letter
```

```python
onoma[1]
```

```python
# indexing counts backwards too
third_from_end = onoma[-3]
third_from_end
```

```python
onoma[-0]
```

```python
last_letter = onoma[-1] #reverse count from -1

last_letter
```

```python
an_interesting_name = "Maria Eleni"
```

```python
len(an_interesting_name)
```

```python
an_interesting_name[5]
```

### 1.2 String Slicing.

```python
onoma = 'nikos'

# Slicing parts of strings
# start index can be omitted if start from 0

first_three = onoma[0:3]  # first_three = onoma[:3] works to
first_three
```

```python
onoma[-3:]
```

```python
first_three = onoma[:3]  # the first index is not necessary. It is 0 by default.
first_three
```

#### NOTE: Slicing DOES NOT include the last index (Not inclusive operation).

```python
onoma_2nd_and_3rd = onoma[1:3] #letters index 0, 1, 2
# onoma_2nd_and_3rd = onoma[-7:-5] #same as above

onoma_2nd_and_3rd

#print('type of a slice = string:')
#print(type(onoma_2nd_and_3rd), '\n')
```

```python
onoma_2nd_and_3rd = onoma[0:-1] #same as above
onoma_2nd_and_3rd
```

```python
# slice from character -2 to the end
last_two = onoma[-2:] # or onoma[-2:len(onoma)]

last_two
```

```python
# Slice with steps by 3
onoma = 'nikolakis'

letters_by_three = onoma[: :3]  # Start from 0, go to the end, step by 3.

letters_by_three
```

```python
len(onoma)
```

```python
onoma[9]
```

```python
## Uncomment to see the IndexError 'out of range' message
## IndexError: string index out of range
# onoma[9]

# onoma[-9]
## Indexing always < len(string)
```

```python
type(onoma[1])
```

```python
# using indices to concatenate --> new string
second_and_fifth = onoma[1] + onoma[4]  # Note the indexes. Second letter and fifth letter.

second_and_fifth
```

```python
# Integers or floats cannot be sliced, first convert to strings
int_number = 4194
int_number_3rd_digit = str(int_number)[2]

print(f'To get digits convert numbers to strings. \
3rd digit of {int_number} is {int_number_3rd_digit}')
```

```python
int_number = 4194
```

```python
#int_number[2]
```

```python
print('Uncomment line below, and run 4194[2] to see the "TypeError" message')
## TypeError  int' object is not subscriptable
# int_number[2]
```

### 1.3 String Functions and methods. Operations on objects means a lot more than math.
> Functions on strings: len(), int("4").   
> Basic string [methods](https://docs.python.org/3/library/stdtypes.html#string-methods).  
> String Methods syntax uses string name and dot.    

replace('x', 'y') # replace x with y.  
count('x') # count occurences of x in string.   
find('x') # find position index of the 1st occurence of x.  
partition() # split once.   
rpartition() # split once from end.  
split() # split using defined delimiter.  
rsplit() # split from end.  
strip() # Strip characters from left & right end of string.  
capitalize() # 1st character to uppercase, all others to lowercase.  
lower() # all to lowercase.  
upper() # all to uppercase.  
title() # capitalise the first letter of each word.  
swapcase() # Guess what it does. Please do try this at home.

#### Today, at this point we write a surprise test to evaluate how boring I have been so far.
Don't worry, it is not graded for you. It is an evaluation of me.

```python
what_students_want = "test please"
```

```python
what_students_want
```

```python
what_students_want*20
```

```python
what_students_want
```

```python
## ipynb syntax to get help for a method
# str.replace?
```

```python
# replace all occurences by default, as many times some text exists.
what_students_want.replace("t", "No t")
```

```python
what_students_want
```

```python
# use string as arguments
what_students_want.replace("t", "No t", 1)  # does not work Inplace without assignment
```

```python
what_students_want
```

```python
# Split the string at the second occurrence of "t"
parts = what_students_want.split("t", 2)
parts
```

```python

# Join the parts back together, replacing the second "t" with "No t"
result = "t".join(parts[:-1]) + "No t" + parts[-1]

print(result)
```

#### Note: To replace or not to replace? No Assignment => Frequent error!

```python
what_students_want
```

```python
len(what_students_want)
```

```python
# this is an assignment
what_students_want = what_students_want.replace("t", "No t", 1)
```

```python
what_students_want
```

```python
# This is just a view of what the method does on the object value
# replace t with four t's for the first 2 occurences
what_students_want.replace("t", "tttt", 2)
```

```python
# This is just a view of what the method does on the object value
what_students_want
```

```python
len(what_students_want)
```

```python
# replace all occurences by default, as many times as the character exists in the string.
what_students_want.replace("t", "r")
```

```python
# replace only a specific number of occurences.
what_students_want.replace("t", "r", )
```

```python
what_students_want = what_students_want.replace("t", "r", 1)
what_students_want
```

<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_05_boolean_conditions_control_statements/reading_material/lec_05a_boolean.ipynb @ 0cc874704aaa -->

# Lecture 5a: Boolean operations, value comparisons.

**Truth Testing Expressions, value comparisons, return `boolean` values: `True` or `False`.**

Expressions in python mean a statement that has a value. Examples: `x = 5`, `y = 2`, `x + y`.

`Boolean` means: binary, "δυαδικές".

Do not confuse with Truth or Dare game.

In Python, `True` and `False` are two "constant" objects.

**New Words**:
* `Expressions`,
* `Boolean`,
* `Value Comparisons`,
* `Identity Comparisons`,
* `Membership test Operations`,
* `Boolean Operations`.

---

#### Learning Goals:

[Truth Value Testing](https://docs.python.org/3/library/stdtypes.html#truth-value-testing)
> Example: if password is correct then open the door.

[Basic Comparisons](https://docs.python.org/3/library/stdtypes.html#comparisons)
> There are **eight** comparison operations in Python: `<, <=, >, >=, ==, !=, is, is not`
> They all have the same priority (which is higher than that of the Boolean operations).

[Comparisons in general](https://docs.python.org/3/reference/expressions.html?#comparisons)


[Value Comparisons](https://docs.python.org/3/reference/expressions.html?#value-comparisons)

> The operators `<, >, ==, >=, <=, !=` compare the values of two objects.

> The objects do not need to have the same type.


[Membership test Operations](https://docs.python.org/3/reference/expressions.html?#membership-test-operations)

[Identity Comparisons](https://docs.python.org/3/reference/expressions.html?#is-not)


[Boolean Operations](https://docs.python.org/3/library/stdtypes.html#boolean-operations-and-or-not)

`Boolean` operations are the `and`, `or`, and `not` operations.
They use `Boolean` operators return a `Boolean` value.


[bool() function](https://docs.python.org/3/library/functions.html#bool)

**[Important reading: The difference between operators `==` and `is`. Read all the replies.](https://stackoverflow.com/questions/132988/is-there-a-difference-between-and-is/1085656#1085656)**

Then ask an AI assistant more about it.



## 1. True in Python.
In python, **True** is the default state of things.
Some exceptions are shown at the end of lecture (parental advisory content).

```python
# Just some boring names and their values.
x = 9
w = 4
y = 2
antonis = 10
z = "last but not least"
```

```python
x  #  Call the variable x.
```

```python
# bool is the function that works behind the scenes to test the truth.
bool(x)
```

```python
bool(t)
```

```python
not x  # True is the default state of things.
```

```python
bool(not x)
```

## 2. Six **Value** Comparison operators: `<, <=, >, >=, ==, !=`

```python
9 > 2
```

```python
x >= y  # True
```

```python
x != y  # True
```

```python
x == y  # False. Value comparison
```

## 3. Identity comparisons operators: `is not`, `is`, `not`

```python
x is not y  # True
```

```python
x is y  # Identity comparison
```

```python
y
```

```python
y is 2
```

```python
a = 256
b = 256
a is b
```

```python
x = antonis  # assignment
```

```python
x == antonis  # Value comparison
```

```python
x is antonis  # Identity comparison
```

```python
y == 2
```

```python
2 == 2
```

```python
y == 2
```

```python
x == y  # False
```

```python
x is y  #False
```

`not` has a lower priority than non-Boolean operators.

Therefore, `not` is evaluated after the comparison operators, even if it is written first from left to right.

So, `not a == b` is interpreted as `not (a == b)`, which is the same result.

```python
x, y
```

```python
x == y
```

```python
not (x == y)
```

```python
not x == y
```

```python
(not x) == y # False
```

```python
bool(x)
```

```python
not x
```

```python
not (x == y)  # True
```

```python
not x is y  # True because if means: not(x is y)
```

```python
not(x is y)  # same as above
```

```python
2 < 4 < 9 < 15  #True. Chained comparison (y < w and w < x)
```

```python
y < w < x  #True. Chained comparison (y < w and w < x)
```

### 3.1 Value Comparisons for other data types.

**Strings and Lists are compared based NOT by their length but by their lexicographical order.**

Lexicographical order is similar to the order used in dictionaries, where strings are compared character by character based on their Unicode values.
The letter that comes first in the alphabet is considered "smaller" than the letter that comes later.

#### Strings

```python
'10' < '2'
```

```python
# 'a' comes before 'b' in lexicographical order
'ahelloooooooooo' < 'bhello' # True
```

```python
"apple_b_eeeeeeeeeeeee" < "apple_c"  # True
```

```python
"ab" < "abc"
```

```python
# uppercase 'A' has a lower Unicode value than lowercase 'a'
"Apple" < "apple"  # True
```

```python
apple = "apple"
ora = "orange"
apple == "orange"
```

```python
apple == "orange"
```

```python
# I told you we can compare apples with oranges!
apple < ora
```

```python
apple < 'orange'
```

```python
two = "2"
```

```python
ten = "10"
```

```python
two < ten
```

```python
"19999991100" < "2"
```

```python
'α' < 'a'
```

### Lists

```python
a = [1, 30]
b = [1, 2, 3, 4]
```

```python
len(a) < len(b)
```

```python
[1, 2, 3] < [1, 2, 3, 4] # True
```

```python
[1, 30] > [1, 2, 3, 4] # False
```

#### Compare Apples with oranges! Yes you can, but ...

Don't compare `str` with `int`.

```python
# Uncomment the line below for comparison `TypeError`. Cannot compare 'int' with 'str'.
# 'hello' > 2  # TypeError
```

```python
a < 'hello'  # False. Cannot compare 'int' with 'str'.
```

## 4. Membership test operations: `in`, `not in`

```python
# dont use list for different types!
my_list = ["John", 1, 2, 3]  # don't use lists for different types!
my_list  # BTW, please don't use lists for different types!
```

```python
"John" in my_list  # True
```

```python
1 in my_list  # True
```

```python
4 in my_list  # False
```

```python
4 not in my_list  # True
```

```python
not (4 in my_list) # True.
```

```python
x is None
```

## 5. Boolean Logical Operations: Logical `or`, `and`, `not`

Number of students that completed Economics and Political Science?

#### My favorite error: Using "and" when I should be using "or".

```python
#bool() function, test true or False
bool(0.0)  # False
```

```python
bool("")  # False
```

```python
bool({})  # False
```

```python
bool(x)  # True
```

```python
bool(False)
```

```python
True and False  # False, because both can not be True => False
```

```python
True or False  # True
```

#### Objects defined as `False` by default

* constants defined to be false: `None` and `False`.

* zero of any numeric type: 0, 0.0, 0j, Decimal(0), Fraction(0, 1)

* empty sequences and collections: "", (), [], {}, set(), range(0)

```python
my_dict = {}
```

```python
bool(my_dict)
```

```python
x = 0
```

```python
t = 0.0
```

```python
bool(x)
```

```python
bool(t)
```

## 6. Extra reading, if you want to get brain damage and become a zombie.

But, necessary if you want to become an expert in Python.

Boolean operations, [Short-circuit evaluations](https://docs.python.org/3/library/stdtypes.html?highlight=short%20circuit%20evaluation#boolean-operations-and-or-not).  
**Evaluation stops if False**.  
So, in logical and tests, if True => last evaluated value is returned.

```python
x = 9
y = 2
z = "last but not least"
```

```python
x
```

```python
x or y # x is True, return value of x. The first value.
```

```python
y or x
```

```python
y
```

```python
0 or y # 0 is False, return value of y. The second value.
```

```python
0 and y # because zero is False and the union is False, return 0. y is not evaluated.
```

```python
z
```

```python
x and y  #  if True => last evaluated value is returned.
```

```python
x and y and z # if True => last evaluated value is returned.
```

```python
(False and x)
```

```python
(False and x) or y #? remember parentheses, left to right
```

```python
False and (x or y) #? remember parentheses,left to right
```

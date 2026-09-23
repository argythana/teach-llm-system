<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_06_define_functions/reading_material/lec_06d_user_modules_scripts.ipynb @ 0cc874704aaa -->

# Lecture 6d. User-defined modules, stand-alone scripts and `if __name__ == "__main__"`

<mark>This is extra/advanced material.</mark> It builds on the concepts from lectures 6a-6c.

**Prerequisites:** You should be comfortable with defining and calling functions (lecture 6b) before reading this notebook.

Learning Goals:
* Understand what a **user-defined module** is (a `.py` file containing functions).
* Learn how to **create** a module file and **import** it in a notebook or another script.
* Understand the two main import styles and when to use each one.
* Understand the `__name__` attribute and the `if __name__ == "__main__"` pattern.
* Know the difference between running a file **as a script** vs. importing it **as a module**.
* Use `importlib.reload()` to pick up changes in a module without restarting the interpreter.

## 1. What is a user-defined module?

A **module** is simply a `.py` file that contains Python code — functions, variables, classes.

You already use modules every day: `math`, `statistics`, `random`, `pandas` are all modules.

The difference is that those are *built-in* or *third-party* modules.  
A **user-defined module** is a `.py` file that *you* write yourself.

**Why create modules?**
* **Reusability:** Write a function once, import it everywhere.
* **Organization:** Keep related functions together in one file.
* **DRY principle:** Don't Repeat Yourself.

[Python docs — Modules](https://docs.python.org/3/tutorial/modules.html)  
[Best practices for using import](https://docs.python.org/3/faq/programming.html#what-are-the-best-practices-for-using-import-in-a-module)

## 2. The `hypot_module.py` file

In this directory there is a file called `hypot_module.py`.  
Open it in your editor and read its contents. It contains:

1. A module-level **docstring** that describes what the file is for.
2. An import statement: `from math import sqrt`.
3. Two function definitions: `calculate_hypot()` and `hypot_calculator_modular()`.
4. An `if __name__ == "__main__"` block at the bottom (we will explain this in section 5).

Let's see the contents of the file from inside this notebook:

```python
# Show the contents of the module file.
# The ! prefix runs a terminal (shell) command from inside a notebook.
!cat hypot_module.py
```

## 3. Importing a user-defined module

There are two main ways to import from a module. Both have trade-offs.

| Style | Syntax | Usage | Recommended? |
|---|---|---|---|
| Import the module | `import hypot_module` | `hypot_module.calculate_hypot(3, 4)` | **Yes** |
| Import specific names | `from hypot_module import calculate_hypot` | `calculate_hypot(3, 4)` | Use with care |

**Recommendation:** Use `import module_name` and reference functions with `module_name.function()`.  
This makes it clear where each function comes from, especially in larger projects.

### 3a. Recommended style: `import module_name`

```python
# Recommended import style.
import hypot_module
```

```python
# Check the module name.
hypot_module.__name__
```

```python
# Access the module docstring.
print(hypot_module.__doc__)
```

```python
# Call a function from the module using module_name.function_name() notation.
hypot_module.calculate_hypot(3, 4)
```

```python
# Call with keyword arguments.
hypot_module.calculate_hypot(side_a=5, side_b=12)
```

### 3b. Alternative style: `from module import function`

```python
# Import a specific function from the module.
from hypot_module import calculate_hypot
```

```python
# Now you can call it directly, without the module_name prefix.
calculate_hypot(3, 4)
```

```python
# You can also import multiple names at once.
from hypot_module import calculate_hypot, hypot_calculator_modular
```

**Why is `from module import *` discouraged?**

`from hypot_module import *` imports *everything* from the module into your current namespace.  
This can silently overwrite existing names and makes it hard to tell where a function came from.

**Rule of thumb:** Always be explicit about what you import.

## 4. How Python finds your module

When you write `import hypot_module`, Python looks for `hypot_module.py` in a list of directories.  
The first place it looks is the **current working directory** (the directory of the notebook).

That is why `hypot_module.py` must be in the same folder as this notebook for the import to work.

```python
# The current working directory of this notebook.
!pwd
```

```python
# Python searches for modules in these directories, in order.
import sys
for p in sys.path:
    print(p)
```

## 5. The `__name__` attribute and stand-alone scripts

Every Python module has a special built-in attribute called `__name__`.

* When a `.py` file is **run directly** (e.g., `python hypot_module.py`), `__name__` is set to `"__main__"`.
* When a `.py` file is **imported** as a module, `__name__` is set to the **module's own name** (e.g., `"hypot_module"`).

This is how Python knows whether a file is being run as the main program or being used as a library.

```python
# In a notebook, __name__ is always "__main__" because this IS the main program.
print(__name__)
```

```python
# When we import a module, its __name__ is the module name, not "__main__".
import hypot_module
print(hypot_module.__name__)
```

```python
# Same for built-in modules.
import math
print(math.__name__)
print(math.sqrt.__name__)
```

## 6. The `if __name__ == "__main__"` pattern

This pattern allows you to write a `.py` file that works **both** as:
1. A **module** (imported by other code), and
2. A **stand-alone script** (run directly from the terminal).

```python
# at the bottom of your .py file:
if __name__ == "__main__":
    # This code runs ONLY when the file is executed directly.
    # It does NOT run when the file is imported.
    print("Running as a script!")
```

**Why is this useful?**
* You can put test code, demos, or a command-line interface in the `if __name__` block.
* When someone imports your module, that test/demo code does not execute.

[Stack Overflow — What does `if __name__ == "__main__"` do?](https://stackoverflow.com/questions/419163/what-does-if-name-main-do)

### 6a. Running the module as a stand-alone script from the terminal

Open a terminal, navigate to this directory, and run:
```bash
python hypot_module.py
```

You should see the output from the `if __name__ == "__main__"` block:  
`Running hypot_module.py as a stand-alone script.`  
`Example: hypotenuse of sides 3 and 4 is 5.0`

Let's do this right here from the notebook:

```python
# Run the module as a stand-alone script.
# The `if __name__ == "__main__"` block WILL execute.
!python hypot_module.py
```

### 6b. Importing the module — the `if __name__` block does NOT run

```python
# When we import, __name__ is "hypot_module", NOT "__main__".
# So the if __name__ == "__main__" block is skipped.
import hypot_module

# We can use the functions normally.
result = hypot_module.calculate_hypot(5, 12)
print("Hypotenuse:", result)
```

## 7. Reloading a module with `importlib.reload()`

For efficiency, Python imports each module **only once** per interpreter session.  
If you change the `.py` file and re-run `import hypot_module`, **nothing happens** — Python uses the cached version.

To pick up changes without restarting the kernel, use `importlib.reload()`:

```python
import importlib
importlib.reload(hypot_module)
```

This is very useful during interactive development.

```python
import importlib

# Suppose you edited hypot_module.py (e.g., changed a print message).
# Reload the module to pick up the changes.
importlib.reload(hypot_module)

# Now hypot_module reflects your latest edits.
hypot_module.calculate_hypot(7, 24)
```

## 8. Practical exercise

1. Open `hypot_module.py` in your editor.
2. Add a new function `calculate_area(base, height)` that returns the area of a triangle (`0.5 * base * height`).
3. Save the file.
4. In the cell below, reload the module and call your new function.

```python
# After editing hypot_module.py, run this cell to test your new function.
import importlib
importlib.reload(hypot_module)

# Uncomment and test:
# hypot_module.calculate_area(10, 5)
```

## 9. Summary

| Concept | Key idea |
|---|---|
| **Module** | A `.py` file containing functions, variables, classes. |
| **`import module`** | Recommended. Use `module.function()` to call. |
| **`from module import func`** | Shorter, but be explicit. Avoid `import *`. |
| **`__name__`** | `"__main__"` when run directly; module name when imported. |
| **`if __name__ == "__main__"`** | Guard block for stand-alone script behavior. |
| **`importlib.reload()`** | Re-import a module after editing it, without restarting. |

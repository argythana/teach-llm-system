<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_06_define_functions/reading_material/lec_06c_stand_alone_scripts.ipynb @ 0cc874704aaa -->

# Lecture 6c. Stand-alone scripts and the `if __name__ == "__main__"` statement.

This notebook is a quick overview and reference. For the full material on user-defined modules, import styles, and `if __name__ == "__main__"`, see `lec_06d_user_modules_scripts.ipynb`.

## 1. From notebook code to a reusable module

The two functions below (`myhypot` and `hypot_calculator`) have been saved to `hypot_module.py` so they can be imported from any notebook or script.

```python
# View the contents of the module file.
!cat hypot_module.py
```

```python
# These are the original function definitions that were later saved to hypot_module.py.
# They are kept here for reference only — you do not need to run this cell.

from math import sqrt


def myhypot(x, y):
    """Calculate hypotenuse. Assign values for x, y."""
    hyp = sqrt(x * x + y * y)  # hyp = local variable
    return hyp  # Only the value is returned to the caller, not the name.


def hypot_calculator():
    """Ask for triangle sides and print hypotenuse. Input should be numbers."""
    a = float(input("Insert side a: "))
    b = float(input("Insert side b: "))
    hypotenuse = myhypot(a, b)
    print("Hypotenuse =", hypotenuse)


# --- Quick tests (uncomment one at a time) ---
# myhypot(6, 8)                  # Works: pass exactly two positional arguments.
# hypot_calculator()              # Works: asks for user input interactively.
# hypot_calculator(3, 4)          # TypeError: takes 0 positional arguments but 2 were given.
# hypot_calculator(x=3, y=4)     # TypeError: got an unexpected keyword argument 'x'.
```

## 2. User-defined modules

[Modules.](https://docs.python.org/3/tutorial/modules.html)

[Best practices](https://docs.python.org/3/faq/programming.html#what-are-the-best-practices-for-using-import-in-a-module) for using import.

### Stand-alone scripts
To create stand-alone scripts, must read: `if __name__ == "__main__"` — [Stack Overflow](https://stackoverflow.com/questions/419163/what-does-if-name-main-do)

### Reloading modules during interactive development

For efficiency, each module is only imported once per interpreter session.

Therefore, if you make changes to your module, you must restart the interpreter — or use `importlib.reload()`:

```python
import importlib
importlib.reload(module_name)
```

```python
# the name of the main module is "__main__"
print(__name__)
```

```python
# the name of the imported module is the explicit module name
import math
math.__name__
```

```python
# The imported function also has a __name__ attribute.
math.sqrt.__name__
```

```python
# Alternative (not recommended) import style: import a specific function.
# This works without the module_name prefix, but makes it harder to track where functions come from.
from hypot_module import hypot_calculator_modular

hypot_calculator_modular()
```

```python
# Recommended import method
import hypot_module
```

```python
hypot_module.__name__
```

```python
# With the recommended style, the function is called with the module prefix.
# This makes it clear that hypot_calculator_modular comes from hypot_module.
hypot_module.hypot_calculator_modular()
```

```python
# If you edit hypot_module.py, the changes are NOT picked up automatically.
# Use importlib.reload() to re-import the module with your latest changes.
import importlib

importlib.reload(hypot_module)

hypot_module.hypot_calculator_modular()
```

```python
# Run hypot_module.py as a stand-alone script.
# The if __name__ == "__main__" block WILL execute because the file is the main program.
!python hypot_module.py
```

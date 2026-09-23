<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_03_cwd_path_pip_venv_imports/reading_material/instruct_03c_pip_venv.ipynb @ 0cc874704aaa -->


# Lecture 3c: Instructions: `venv`, `pip`

## Package management, virtual environment

----------------
INSTRUCTIONS TO:
----------------

A) Create a [Virtual Environment](https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments) for project reproducibility, resolve conflicting dependencies, maintaining purposes.  
We will use a package called `venv`: Virtual Environment.   

B) Install [modules in it](https://docs.python.org/3/installing/index.html) (only in it) or [install packages in it.](https://packaging.python.org/tutorials/installing-packages/#id13)   
We will use a package called `pip`: Python Install Package.  

Can you tell the difference between modules and packages? Read the top two answers [here.](https://stackoverflow.com/questions/7948494/whats-the-difference-between-a-python-module-and-a-python-package)  


Note: install packages and modules ONLY from trusted sources.  

  

## 1. Make sure Python is included in Windows PATH (when installing).  

This means that you should tick the box: `Add python to environment variables` when installing python.  

Else: modify the installation from control panel (add or remove programs)


### 1a. Open "Console", aka "Command Line Interface" (CLI), aka "Terminal, aka "Shell".  
I will refer to it as **CLI**.    
    For Windows:  
	a) press "Windows key" + "R"  
	b) type the word "cmd", press Enter

You should now be inside the **Windows Operating System** terminal in **your user** default path:  
`C:\Users\thanasis_argyriou>`

"thanasis_argyriou" is me, not you!
	
For [Mac OS CLI usage check here.](https://opentechschool.github.io/python-beginners/en/getting_started.html#what-is-python-exactly)

*For ALL the following commands, if python CLI command does not work in Windows CLI:
replace "python" with "py" (py is the Windows Python launcher).* 


### 1b. Check python and pip version.  

Inside the Windows CLI, type:

```bash
python --version
(output should be like: 3.12.8)
```

To check pip version and if pip is installed, inside the Windows CLI, type:

```bash
python -m pip --version
```


Notice the version number and the location (aka path) where the pip package is located by default.

A similar way to see the pip version is:  

```bash
pip --version
```  

**Make sure you are at your Windows Command line, which has a single > (or the \$ sign in MacOS) at the end,
not at python’s command prompt (python command line has three >>> instead)**

## 2. Make a new directory, (folder) using `mkdir` (make directory) command.  

Name the new directory as: ```uoa_py_course```    

Inside the CLI, type:  
```bash
C:\Users\thanasis_argyriou> mkdir uoa_py_course
```

## 3. Go inside that new directory using `cd` (change directory) command.

```bash
cd uoa_py_course
```

The prompt output would be:  
```C:\Users\tharg\uoa_py_course>```


## 4. Create a new virtual environment with `venv` python package.  

The virtual environment should be made **inside** the `uoa_py_course` folder.  

The name of the virtual environment folder will be: `course_venv`

To create the virtual environment, make sure that:
    a. you are using the Windows CLI and 
    b. you are in the `uoa_py_course` folder.
    
In the windows CLI type:

```bash
python -m venv course_venv
```  

You just created a virtual environment named: `course_venv`.   

If you want to inspect what is inside use:
`cd course_venv`  and then `dir` command, or when you are in the `uoa_py_course` folder just type: `dir course_venv`,
or in MacOS `ls course_venv`.

## 5. Activate the virtual environment.   

Be inside the directory `uoa_py_course` and **above** of the directory `course_venv` and type in the CLI:    
```bash
.\course_venv\Scripts\activate
```

or
```bash
course_venv\Scripts\activate
```

The complete command prompt in the Windows CLI will look like:  
```bash
C:\Users\USER_YOU_NOT_ME\uoa_py_course> .\course_venv\Scripts\activate
```
    
If done correctly, when you activate the virtual environment you will see **a parenthesis** in front of directory path:  
```bash
(course_venv) C:\Users\USER_YOU_NOT_ME\uoa_py_course
```

To deactivate the virtual environment, just close the Command Line Window, or type:  
```bash
deactivate
```

To activate the virtual environment in **MacOS** type:  
```bash
myusername$ source ./course_venv/bin/activate
```

## 6. Install python packages.  

The virtual environment should **ALWAYS** be activated.  

We use the [pip package](https://pip.pypa.io/en/latest/) 

> pip is the package installer for Python.   
You can use pip to install packages from the Python Package Index and other indexes.


### 6a. Upgrade to latest version of pip, in CLI, type:  
```bash
python -m pip install --upgrade pip
```

### 6b. Install and upgrade `setuptools` and `wheel` packages (install if not installed, or upgrade if installed).  
Using the Windows CLI:

```bash
python -m pip install --upgrade setuptools wheel
```  

> The general syntax to install any package is:
    ```bash
    > python -m pip install <package_name>
    ```


### 6c. Install the latest version of a single package (e.g. `numpy`).
```bash
python -m pip install numpy
```  


### 6d. Install a specific version of a single package (e.g. `jupyterlab` version 4.3.4).
```bash
python -m pip install jupyterlab==4.3.4
```  

Make sure to install `jupyterlab` using a stable version before the latest version to avoid possible bugs.  

### 6e. Install many packages at once. 

> You may install more than one packages at once:  

```bash
python -m pip install <package_a> <package_b> <package_x>
```

```bash
python -m pip install pandas matplotlib seaborn
```

## 7. Inspect and manage installed packages using pip.

### 7a. Show all packages installed by `pip` in the virtual environment:
```bash
python -m pip freeze
```

### 7b. Show any single package version, dependencies, required by:  
```bash
pip show <package name>
```

Notice the version number and the location (the complete path) where the package is located in the virtual environment.  
Notice that this is different from the default python installation path.

### 7c. See all available versions of a python package:

A new "experimental" command that **might change** in the future is:   
```bash
pip index versions jupyterlab
```  
This will return all available packages from latest to oldest.   

### 7d. Save all packages installed by pip to new ".txt" file:
```bash
pip freeze > requirements.txt
```

**Try the following to practice and do it all over again.**

To delete a virtual environment, just delete the whole folder.  

To uninstall a module, type:  
```bash
python -m pip uninstall <package_name>
```

	
To install many packages at once from a file (recursively):   
```python -m pip install -r requirements.txt```

To upgrade many packages at once from a file:  
```python -m pip install --upgrade -r requirements.txt```

# This is the end of the installation instructions!
---

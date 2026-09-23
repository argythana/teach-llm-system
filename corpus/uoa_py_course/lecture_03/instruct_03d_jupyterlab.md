<!-- source: lectures_01_06_fundamentals_for_data_science/lecture_03_cwd_path_pip_venv_imports/reading_material/instruct_03d_jupyterlab.ipynb @ 0cc874704aaa -->

# Lecture 3d: Instructions: Install and open `jupyter lab`

`JupyterLab` is an "Interactive Python Notebook" `.ipynb` editor.  
It is an **interactive** feature rich editor that uses your browser.    

Read all about it [here](https://jupyter.org/).


----------------
INSTRUCTIONS TO:
----------------

A) Install and 
B) Run a `jupyter notebook` from your PC.  
We will use a package called `jupyterlab` **(ju-PY-ter NOT ju-PI-ter)**.  


## 1. Open the Windows `Command Prompt`.
I will refer to it as **CLI**.    
    For Windows:  
	a) press "Windows key" + "R"  
	b) type the word "cmd", press Enter

You should now be inside the **Windows** terminal in **your user** default path:  
`C:\Users\thanasis_argyriou>`

"thanasis_argyriou" is me, not you!
	
For [Mac OS CLI usage check here.](https://opentechschool.github.io/python-beginners/en/getting_started.html#what-is-python-exactly)

*For ALL the following commands, if python CLI command does not work in Windows shell:	 
replace "python" with "py" (py is the Windows Python launcher).* 


**Make sure you are at your Windows `Command Prompt`, which has a single > (or the \$ sign in MacOS) at the end,
not at python’s command prompt (which has three >>> instead)**

Then go to the python course folder using `cd`:
```bash
cd uoa_py_course
```

## 2. Activate the virtual environment.   

Be inside the directory `uoa_py_course` and **above** of the directory `course_venv` and type in the Windows CLI:    
```bash
course_venv\Scripts\activate
```

The complete command prompt in the Windows CLI will look like:  
```bash
C:\Users\USER_YOU_NOT_ME\uoa_py_course> course_venv\Scripts\activate
```
    
If done correctly, when you activate the virtual environment you will see **a parenthesis** in front of directory path:  
```bash
(course_venv) C:\Users\USER_YOU_NOT_ME\uoa_py_course
```

## 3. Install a specific version of `jupyterlab` (e.g.version x.x.x).

Inside the Windows CLI type:

```bash
python -m pip install jupyterlab==4.3.4
```

Make sure to install a stable version of `jupyterlab` (at least one before the latest version) to avoid possible new bugs.

## 4. Run a `jupyterlab` "*interactive python notebook*" on your PC:


If you installed the packages in a virtual environment,   
the environment **should be activated** to use them.  
So, **activate** the `venv` **every time** before running the `jupyter-lab` interactive editor.  

In the Windows CLI type:  

```bash
jupyter-lab
```

After entering this command, the `jupyterlab` server will start, your terminal will give you several messages and the `jupyterlab` will open in your default browser.

One of the messages will be like:

```bash
[I 2025-03-11 21:51:58.168 ServerApp] Jupyter Server 2.13.0 is running at:
[I 2025-03-11 21:51:58.168 ServerApp] http://localhost:8888/lab?token=a89357360e451f535145ce697e6a8e9f388f6688a8e5286b
```

**If your browser is playing hard to get, copy the `url` from the terminal and paste it in the browser.**


#### BE CAREFULL:   
> When **installing** jupyterlab, there should be **no space** in between. It is **one word**.

> When **running** jupyter lab, there should be a **dash** inbetween: **jupyter-lab**.

Put any **interactive python notebook type** files (```.ipynb```) in the proper folder.  
It is recommended to make separate folders, **next** to the `course_venv` folder.   

E.g.  
> Make a directory called **data** to store data files.  
> Make a directory called **playground** to store draft notebooks and whatever files.  
> Make a directory called **lectures_files** to store the lecture interactive python notebooks.  
> Inside the `lectures_files` directory create a separate directory for each lecture.  


You can make directories either from the CLI, or by using the file explorer graphical Interface and your mouse.  
The result is the same. 

The **course_venv** directory should NOT be touched, unless you know what you are doing.  
This directory is reserved for python packages and virtual environment management and configuration.  
It is recommended NOT to save or move any files in **course_venv**, unless related to environment management.  


### 5. Extra:
#### To convert a .ipynb file to .py  
Go to the Menu tab "File" -> "Save and Export Notebook As" -> "Executable Script"  
Add .py extension at the end of the filename and save if as "All Files" file type format.  

#### To convert a .ipynb file to .html  
Go to the Menu tab "File" -> "Save and Export Notebook As" -> "HTML".

# Setup venv

WARNING:
XGBoost library requires 64-bit version of Python for 64-bit OS.

### Mac/Win

0. Open terminal to run the comands given below.

(Win) Open the integrated terminal (press `Ctrl + `` or go to View > Terminal).
(Mac) Open terminal directly, or from Finder, or in VS Code for the current project folder.

1. Create the virtual environment.

MAC:
```bash
python3 -m venv myenv64; \
source myenv64/bin/activate
```

WIN:
```powershell
$pythonPath = "C:\Users\YourUsername\AppData\Local\Programs\Python\Python310\python.exe"
```
- adjust the path to your 64-bit Python executable.

```
& $pythonPath -m venv C:\path\to\your\env64
# Example:
# & "C:\Users\Alice\AppData\Local\Programs\Python\Python310\python.exe" -m venv "D:\Projects\myenv64"
```
- this creates the venv


2. Activate the virtual environment:

- For Windows: 
```bash
myenv64\Scripts\activate
```

- For Linux/Mac: 
```bash
source myenv64/bin/activate
```

(Win)

```bash
venv\Scripts\activate
```

You should see (venv) in your terminal prompt, indicating that the environment is active.

Deactivate to switch back to normal terminal usage or activate another virtual environment to work with another project run:

```bash
deactivate
```

# Install dependencies

```bash
pip install --upgrade pip; \
pip install -r requirements.txt
```

Activate internal packages:

```bash
pip install -e .
```

## Configure using the same environment for Jupyter Notebooks

Preconditions: install dependencies first (see package management), including `ipykernel`!

After running this comand in terminal:

```bash
python -m ipykernel install --user --name=venv --display-name "Python 64 (venv)"
```

restart VS code.

Open a `.ipynb` file and add code block at the top with:

```python
!which python
```

Try running the cell. VS Code will prompt for choosing a Jupyter kernel:

- Select another kernel
- Jupyter kernel
- _Refresh the list using icon on the top right!_
- Choose the virtual environment of this project.

Run the code cell again and make sure it returns the correct path to `<this project folder>/venv/bin/python3`

This will resolve all possible `ModuleNotFoundError` occurences when importing libraries.

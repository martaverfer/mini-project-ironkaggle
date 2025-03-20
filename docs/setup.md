# Setup venv

### Mac/Win

Create the virtual environment:

(Win) Open the integrated terminal (press `Ctrl + `` or go to View > Terminal).

```bash
python3 -m venv venv
```

Activate the virtual environment:
<<<<<<< HEAD

(Mac)

=======
- For Windows: 
```bash
venv\Scripts\activate
```

- For Linux/Mac: 
>>>>>>> main
```bash
source venv/bin/activate
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
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
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

# Setup venv and install dependencies

Create the virtual environment: 
```bash
python3 -m venv venv
```

Activate the virtual environment:
```bash
source venv/bin/activate
```

Deactivate to switch back to normal terminal usage or activate another virtual environment to work with another project run:
```bash
deactivate
```

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
- *Refresh the list using icon on the top right!*
- Choose the virtual environment of this project.

Run the code cell again and make sure it returns the correct path to `<this project folder>/venv/bin/python3`

This will resolve all possible `ModuleNotFoundError` occurences when importing libraries.


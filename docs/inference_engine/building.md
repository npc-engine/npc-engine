## Build on Windows

#### Create virtualenv and activate it

```cmd
python3 -m venv npc-engine-venv
.\npc-engine-venv\activate.bat
```

#### Install dependencies

```
pip install -e .[dev,dml]
```

#### (Optional) Compile, build and install your custom ONNX python runtime" 

Build instructions can be found [here](https://onnxruntime.ai/)

#### (Optional) Run tests

```
tox -e py38
```

#### Compile to exe with

```
> pyinstaller --hidden-import="sklearn.utils._cython_blas" --hidden-import="sklearn.neighbors.typedefs" ^
--hidden-import="sklearn.neighbors.quad_tree" --hidden-import="sklearn.tree._utils" ^
--hidden-import="sklearn.neighbors._typedefs" --hidden-import="sklearn.utils._typedefs" ^
--hidden-import="sklearn.neighbors._partition_nodes" --additional-hooks-dir hooks ^
--exclude-module tkinter --exclude-module matplotlib .\npc_engine\cli.py --onedir
```
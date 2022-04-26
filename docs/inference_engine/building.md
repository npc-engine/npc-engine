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
pyinstaller --additional-hooks-dir hooks --exclude-module matplotlib --exclude-module jupyter --exclude-module torch --exclude-module torchvision .\npc-engine\cli.py --onedir
```
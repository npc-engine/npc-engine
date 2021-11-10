## Build on Windows

#### Create virtualenv and activate it

```cmd
python3 -m venv npc-engine-venv
.\npc-engine-venv\activate.bat
```

#### Install dependencies

```
pip install -e .[dev]
```

#### (Optional) Compile, build and install your custom ONNX python runtime" 

Build instructions can be found [here](https://onnxruntime.ai/)

#### (Optional) Run tests

+ Download models to run tests against into `npc-engine\resources\models`.  
You can use default models from [here](https://drive.google.com/drive/folders/1_3iOrhgvDyrKnC-tWEdysxpJyUcun0X3?usp=sharing)
+ Run tests with
    ```
    tox
    ```

#### Compile to exe with

```
pyinstaller --additional-hooks-dir hooks --exclude-module matplotlib --exclude-module jupyter --exclude-module torch --exclude-module torchvision .\npc-engine\cli.py --onedir
```
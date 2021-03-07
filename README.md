# Inference engine

ZMQ server for providing onnx runtime predictions for text generation and speech synthesis.

Build:

- Compile to exe with:

`pyinstaller --additional-hooks-dir hooks --exclude-module matplotlib --exclude-module jupyter --exclude-module torch --exclude-module torchvision .\inference_engine\cli.py --onedir`

- Move `dist\cli` to Unity project main folder

- Rename `cli` to `AISidecar`

## Authors

- **evil.unicorn1** - _Initial work_ - [github](https://github.com/eublefar)

See also the list of [contributors](https://github.com/eublefar/inference_engine/contributors) who participated in this project.

## LicenseCopyright (c) evil.unicorn1

All rights reserved.

It's possible to convert models from popular libraries into npc-engine services via [npc-engine-import-wizard](https://github.com/npc-engine/npc-engine-import-wizard).

You can find the installation instructions and list of supported libraries in it's README.

Most import wizards require some extras installed, here is the list of possible extras:

- `transformers` for [🤗 Transformers](https://huggingface.co/docs/transformers/main/en/index) integration.
- `flowtron-tts` for NVIDIA's [Flowtron](https://github.com/NVIDIA/flowtron) integration.
- `espnet` for [ESPNet2](https://espnet.github.io/espnet/espnet2_tutorial.html) integration.

Some extras might require additional dependencies installed outside of pip, they will usually report them missing via error messages.

## Tutorials

To export model you need to use `npc-engine-import-wizard import` command.
With Huggingface models you can use Huggingface Hub model id as well as path to the model.

```bash
npc-engine-import-wizard import --models-path <path-to-models-folder> <model_id or folder>
```

This will prompt you to select correct Exporter class and take you through filling in the required parameters.

We will try to cover all the import wizards in the upcoming video tutorials that will appear here.
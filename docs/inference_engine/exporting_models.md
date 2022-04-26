It's possible to export models from popular libraries to npc-engine.

Currently, npc-engine supports export from the following libraries/architectures:

- HuggingFace transformers
  - Sequence classification
  - Cosine similarity models
  - Text generation

To use this functionality you need to install a special extra called `export`

```bash
pip install npc-engine[export]
```

To export model you need to use export-model command.
With Huggingface models you can use Huggingface Hub model id as well as path to the model.

```bash
npc-engine export-model --models-path <path-to-models-folder> <model_id or folder>
```

This will prompt you to select correct Exporter class and take you through filling in the required parameters.


## Create youre own exporter

To define your own exporter you need to create child class of [Exporter](../reference/#npc_engine.exporters.base_exporter.Exporter) and import it into `npc_engine.exporters` module so that it's discovered by the CLI.

This base class has five abstract methods to implement that cover all the functionality of the exporter.





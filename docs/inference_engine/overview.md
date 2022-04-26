This is the documentation for inference 0MQ server. It describes how it works internally, how to integrate with it and how to extend it.

## How does it work

Inference server is build around 0MQ REQ/REP sockets and is using [JSON-RPC 2.0](https://www.jsonrpc.org/specification) protocol to communicate.

When starting a server models path must be provided

```
cli.exe run --models-path models
```

When the server starts it scans the folder for any valid models, loads them and exposes their API.

Each model's API is defined in [API classes](api_classes.md).  
Server exposes methods that are listed in `API_METHODS` class variable.

You can find a description of default models available in [Default Models](../models/#default-models) section.

The specifics of how model is loaded and how inference is done is defined in [specific model classes](models.md)

!!! danger "Warning"
    Right now only single model of the same API type can be loaded.

When the server is started you can run model apis via JSON-RPC requests.

!!! example ""
    ```
    {  
        "method": "do_smth",  
        "params": ["hello"],  
        "jsonrpc": "2.0",  
        "id": 0  
    }
    ```  
    will result in call to `some_model.do_smth('hello')` on the server.  

## Creating an integration

A checklist for a new integration would be to:

* Create a class that manages npc-engine subprocess (starts, terminates, checks if it's alive).
* Create a connection class that talks to npc-engine.
* Review [API classes](api_classes.md) and wrap JSON-RPC requests into the native functions.
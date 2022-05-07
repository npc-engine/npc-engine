NPC Engine is a local python [JSON-RPC](https://www.jsonrpc.org/specification) server. It supports using 0MQ TCP sockets and HTTP protocols.

The main goal of this server is to provide access to inference services that handle various functionalities required for storytelling and in-game AI.

## How does it work

Each service is defined in it's own folder in provided `--models-path`. Service's folder name becomes it's unique Id. On start NPC Engine will expose a special service called `control` that allows you to get services metadata and to control their lifetime. 

Services are running in separate processes but handle requests sequentially (including situations when services call each-other).

To route requests NPC Engine uses [0MQ identity](https://zguide.zeromq.org/docs/chapter3/#Identities-and-Addresses). It can be defined only before connecting, therefore each connected socket will be refering to it's own service. 

Service resolution rules are simple:
- If no identity provided, request is routed to the first service that implements the method called.
- If identity is API name, request is routed to the first service that implements the API.
- Same logic in case identity is a service class name.
- Identity could be a service ID (folder name), then request is routed to the service with that ID. 

Service APIs are defined in [API classes](api_classes.md).  
Service exposes methods that are listed in `API_METHODS` class variable.
You can find a description of default services available in [Default Models](../models/#default-models) section.

The specifics of how model is loaded and how inference is done is defined in [specific service classes](models.md)

Here is an example of JSON RPC request:

!!! example ""
    request with identity "some_model"
    ```
    {  
        "method": "do_smth",  
        "params": ["hello"],  
        "jsonrpc": "2.0",  
        "id": 0  
    }
    ```  
    will result in call to `some_model.do_smth('hello')` on the server, or will fail if the service wasn't started.  

## Creating an integration

A checklist for a new integration would be to:

* Create a class that manages npc-engine subprocess (starts, terminates, checks if it's alive).
* Create a connection class that talks to npc-engine.
* Review [API classes](api_classes.md) and wrap JSON-RPC requests into the native functions.
This is the documentation for Unity integration for NPC Engine.

!!! danger "Warning"
    Before using NPC Engine integration you must turn off play mode compilation
    in `Edit -> Preferences -> General -> Script changes while playing`.
    If play mode compilation will happen Unity will freeze
    and only way to restart it would be to kill the process manually!

## Dependencies

- Welcome window depends on [EditorCoroutines unity package](https://docs.unity3d.com/Packages/com.unity.editorcoroutines@1.0/manual/index.html).  
    You can add this line to your Packages\manifest.json:  

        {
            "dependencies": {
                ...
                "com.unity.editorcoroutines": "1.0.0",
                ...
            }
        }
    
- Advanced demo scene requires these free asset store packages:
    - [VIDE dialogues](https://assetstore.unity.com/packages/tools/ai/vide-dialogues-69932)
    - [Modular First Person Controller](https://assetstore.unity.com/packages/3d/characters/modular-first-person-controller-189884)
    - [Low Poly Modular Armours](https://assetstore.unity.com/packages/3d/characters/lowpoly-modular-armors-free-pack-199890)
    - [RPG Poly Pack - Lite](https://assetstore.unity.com/packages/3d/environments/landscapes/rpg-poly-pack-lite-148410)

## Getting started

NPC Engine is soon to be released on Asset Store, but for now:

- Clone [Integration repository](https://github.com/npc-engine/npc-engine-unity)
- Install [dependencies](#dependencies)
- Move integration Assets folder to your Unity project.
- Follow welcome window instructions
- Check out [Basic Demo](../basic_demo_tutorial) tutorial to see the basic usage of the NPC-engine API
- Check out [Advanced Demo](../advanced_demo_tutorial) to understand how higher-level components work and how to integrate NPC Engine into your game.
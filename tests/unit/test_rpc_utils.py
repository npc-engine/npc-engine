"""RPC utilities module tests"""


def test_schema_to_json():
    """Test if schema_to_json works."""
    from npc_engine.server.utils import schema_to_json

    s = {
        "type": "object",
        "properties": {
            "a": {"title": "a", "anyOf": ["string"]},
            "b": {"title": "b", "anyOf": ["string"]},
            "c": {"title": "c", "anyOf": ["string"]},
        },
    }
    assert schema_to_json(s) == {
        "a": "",
        "b": "",
        "c": "",
    }

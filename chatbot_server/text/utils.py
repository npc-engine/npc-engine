import os

def get_resource_path(name):
    return os.path.join(os.path.dirname(__file__), name)

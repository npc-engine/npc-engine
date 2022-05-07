from copy import deepcopy


def stub_all(cls):
    """Stub all methods of a class."""

    class StubbedClass(cls):
        pass

    method_list = [
        attribute
        for attribute in dir(StubbedClass)
        if callable(getattr(StubbedClass, attribute))
        and (attribute.startswith("__") is False)
    ]
    for method in method_list:
        setattr(StubbedClass, method, lambda *args, **kwargs: None)
    return cls

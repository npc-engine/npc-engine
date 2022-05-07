from npc_engine.service_clients import (
    ControlClient,
    TextGenerationClient,
    SimilarityClient,
)
from tests.unit.mocks import stub_all
from tests.unit.mocks.zmq_mocks import Context


TextGenerationClient = stub_all(TextGenerationClient)


class MockTextGenerationClient(TextGenerationClient):
    def __init__(self, context_template={}, *args, **kwargs):
        super().__init__(Context())
        self.context_template = context_template

    def generate_reply(self, context):
        return "test_reply"

    def get_context_template(self):
        return self.context_template


SimilarityClient = stub_all(SimilarityClient)


class MockSimilarityClient(SimilarityClient):
    def __init__(self, *args, **kwargs):
        super().__init__(Context())

    def compare(self, query, context):
        return [0.5] * len(context)


ControlClient = stub_all(ControlClient)


class MockControlClient(ControlClient):
    def __init__(self, *args, **kwargs):
        super().__init__(Context())

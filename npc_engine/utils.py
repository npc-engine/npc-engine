from enum import Enum
from dataclasses import dataclass
from dataclasses_json import dataclass_json


class StrEnum(str, Enum):

    def __str__(self):
        return self.value


@dataclass_json
@dataclass
class ServerRequest:
    jsonrpc: str
    method: str
    id: int
    params: list
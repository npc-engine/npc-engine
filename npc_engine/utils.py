"""DRY utils."""
from enum import Enum
from dataclasses import dataclass
from dataclasses_json import dataclass_json


class StrEnum(str, Enum):
    """Class for creating enum str."""
    def __str__(self):
        """getting source from enum values."""
        return self.value


@dataclass_json
@dataclass
class ServerRequest:
    """Server request typing."""
    jsonrpc: str
    method: str
    id: int
    params: list
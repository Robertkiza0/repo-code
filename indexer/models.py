from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    chunk_id: str
    file_path: str
    language: str
    type: str  # "class" | "function" | "method"
    name: str
    class_name: Optional[str]
    signature: str
    docstring: Optional[str]
    imports: List[str] = field(default_factory=list)
    source_code: str = ""
    start_line: int = 0
    end_line: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

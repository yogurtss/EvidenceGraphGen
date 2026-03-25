import math
from dataclasses import dataclass, field
from typing import List, Union

try:
    from pydantic import BaseModel, Field, field_validator

    def compat_field_validator(*fields):
        return field_validator(*fields)

except ImportError:  # pragma: no cover - pydantic v1 compatibility
    from pydantic import BaseModel, Field, validator

    def compat_field_validator(*fields):
        return validator(*fields, allow_reuse=True)


@dataclass
class Chunk:
    id: str = ""
    content: str = ""
    type: str = "text"
    meta_data: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(key: str, data: dict) -> "Chunk":
        raw_meta_data = data.get("meta_data", {})
        return Chunk(
            id=key,
            content=data.get("content", ""),
            type=data.get("type", "text"),
            meta_data=dict(raw_meta_data) if isinstance(raw_meta_data, dict) else {},
        )


@dataclass
class QAPair:
    """
    A pair of question and answer.
    """

    question: str
    answer: str

    @staticmethod
    def from_dict(data: dict) -> "QAPair":
        return QAPair(
            question=data.get("question", ""),
            answer=data.get("answer", ""),
        )


@dataclass
class Token:
    text: str
    prob: float
    top_candidates: List = field(default_factory=list)
    ppl: Union[float, None] = field(default=None)

    @property
    def logprob(self) -> float:
        return math.log(self.prob)


@dataclass
class Community:
    id: Union[int, str]
    nodes: List[str] = field(default_factory=list)
    edges: List[tuple] = field(default_factory=list)
    meta_data: dict = field(default_factory=dict)


class Node(BaseModel):
    id: str = Field(..., description="unique node id")
    op_name: str = Field(..., description="operator name")
    type: str = Field(
        ..., description="task type, e.g., map, filter, flatmap, aggregate, map_batch"
    )
    params: dict = Field(default_factory=dict, description="operator parameters")
    dependencies: List[str] = Field(
        default_factory=list, description="list of dependent node ids"
    )
    execution_params: dict = Field(
        default_factory=dict,
        description="execution parameters like replicas, batch_size",
    )
    save_output: bool = Field(
        default=False, description="whether to save the output of this node"
    )

    @classmethod
    @compat_field_validator("type")
    def validate_type(cls, v: str) -> str:
        valid_types = {"map", "filter", "flatmap", "aggregate", "map_batch"}
        if v not in valid_types:
            raise ValueError(f"Invalid node type: {v}. Must be one of {valid_types}.")
        return v


class Config(BaseModel):
    global_params: dict = Field(
        default_factory=dict, description="global context for the computation graph"
    )

    nodes: List[Node] = Field(..., description="list of nodes in the computation graph")

    @classmethod
    @compat_field_validator("nodes")
    def validate_nodes(cls, v: List[Node]) -> List[Node]:
        if not v:
            raise ValueError("At least one node is required in the computation graph.")
        ids = [node.id for node in v]
        if len(ids) != len(set(ids)):
            duplicates = {id_ for id_ in ids if ids.count(id_) > 1}
            raise ValueError(f"Duplicate node ids found: {duplicates}")
        return v

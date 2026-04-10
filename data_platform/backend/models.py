from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    id: str
    kind: Literal["node", "edge"]
    graph_item_id: str | None = None
    label: str
    evidence_span: str
    source_id: str | None = None
    description: str | None = None


class SourceContext(BaseModel):
    source_id: str
    title: str
    content: str
    content_type: Literal["text", "image_caption", "table", "unknown"] = "unknown"


class RunStats(BaseModel):
    question_texts: list[str] = Field(default_factory=list)
    answer_texts: list[str] = Field(default_factory=list)
    entity_type_counts: dict[str, int] = Field(default_factory=dict)
    relation_type_counts: dict[str, int] = Field(default_factory=dict)
    evidence_coverage: float = 0.0
    invalid_graph_sample_count: int = 0
    invalid_graph_edge_count: int = 0


class RunRecord(BaseModel):
    run_id: str
    root_path: str
    config_path: str | None = None
    generated_at: int | None = None
    sample_count: int = 0
    task_type: str = "unknown"
    has_image: bool = False
    has_sub_graph: bool = False
    stats: RunStats = Field(default_factory=RunStats)


class SampleListItem(BaseModel):
    sample_id: str
    run_id: str
    question: str
    answer_preview: str
    image_path: str | None = None
    node_count: int = 0
    edge_count: int = 0
    has_graph: bool = False


class SampleRecord(BaseModel):
    sample_id: str
    run_id: str
    source_file: str
    trace_id: str | None = None
    question: str
    answer: str
    image_path: str | None = None
    sub_graph: dict[str, Any] | None = None
    sub_graph_summary: dict[str, Any] | None = None
    visualization_trace: dict[str, Any] | None = None
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    source_contexts: list[SourceContext] = Field(default_factory=list)
    raw_record: dict[str, Any]
    graph_parse_error: str | None = None


class SamplePage(BaseModel):
    items: list[SampleListItem]
    total: int
    page: int
    page_size: int


class ScanRequest(BaseModel):
    root_path: str


class ScanResponse(BaseModel):
    root_path: str
    run_count: int
    sample_count: int
    runs: list[RunRecord]

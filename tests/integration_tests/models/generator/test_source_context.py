import asyncio

from graphgen.models.generator.aggregated_generator import AggregatedGenerator
from graphgen.models.generator.aggregated_vqa_generator import AggregatedVQAGenerator
from graphgen.models.generator.multi_hop_generator import MultiHopGenerator
from graphgen.models.generator.multi_hop_vqa_generator import MultiHopVQAGenerator
from graphgen.models.generator.source_context import SourceChunkContextBuilder
from graphgen.models.generator.vqa_generator import VQAGenerator


class FakeStorage:
    def __init__(self, data):
        self.data = data

    def get_by_id(self, key):
        return self.data.get(key)


class DummyLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def generate_answer(self, prompt, image_path=None):
        self.prompts.append((prompt, image_path))
        return self.responses.pop(0)


def _item(content, **metadata):
    return {"content": content, "metadata": metadata}


def test_source_context_limits_and_prioritizes_visual_document_chunks():
    storage = FakeStorage(
        {
            "img": _item(
                "image chunk",
                source_trace_id="doc-1",
                source_path="/tmp/doc1.md",
                source_file="doc1.md",
            ),
            "a": _item("same document A", source_trace_id="doc-1"),
            "b": _item("other document B", source_trace_id="doc-2"),
            "c": _item("same document C", source_trace_id="doc-1"),
            "d": _item("other document D", source_trace_id="doc-3"),
        }
    )
    batch = (
        [
            ("FIGURE", {"entity_type": "IMAGE", "source_id": "img"}),
            ("ENTITY", {"entity_type": "TERM", "source_id": "b<SEP>a<SEP>c<SEP>d"}),
        ],
        [],
    )

    context = SourceChunkContextBuilder([storage], chunks_per_entity=3).build(batch)

    assert "Entity 2: ENTITY" in context
    assert context.index("same document A") < context.index("other document B")
    assert context.index("same document C") < context.index("other document B")
    assert "other document D" not in context


def test_source_context_falls_back_to_visual_file_match():
    storage = FakeStorage(
        {
            "same-file": _item("same file context", source_file="doc.md"),
            "other-file": _item("other file context", source_file="other.md"),
        }
    )
    batch = (
        [
            (
                "FIGURE",
                {
                    "entity_type": "IMAGE",
                    "metadata": {"source_file": "doc.md"},
                },
            ),
            (
                "ENTITY",
                {"entity_type": "TERM", "source_id": "other-file<SEP>same-file"},
            ),
        ],
        [],
    )

    context = SourceChunkContextBuilder([storage], chunks_per_entity=2).build(batch)

    assert context.index("same file context") < context.index("other file context")


def test_source_context_outputs_source_name_without_paths_or_chunk_ids():
    storage = FakeStorage(
        {
            "chunk-1": _item(
                "content near the entity",
                source_path="/tmp/nested/paper.md",
                source_file="paper.md",
            )
        }
    )
    batch = (
        [
            (
                "ENTITY",
                {"entity_type": "TERM", "source_id": "chunk-1"},
            )
        ],
        [],
    )

    context = SourceChunkContextBuilder([storage], chunks_per_entity=3).build(batch)

    assert "Source: paper" in context
    assert "paper.md" not in context
    assert "/tmp/nested" not in context
    assert "chunk_id" not in context
    assert "chunk-1" not in context


def test_source_context_skips_missing_empty_and_duplicate_chunks():
    storage = FakeStorage(
        {
            "empty": _item(""),
            "ok": _item("usable context", source_trace_id="doc-1"),
        }
    )
    batch = (
        [
            (
                "ENTITY",
                {"entity_type": "TERM", "source_id": "missing<SEP>empty<SEP>ok<SEP>ok"},
            )
        ],
        [],
    )

    context = SourceChunkContextBuilder([storage], chunks_per_entity=3).build(batch)

    assert "usable context" in context
    assert context.count("usable context") == 1
    assert "missing" not in context


def test_prompt_templates_only_include_source_context_when_enabled():
    storage = FakeStorage({"chunk-1": _item("original source text")})
    builder = SourceChunkContextBuilder([storage], chunks_per_entity=3)
    batch = (
        [
            (
                "ENTITY",
                {
                    "entity_type": "TERM",
                    "description": "A source-grounded entity.",
                    "source_id": "chunk-1",
                },
            )
        ],
        [],
    )

    default_vqa = VQAGenerator.build_prompt(batch)
    default_aggregated = AggregatedGenerator.build_prompt(batch)
    default_multi_hop = MultiHopGenerator.build_prompt(batch)

    assert "Original Source Chunks" not in default_vqa
    assert "ORIGINAL SOURCE CHUNKS" not in default_aggregated
    assert "Original Source Chunks" not in default_multi_hop

    source_vqa = VQAGenerator.build_prompt(
        batch,
        include_source_chunks_in_prompt=True,
        source_chunk_context_builder=builder,
    )
    source_aggregated = AggregatedGenerator.build_prompt(
        batch,
        include_source_chunks_in_prompt=True,
        source_chunk_context_builder=builder,
    )
    source_multi_hop = MultiHopGenerator.build_prompt(
        batch,
        include_source_chunks_in_prompt=True,
        source_chunk_context_builder=builder,
    )

    assert "Original Source Chunks" in source_vqa
    assert "ORIGINAL SOURCE CHUNKS" in source_aggregated
    assert "Original Source Chunks" in source_multi_hop
    assert "original source text" in source_vqa
    assert "original source text" in source_aggregated
    assert "original source text" in source_multi_hop
    assert "Do not ask about source names" in source_vqa
    assert "Do not ask about source names" in source_multi_hop
    assert "Do not include source names" in source_aggregated


def test_vqa_subclass_generators_include_source_context_during_generation():
    storage = FakeStorage({"chunk-1": _item("original source text")})
    batch = (
        [
            (
                "FIGURE",
                {
                    "entity_type": "IMAGE",
                    "description": "A figure.",
                    "source_id": "chunk-1",
                    "metadata": {"img_path": "demo.png"},
                },
            )
        ],
        [],
    )

    multi_hop_llm = DummyLLM(["<question>Q?</question><answer>A.</answer>"])
    multi_hop = MultiHopVQAGenerator(
        multi_hop_llm,
        include_source_chunks_in_prompt=True,
        source_chunk_storages=[storage],
    )
    multi_hop_result = asyncio.run(multi_hop.generate(batch))

    assert "Original Source Chunks" in multi_hop_llm.prompts[0][0]
    assert multi_hop_result[0]["img_path"] == "demo.png"

    aggregated_llm = DummyLLM(
        [
            "<rephrased_text>A grounded answer.</rephrased_text>",
            "<question>Q?</question>",
        ]
    )
    aggregated = AggregatedVQAGenerator(
        aggregated_llm,
        include_source_chunks_in_prompt=True,
        source_chunk_storages=[storage],
    )
    aggregated_result = asyncio.run(aggregated.generate(batch))

    assert "ORIGINAL SOURCE CHUNKS" in aggregated_llm.prompts[0][0]
    assert "Original Source Chunks" not in aggregated_llm.prompts[1][0]
    assert aggregated_result[0]["img_path"] == "demo.png"

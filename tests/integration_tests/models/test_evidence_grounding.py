import asyncio
import logging
from unittest.mock import patch

from graphgen.bases.datatypes import Chunk
from graphgen.models.generator.aggregated_vqa_generator import AggregatedVQAGenerator
from graphgen.models.generator.vqa_generator import VQAGenerator
from graphgen.models.kg_builder.mm_kg_builder import MMKGBuilder
from graphgen.models.kg_builder.light_rag_kg_builder import LightRAGKGBuilder
from graphgen.operators.build_kg.build_kg_service import BuildKGService
from graphgen.operators.tree_pipeline import BuildGroundedTreeKGService
from graphgen.utils.log import CURRENT_LOGGER_VAR


class _DummyTokenizer:
    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())


class _DummyLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.tokenizer = _DummyTokenizer()

    async def generate_answer(self, *args, **kwargs):
        return self.responses.pop(0)


class _DummyGraphStorage:
    def index_done_callback(self):
        return None


def test_light_rag_filters_entities_and_relations_without_grounded_evidence():
    llm = _DummyLLM(
        [
            (
                '("entity"<|>"Alpha"<|>"concept"<|>"Alpha summary"<|>"Alpha is present")##'
                '("entity"<|>"Ghost"<|>"concept"<|>"Ghost summary"<|>"Ghost evidence")##'
                '("relationship"<|>"Alpha"<|>"Ghost"<|>"related_to"<|>"unsupported link"<|>"Ghost evidence"<|>0.9)'
                "<|COMPLETE|>"
            ),
            "no",
        ]
    )
    builder = LightRAGKGBuilder(
        llm_client=llm,
        require_entity_evidence=True,
        require_relation_evidence=True,
        validate_evidence_in_source=True,
    )
    token = CURRENT_LOGGER_VAR.set(logging.getLogger("test-evidence"))

    try:
        nodes, edges = asyncio.run(
            builder.extract(
                Chunk(
                    id="chunk-1",
                    type="text",
                    content="Alpha is present in the source text.",
                    metadata={},
                )
            )
        )
    finally:
        CURRENT_LOGGER_VAR.reset(token)

    assert set(nodes.keys()) == {"ALPHA"}
    assert edges == {}
    assert nodes["ALPHA"][0]["evidence_span"] == "Alpha is present"


def test_vqa_prompt_includes_grounding_evidence():
    prompt = VQAGenerator.build_prompt(
        (
            [
                (
                    "FIGURE-1",
                    {
                        "description": "A microscopy image of treated tissue.",
                        "evidence_span": "Figure 1 shows treated tissue.",
                        "metadata": '{"img_path":"demo.png"}',
                    },
                )
            ],
            [
                (
                    "FIGURE-1",
                    "LATENCY",
                    {
                        "description": "The figure reports a 12 ms latency.",
                        "relation_type": "has_latency",
                        "evidence_span": "Latency is 12 ms.",
                    },
                )
            ],
        )
    )

    assert "Evidence: Figure 1 shows treated tissue." in prompt
    assert "Evidence: Latency is 12 ms." in prompt
    assert "[has_latency]" in prompt


def test_vqa_generator_keeps_short_qa_when_other_quality_checks_pass():
    llm = _DummyLLM(["<question>图里是什么?</question><answer>DRAM</answer>"])
    generator = VQAGenerator(llm)

    result = asyncio.run(
        generator.generate(
            (
                [
                    (
                        "DRAM",
                        {
                            "description": "DRAM chip layout.",
                            "metadata": '{"img_path":"demo.png"}',
                        },
                    )
                ],
                [],
            )
        )
    )

    assert result == [
        {
            "question": "图里是什么?",
            "answer": "DRAM",
            "img_path": "demo.png",
        }
    ]


def test_mm_kg_builder_accepts_entity_records_without_evidence():
    llm = _DummyLLM(
        [
            (
                '("entity"<|>"image-1"<|>"image"<|>"A microscopy image of treated tissue.")##'
                '("entity"<|>"Tissue"<|>"component"<|>"Tissue highlighted in the image.")##'
                '("relationship"<|>"image-1"<|>"Tissue"<|>"contains"<|>"The image contains tissue."<|>"treated tissue"<|>0.9)'
                "<|COMPLETE|>"
            )
        ]
    )
    builder = MMKGBuilder(
        llm_client=llm,
        require_entity_evidence=False,
        require_relation_evidence=True,
        validate_evidence_in_source=True,
    )

    nodes, edges = asyncio.run(
        builder.extract(
            Chunk(
                id="image-1",
                type="image",
                content="",
                metadata={"image_caption": ["Figure 1 shows treated tissue."]},
            )
        )
    )

    assert set(nodes.keys()) == {"IMAGE-1", "TISSUE"}
    assert nodes["IMAGE-1"][0]["evidence_span"] == ""
    assert nodes["TISSUE"][0]["evidence_span"] == ""
    assert ("IMAGE-1", "TISSUE") in edges


def test_build_grounded_tree_kg_service_splits_text_and_mm_entity_evidence_policy(
    tmp_path,
):
    with patch(
        "graphgen.operators.tree_pipeline.build_tree_kg_service.init_llm",
        return_value=_DummyLLM([]),
    ), patch(
        "graphgen.operators.tree_pipeline.build_tree_kg_service.init_storage",
        return_value=_DummyGraphStorage(),
    ):
        service = BuildGroundedTreeKGService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
        )

    assert service.text_require_entity_evidence is True
    assert service.mm_require_entity_evidence is False
    assert service.text_require_relation_evidence is True
    assert service.mm_require_relation_evidence is True
    assert service.text_validate_evidence_in_source is True
    assert service.mm_validate_evidence_in_source is True


def test_build_kg_service_uses_split_evidence_settings_for_text_and_mm(tmp_path):
    captured = {}

    def _fake_text_kg(**kwargs):
        captured["text"] = kwargs
        return [], []

    def _fake_mm_kg(**kwargs):
        captured["mm"] = kwargs
        return [], []

    with patch(
        "graphgen.operators.build_kg.build_kg_service.init_llm",
        return_value=_DummyLLM([]),
    ), patch(
        "graphgen.operators.build_kg.build_kg_service.init_storage",
        return_value=_DummyGraphStorage(),
    ), patch(
        "graphgen.operators.build_kg.build_kg_service.build_text_kg",
        side_effect=_fake_text_kg,
    ), patch(
        "graphgen.operators.build_kg.build_kg_service.build_mm_kg",
        side_effect=_fake_mm_kg,
    ):
        service = BuildKGService(
            working_dir=str(tmp_path / "cache"),
            require_entity_evidence=True,
            require_relation_evidence=True,
            validate_evidence_in_source=True,
            mm_require_entity_evidence=False,
        )
        service.process(
            [
                {
                    "_trace_id": "text-1",
                    "type": "text",
                    "content": "Alpha",
                    "metadata": {},
                },
                {
                    "_trace_id": "image-1",
                    "type": "image",
                    "content": "",
                    "metadata": {"image_caption": ["Figure 1 shows Alpha."]},
                },
            ]
        )

    assert captured["text"]["require_entity_evidence"] is True
    assert captured["text"]["require_relation_evidence"] is True
    assert captured["text"]["validate_evidence_in_source"] is True
    assert captured["mm"]["require_entity_evidence"] is False
    assert captured["mm"]["require_relation_evidence"] is True
    assert captured["mm"]["validate_evidence_in_source"] is True


def test_aggregated_vqa_generator_attaches_image_path():
    llm = _DummyLLM(
        [
            "<rephrased_text>图中展示了 DRAM 结构。</rephrased_text>",
            "<question>图中展示了什么结构?</question>",
        ]
    )
    generator = AggregatedVQAGenerator(llm)
    token = CURRENT_LOGGER_VAR.set(logging.getLogger("test-aggregated-vqa"))

    try:
        result = asyncio.run(
            generator.generate(
                (
                    [
                        (
                            "FIG-1",
                            {
                                "description": "DRAM architecture",
                                "metadata": '{"img_path":"demo.png"}',
                            },
                        )
                    ],
                    [],
                )
            )
        )
    finally:
        CURRENT_LOGGER_VAR.reset(token)

    assert result == [
        {
            "question": "图中展示了什么结构?",
            "answer": "图中展示了 DRAM 结构。",
            "img_path": "demo.png",
        }
    ]


def test_aggregated_vqa_generator_formats_chatml_with_image():
    result = AggregatedVQAGenerator.format_generation_results(
        {
            "question": "图中展示了什么结构?",
            "answer": "DRAM 结构。",
            "img_path": "demo.png",
        },
        output_data_format="ChatML",
    )

    assert result["messages"][0]["content"] == [
        {"text": "图中展示了什么结构?", "image": "demo.png"}
    ]

import json

from graphgen.models.generator.context_utils import build_grounded_context
from graphgen.models.generator.vqa_generator import VQAGenerator


def test_build_grounded_context_includes_visual_metadata_only_when_enabled():
    batch = (
        [
            (
                "image_seed",
                {
                    "entity_type": "IMAGE",
                    "description": "Timing figure.",
                    "metadata": {
                        "image_caption": ["Fig. 1 shows rowbuffer timing."],
                        "note_text": "Nearby text mentions prefetch latency.",
                    },
                },
            )
        ],
        [],
    )

    default_entities, _ = build_grounded_context(batch)
    assert "Image Caption:" not in default_entities
    assert "Notes:" not in default_entities

    enriched_entities, _ = build_grounded_context(
        batch,
        include_visual_metadata=True,
    )
    assert "Image Caption: Fig. 1 shows rowbuffer timing." in enriched_entities
    assert "Notes: Nearby text mentions prefetch latency." in enriched_entities


def test_build_grounded_context_supports_json_visual_metadata():
    batch = (
        [
            (
                "virtual_image",
                {
                    "entity_type": "IMAGE",
                    "description": "Virtual image root.",
                    "metadata": json.dumps(
                        {
                            "caption": "The diagram highlights activation windows.",
                            "notes": ["Refresh note", "Voltage guard note"],
                        }
                    ),
                },
            )
        ],
        [],
    )

    entities, _ = build_grounded_context(batch, include_visual_metadata=True)
    assert "Image Caption: The diagram highlights activation windows." in entities
    assert "Notes: Refresh note\nVoltage guard note" in entities


def test_vqa_context_keywords_include_visual_metadata():
    batch = (
        [
            (
                "image_seed",
                {
                    "entity_type": "IMAGE",
                    "description": "Timing figure.",
                    "metadata": {
                        "image_caption": "Rowbuffer timing map.",
                        "note_text": "Prefetch latency guard.",
                    },
                },
            )
        ],
        [],
    )

    keywords = VQAGenerator._build_context_keywords(batch)
    assert "rowbuffer" in keywords
    assert "prefetch" in keywords

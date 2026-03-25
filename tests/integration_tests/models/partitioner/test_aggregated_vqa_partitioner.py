import tempfile

from graphgen.models import AggregatedVQAPartitioner
from graphgen.storage import NetworkXStorage


def test_aggregated_vqa_partitioner_respects_section_scope():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = NetworkXStorage(working_dir=tmpdir, namespace="agg_vqa_scope")
        storage.upsert_node("IMG_A", {"entity_type": "IMAGE", "meta_data": {"path": "sec/A"}})
        storage.upsert_node("TXT_A", {"entity_type": "TEXT", "meta_data": {"path": "sec/A"}})
        storage.upsert_node("TXT_B", {"entity_type": "TEXT", "meta_data": {"path": "sec/B"}})

        storage.upsert_edge("IMG_A", "TXT_A", {"relation_type": "described_by"})
        storage.upsert_edge("IMG_A", "TXT_B", {"relation_type": "mentions"})

        partitioner = AggregatedVQAPartitioner(anchor_type=["image"])
        communities = list(
            partitioner.partition(
                storage,
                max_units_per_community=10,
                min_units_per_community=2,
                section_scoped=True,
                required_modalities=["image", "text"],
            )
        )

        assert len(communities) == 1
        assert "IMG_A" in communities[0].nodes
        assert "TXT_A" in communities[0].nodes
        assert "TXT_B" not in communities[0].nodes


def test_aggregated_vqa_partitioner_filters_by_modalities():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = NetworkXStorage(working_dir=tmpdir, namespace="agg_vqa_modal")
        storage.upsert_node("IMG_ONLY", {"entity_type": "IMAGE", "meta_data": {"path": "sec/A"}})

        partitioner = AggregatedVQAPartitioner(anchor_type=["image"])
        communities = list(
            partitioner.partition(
                storage,
                max_units_per_community=5,
                min_units_per_community=1,
                section_scoped=True,
                required_modalities=["image", "text"],
            )
        )

        assert communities == []

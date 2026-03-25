from graphgen.operators import operators
from graphgen.operators.partition.aggregated_vqa_partition_service import (
    AggregatedVQAPartitionService,
)


def test_aggregated_vqa_partition_operator_registered():
    assert operators["aggregated_vqa_partition"] is AggregatedVQAPartitionService

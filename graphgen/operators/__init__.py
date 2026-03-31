from .build_kg import BuildKGService
from .chunk import ChunkService
from .evaluate import EvaluateService
from .extract import ExtractService
from .filter import FilterService
from .generate import GenerateService
from .judge import JudgeService
from .partition import AggregatedVQAPartitionService, PartitionService
from .quiz import QuizService
from .read import read
from .rephrase import RephraseService
from .sample_subgraph import SampleSubgraphService
from .sample_subgraph_v2 import SampleSubgraphV2Service
from .search import SearchService
from .tree_pipeline import (
    BuildGroundedTreeKGService,
    BuildTreeKGService,
    FilterEntitiesService,
    HierarchyGenerateService,
    StructureAnalyzeService,
    TreeChunkService,
    TreeConstructService,
)

operators = {
    "read": read,
    "chunk": ChunkService,
    "build_kg": BuildKGService,
    "quiz": QuizService,
    "judge": JudgeService,
    "extract": ExtractService,
    "search": SearchService,
    "partition": PartitionService,
    "aggregated_vqa_partition": AggregatedVQAPartitionService,
    "generate": GenerateService,
    "evaluate": EvaluateService,
    "rephrase": RephraseService,
    "filter": FilterService,
    "sample_subgraph": SampleSubgraphService,
    "sample_subgraph_v2": SampleSubgraphV2Service,
    "structure_analyze": StructureAnalyzeService,
    "hierarchy_generate": HierarchyGenerateService,
    "tree_construct": TreeConstructService,
    "tree_chunk": TreeChunkService,
    "build_tree_kg": BuildTreeKGService,
    "build_grounded_tree_kg": BuildGroundedTreeKGService,
    "filter_entities": FilterEntitiesService,
}

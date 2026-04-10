export type EvidenceItem = {
  id: string;
  kind: "node" | "edge";
  graph_item_id?: string | null;
  label: string;
  evidence_span: string;
  source_id?: string | null;
  description?: string | null;
};

export type SourceContext = {
  source_id: string;
  title: string;
  content: string;
  content_type: "text" | "image" | "image_caption" | "table" | "unknown";
};

export type RunStats = {
  question_texts: string[];
  answer_texts: string[];
  entity_type_counts: Record<string, number>;
  relation_type_counts: Record<string, number>;
  evidence_coverage: number;
  invalid_graph_sample_count: number;
  invalid_graph_edge_count: number;
};

export type ImportedRun = {
  run_id: string;
  root_path: string;
  config_path?: string | null;
  generated_at?: number | null;
  sample_count: number;
  task_type: string;
  has_image: boolean;
  has_sub_graph: boolean;
  stats: RunStats;
};

export type SampleListItem = {
  sample_id: string;
  run_id: string;
  question: string;
  answer_preview: string;
  image_path?: string | null;
  node_count: number;
  edge_count: number;
  has_graph: boolean;
};

export type GraphNodeRecord = [string, Record<string, unknown>];
export type GraphEdgeRecord = [string, string, Record<string, unknown>];

export type GraphCatalog = {
  nodes?: Record<string, Record<string, unknown>>;
  edges?: Record<string, Record<string, unknown>>;
};

export type VisualizationCandidate = {
  candidate_uid?: string;
  candidate_node_id?: string;
  bind_from_node_id?: string;
  bound_edge_pair?: string[];
  relation_type?: string;
  entity_type?: string;
  evidence_summary?: string;
  score?: number;
  depth?: number;
  [key: string]: unknown;
};

export type VisualizationEvent = {
  event_id: string;
  order: number;
  qa_family: string;
  phase: string;
  event_type: string;
  status: string;
  selected_node_ids?: string[];
  selected_edge_pairs?: string[][];
  candidate_pool?: VisualizationCandidate[];
  chosen_candidate?: VisualizationCandidate | Record<string, unknown>;
  judge?: Record<string, unknown>;
  reason?: string;
  termination_reason?: string;
  subgraph_id?: string;
  generator_key?: string;
  question?: string;
  answer?: string;
  qa_trace_id?: string;
  sub_graph_summary?: Record<string, unknown>;
};

export type VisualizationTrace = {
  schema_version: string;
  sampler_version: string;
  seed_node_id?: string;
  seed_image_path?: string;
  graph_catalog?: GraphCatalog;
  events?: VisualizationEvent[];
};

export type SampleDetail = {
  sample_id: string;
  run_id: string;
  source_file: string;
  trace_id?: string | null;
  question: string;
  answer: string;
  image_path?: string | null;
  sub_graph?: {
    nodes?: GraphNodeRecord[];
    edges?: GraphEdgeRecord[];
  } | null;
  sub_graph_summary?: {
    node_count?: number;
    edge_count?: number;
    node_ids?: string[];
    edge_pairs?: string[];
  } | null;
  evidence_items: EvidenceItem[];
  source_contexts: SourceContext[];
  visualization_trace?: VisualizationTrace | null;
  raw_record: Record<string, unknown>;
  graph_parse_error?: string | null;
};

export type SamplePage = {
  items: SampleListItem[];
  total: number;
  page: number;
  page_size: number;
};

export type GraphSelection = {
  kind: "node" | "edge";
  id: string;
  label: string;
  title: string;
  entityType?: string;
  relationType?: string;
  description?: string;
  evidenceSpan?: string;
  sourceId?: string;
  metadata?: Array<{ key: string; value: string }>;
  connectedTo?: string;
};

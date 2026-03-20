export type EvidenceItem = {
  kind: "node" | "edge";
  label: string;
  evidence_span: string;
  source_id?: string | null;
  description?: string | null;
};

export type RunStats = {
  question_texts: string[];
  answer_texts: string[];
  entity_type_counts: Record<string, number>;
  relation_type_counts: Record<string, number>;
  evidence_coverage: number;
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
  entityType?: string;
  relationType?: string;
  description?: string;
  evidenceSpan?: string;
  sourceId?: string;
};

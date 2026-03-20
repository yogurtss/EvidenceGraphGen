import cytoscape from "cytoscape";
import { startTransition, useEffect, useRef, useState } from "react";

import { buildAssetUrl, fetchSampleDetail, fetchSamples, scanRuns } from "./api";
import type {
  GraphEdgeRecord,
  GraphNodeRecord,
  GraphSelection,
  ImportedRun,
  SampleDetail,
  SampleListItem,
} from "./types";

const PAGE_SIZE = 12;

function GraphCanvas(props: {
  sample: SampleDetail | null;
  onSelect: (selection: GraphSelection | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<cytoscape.Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    graphRef.current?.destroy();

    const subGraph = props.sample?.sub_graph;
    if (!subGraph || !subGraph.nodes?.length) {
      return;
    }

    const elements = buildGraphElements(subGraph.nodes, subGraph.edges || []);
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      layout: { name: "cose", animate: false, padding: 20 },
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "background-color": "data(color)",
            color: "#f7f3ea",
            "text-wrap": "wrap",
            "text-max-width": 110,
            "font-size": 11,
            "text-valign": "center",
            "text-halign": "center",
            width: 36,
            height: 36,
            "border-width": 2,
            "border-color": "#f7f3ea",
          },
        },
        {
          selector: "edge",
          style: {
            width: 2,
            label: "data(label)",
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "line-color": "#89a8b2",
            "target-arrow-color": "#89a8b2",
            color: "#f2efe5",
            "font-size": 10,
            "text-background-color": "#20333a",
            "text-background-opacity": 0.9,
            "text-background-padding": 3,
          },
        },
        {
          selector: ":selected",
          style: {
            "border-color": "#ffb347",
            "line-color": "#ffb347",
            "target-arrow-color": "#ffb347",
            "border-width": 4,
          },
        },
      ],
    });

    cy.on("tap", "node", (event) => {
      const target = event.target;
      props.onSelect({
        kind: "node",
        id: target.id(),
        label: target.data("label") as string,
        entityType: target.data("entityType") as string | undefined,
        description: target.data("description") as string | undefined,
        evidenceSpan: target.data("evidenceSpan") as string | undefined,
        sourceId: target.data("sourceId") as string | undefined,
      });
    });

    cy.on("tap", "edge", (event) => {
      const target = event.target;
      props.onSelect({
        kind: "edge",
        id: target.id(),
        label: target.data("label") as string,
        relationType: target.data("relationType") as string | undefined,
        description: target.data("description") as string | undefined,
        evidenceSpan: target.data("evidenceSpan") as string | undefined,
        sourceId: target.data("sourceId") as string | undefined,
      });
    });

    cy.on("tap", (event) => {
      if (event.target === cy) {
        props.onSelect(null);
      }
    });

    graphRef.current = cy;
    return () => {
      cy.destroy();
    };
  }, [props.sample]);

  return (
    <div className="graph-shell">
      <div className="graph-toolbar">
        <span>Sub Graph</span>
        <button
          type="button"
          onClick={() => {
            graphRef.current?.fit(undefined, 24);
          }}
        >
          Fit
        </button>
      </div>
      {props.sample?.sub_graph?.nodes?.length ? (
        <div className="graph-canvas" ref={containerRef} />
      ) : (
        <div className="empty-panel">
          {props.sample?.graph_parse_error
            ? `Graph parse error: ${props.sample.graph_parse_error}`
            : "No visualizable sub_graph for this sample."}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [rootPath, setRootPath] = useState("cache");
  const [runs, setRuns] = useState<ImportedRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [samples, setSamples] = useState<SampleListItem[]>([]);
  const [selectedSample, setSelectedSample] = useState<SampleDetail | null>(null);
  const [selectedGraphItem, setSelectedGraphItem] = useState<GraphSelection | null>(null);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [hasImageOnly, setHasImageOnly] = useState(false);
  const [hasGraphOnly, setHasGraphOnly] = useState(false);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    void handleScan("cache");
  }, []);

  useEffect(() => {
    if (!selectedRunId) {
      return;
    }
    void loadSamples(selectedRunId, page);
  }, [selectedRunId, page, hasImageOnly, hasGraphOnly]);

  async function handleScan(path: string) {
    setLoadingRuns(true);
    setError("");
    try {
      const result = await scanRuns(path);
      startTransition(() => {
        setRuns(result.runs);
        setSelectedRunId(result.runs[0]?.run_id || "");
        setPage(1);
      });
    } catch (scanError) {
      setError(scanError instanceof Error ? scanError.message : "Failed to scan runs");
    } finally {
      setLoadingRuns(false);
    }
  }

  async function loadSamples(runId: string, nextPage: number) {
    setLoadingSamples(true);
    setError("");
    try {
      const result = await fetchSamples({
        runId,
        page: nextPage,
        pageSize: PAGE_SIZE,
        search,
        hasImage: hasImageOnly ? true : undefined,
        hasGraph: hasGraphOnly ? true : undefined,
      });
      startTransition(() => {
        setSamples(result.items);
        setTotal(result.total);
      });
      if (result.items[0]) {
        void handleSampleSelect(result.items[0].sample_id);
      } else {
        setSelectedSample(null);
        setSelectedGraphItem(null);
      }
    } catch (sampleError) {
      setError(sampleError instanceof Error ? sampleError.message : "Failed to load samples");
    } finally {
      setLoadingSamples(false);
    }
  }

  async function handleSampleSelect(sampleId: string) {
    setLoadingDetail(true);
    setError("");
    try {
      const detail = await fetchSampleDetail(sampleId);
      startTransition(() => {
        setSelectedSample(detail);
        setSelectedGraphItem(null);
      });
    } catch (detailError) {
      setError(detailError instanceof Error ? detailError.message : "Failed to load detail");
    } finally {
      setLoadingDetail(false);
    }
  }

  async function handleSearchSubmit() {
    if (!selectedRunId) {
      return;
    }
    setPage(1);
    await loadSamples(selectedRunId, 1);
  }

  const selectedRun = runs.find((run) => run.run_id === selectedRunId) || null;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">GraphGen Data Platform</p>
          <h1>Run explorer for VQA and graph-grounded samples</h1>
        </div>
        <form
          className="import-form"
          onSubmit={(event) => {
            event.preventDefault();
            void handleScan(rootPath);
          }}
        >
          <input
            value={rootPath}
            onChange={(event) => setRootPath(event.target.value)}
            placeholder="cache"
          />
          <button type="submit" disabled={loadingRuns}>
            {loadingRuns ? "Scanning..." : "Import Directory"}
          </button>
        </form>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <main className="workspace">
        <section className="panel left-panel">
          <div className="panel-heading">
            <h2>Runs</h2>
            <span>{runs.length}</span>
          </div>
          <div className="run-list">
            {runs.map((run) => (
              <button
                key={run.run_id}
                type="button"
                className={`run-card ${selectedRunId === run.run_id ? "active" : ""}`}
                onClick={() => {
                  setSelectedRunId(run.run_id);
                  setPage(1);
                }}
              >
                <div className="run-card-top">
                  <strong>{run.run_id}</strong>
                  <span>{run.task_type}</span>
                </div>
                <p>{run.sample_count} samples</p>
                <div className="run-badges">
                  {run.has_image ? <span>Image</span> : null}
                  {run.has_sub_graph ? <span>Graph</span> : null}
                </div>
              </button>
            ))}
          </div>

          {selectedRun ? (
            <div className="run-meta">
              <h3>Run stats</h3>
              <p>Evidence coverage: {(selectedRun.stats.evidence_coverage * 100).toFixed(1)}%</p>
              <p>Entity types: {Object.keys(selectedRun.stats.entity_type_counts).length}</p>
              <p>Relation types: {Object.keys(selectedRun.stats.relation_type_counts).length}</p>
            </div>
          ) : (
            <div className="empty-panel">Import a GraphGen output directory to begin.</div>
          )}
        </section>

        <section className="panel middle-panel">
          <div className="panel-heading">
            <h2>Samples</h2>
            <span>{total}</span>
          </div>
          <div className="sample-toolbar">
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search question or answer"
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  void handleSearchSubmit();
                }
              }}
            />
            <button type="button" onClick={() => void handleSearchSubmit()}>
              Search
            </button>
          </div>
          <div className="filter-row">
            <label>
              <input
                type="checkbox"
                checked={hasImageOnly}
                onChange={(event) => {
                  setHasImageOnly(event.target.checked);
                  setPage(1);
                }}
              />
              Has image
            </label>
            <label>
              <input
                type="checkbox"
                checked={hasGraphOnly}
                onChange={(event) => {
                  setHasGraphOnly(event.target.checked);
                  setPage(1);
                }}
              />
              Has graph
            </label>
          </div>
          <div className="sample-list">
            {loadingSamples ? <div className="empty-panel">Loading samples...</div> : null}
            {!loadingSamples &&
              samples.map((sample) => (
                <button
                  key={sample.sample_id}
                  type="button"
                  className={`sample-card ${
                    selectedSample?.sample_id === sample.sample_id ? "active" : ""
                  }`}
                  onClick={() => void handleSampleSelect(sample.sample_id)}
                >
                  <strong>{sample.question || "Untitled sample"}</strong>
                  <p>{sample.answer_preview || "No answer"}</p>
                  <div className="sample-meta">
                    <span>{sample.node_count} nodes</span>
                    <span>{sample.edge_count} edges</span>
                    {sample.image_path ? <span>Image</span> : null}
                  </div>
                </button>
              ))}
          </div>
          <div className="pager">
            <button type="button" disabled={page <= 1} onClick={() => setPage(page - 1)}>
              Prev
            </button>
            <span>
              {page} / {pageCount}
            </span>
            <button
              type="button"
              disabled={page >= pageCount}
              onClick={() => setPage(page + 1)}
            >
              Next
            </button>
          </div>
        </section>

        <section className="panel right-panel">
          <div className="panel-heading">
            <h2>Detail</h2>
            <span>{loadingDetail ? "Loading..." : selectedSample?.run_id || "-"}</span>
          </div>
          {selectedSample ? (
            <div className="detail-stack">
              <article className="qa-card">
                <h3>Question</h3>
                <p>{selectedSample.question || "No question text"}</p>
                <h3>Answer</h3>
                <p>{selectedSample.answer || "No answer text"}</p>
              </article>

              {selectedSample.image_path ? (
                <article className="asset-card">
                  <div className="asset-card-top">
                    <h3>Image</h3>
                    <span>{selectedSample.image_path.split("/").pop()}</span>
                  </div>
                  <img src={buildAssetUrl(selectedSample.image_path)} alt={selectedSample.question} />
                </article>
              ) : null}

              <GraphCanvas sample={selectedSample} onSelect={setSelectedGraphItem} />

              <article className="inspector-card">
                <h3>Selection</h3>
                {selectedGraphItem ? (
                  <div className="selection-grid">
                    <p>
                      <span>Label</span>
                      {selectedGraphItem.label}
                    </p>
                    <p>
                      <span>Type</span>
                      {selectedGraphItem.entityType || selectedGraphItem.relationType || "-"}
                    </p>
                    <p>
                      <span>Source</span>
                      {selectedGraphItem.sourceId || "-"}
                    </p>
                    <p>
                      <span>Description</span>
                      {selectedGraphItem.description || "-"}
                    </p>
                    <p>
                      <span>Evidence</span>
                      {selectedGraphItem.evidenceSpan || "-"}
                    </p>
                  </div>
                ) : (
                  <div className="empty-panel">Click a node or edge to inspect metadata.</div>
                )}
              </article>

              <article className="evidence-card">
                <div className="panel-heading compact">
                  <h3>Evidence</h3>
                  <span>{selectedSample.evidence_items.length}</span>
                </div>
                <div className="evidence-list">
                  {selectedSample.evidence_items.length ? (
                    selectedSample.evidence_items.map((item, index) => (
                      <div key={`${item.kind}-${item.label}-${index}`} className="evidence-item">
                        <div className="evidence-top">
                          <strong>{item.label}</strong>
                          <span>{item.kind}</span>
                        </div>
                        <p>{item.evidence_span}</p>
                        <small>{item.source_id || "No source id"}</small>
                      </div>
                    ))
                  ) : (
                    <div className="empty-panel">No evidence spans found for this sample.</div>
                  )}
                </div>
              </article>
            </div>
          ) : (
            <div className="empty-panel">Select a sample to inspect its QA pair and graph.</div>
          )}
        </section>
      </main>
    </div>
  );
}

function buildGraphElements(nodes: GraphNodeRecord[], edges: GraphEdgeRecord[]) {
  const elements: cytoscape.ElementDefinition[] = [];

  for (const [nodeId, metadata] of nodes) {
    const entityType = String(metadata["entity_type"] || "unknown");
    elements.push({
      data: {
        id: nodeId,
        label: metadata["entity_name"] || nodeId,
        entityType,
        description: metadata["description"] || "",
        evidenceSpan: metadata["evidence_span"] || "",
        sourceId: metadata["source_id"] || "",
        color: colorForEntityType(entityType),
      },
    });
  }

  edges.forEach(([src, tgt, metadata], index) => {
    elements.push({
      data: {
        id: `${src}-${tgt}-${index}`,
        source: src,
        target: tgt,
        label: metadata["relation_type"] || `${src} -> ${tgt}`,
        relationType: metadata["relation_type"] || "",
        description: metadata["description"] || "",
        evidenceSpan: metadata["evidence_span"] || "",
        sourceId: metadata["source_id"] || "",
      },
    });
  });

  return elements;
}

function colorForEntityType(entityType: string) {
  const palette = [
    "#d96c4f",
    "#2a6f97",
    "#6b8f71",
    "#c08457",
    "#7b5ea7",
    "#b56576",
    "#3e7c59",
  ];

  let hash = 0;
  for (let index = 0; index < entityType.length; index += 1) {
    hash = entityType.charCodeAt(index) + ((hash << 5) - hash);
  }
  return palette[Math.abs(hash) % palette.length];
}

import cytoscape from "cytoscape";
import { startTransition, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import { buildAssetUrl, fetchSampleDetail, fetchSamples, scanRuns } from "./api";
import type {
  EvidenceItem,
  GraphEdgeRecord,
  GraphNodeRecord,
  GraphSelection,
  ImportedRun,
  SampleDetail,
  SampleListItem,
  SourceContext,
} from "./types";

const PAGE_SIZE = 5;
const DEFAULT_ROOT_PATH = "/home/lukashe/data/projects/EvidenceGraphGen/cache";

function GraphCanvas(props: {
  sample: SampleDetail | null;
  activeGraphItemId: string | null;
  onSelect: (selection: GraphSelection | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<cytoscape.Core | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const graphIssues = useMemo(() => {
    if (!props.sample?.sub_graph?.nodes?.length) {
      return { validEdges: [] as GraphEdgeRecord[], invalidEdges: [] as Array<{ src: string; tgt: string; missingNodes: string[]; metadata: Record<string, unknown> }> };
    }
    return inspectGraphReferences(props.sample.sub_graph.nodes || [], props.sample.sub_graph.edges || []);
  }, [props.sample]);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    graphRef.current?.destroy();
    setGraphError(null);

    const subGraph = props.sample?.sub_graph;
    if (!subGraph || !subGraph.nodes?.length) {
      return;
    }
    try {
      const elements = buildGraphElements(
        subGraph.nodes,
        graphIssues.validEdges,
        props.sample?.image_path,
      );
      const cy = cytoscape({
        container: containerRef.current,
        elements,
        layout: { name: "cose", animate: false, padding: 24, idealEdgeLength: 76 },
        style: [
          {
            selector: "node",
            style: {
              label: "data(label)",
              "background-color": "data(color)",
              shape: "ellipse",
              color: "#24364c",
              "text-wrap": "wrap",
              "text-max-width": "64px",
              "font-size": 8,
              "font-weight": 600,
              "text-valign": "center",
              "text-halign": "center",
              width: 50,
              height: 50,
              "border-width": 1.5,
              "border-color": "#f8fbff",
              "overlay-padding": 8,
              opacity: 1,
            },
          },
          {
            selector: "edge",
            style: {
              width: 1.8,
              label: "data(label)",
              "curve-style": "bezier",
              "target-arrow-shape": "triangle",
              "line-color": "#a9b9ce",
              "target-arrow-color": "#a9b9ce",
              color: "rgba(146, 169, 199, 0.72)",
              "font-size": 6,
              "font-weight": 500,
              "text-wrap": "wrap",
              "text-max-width": "78px",
              "text-background-color": "rgba(10, 20, 34, 0.22)",
              "text-background-opacity": 0.25,
              "text-background-padding": "1px",
              "text-border-width": 0,
              "text-border-opacity": 0,
              opacity: 0.72,
            },
          },
          {
            selector: ".dimmed",
            style: {
              opacity: 0.34,
              "text-opacity": 0.34,
            },
          },
          {
            selector: "node.related",
            style: {
              opacity: 0.96,
              "border-color": "#89c7ff",
              "border-width": 2.5,
            },
          },
          {
            selector: "edge.related",
            style: {
              opacity: 0.88,
              width: 2.1,
              "line-color": "#7dbaff",
              "target-arrow-color": "#7dbaff",
              color: "#d7ebff",
              "font-size": 7,
              "font-weight": 600,
              "text-background-color": "rgba(14, 28, 47, 0.94)",
              "text-background-opacity": 1,
              "text-background-padding": "3px",
              "text-border-width": 1,
              "text-border-color": "rgba(125, 186, 255, 0.28)",
              "text-border-opacity": 1,
            },
          },
          {
            selector: ".focus",
            style: {
              opacity: 1,
              "z-index": 999,
            },
          },
          {
            selector: ":selected",
            style: {
              "border-color": "#5b8def",
              "line-color": "#5b8def",
              "target-arrow-color": "#5b8def",
              "border-width": 4,
              "underlay-color": "rgba(91, 141, 239, 0.18)",
              "underlay-opacity": 1,
              "underlay-padding": 8,
            },
          },
        ],
      });

      cy.on("tap", "node", (event) => {
        props.onSelect(toGraphSelection("node", event.target.data(), event.target.id()));
      });

      cy.on("tap", "edge", (event) => {
        props.onSelect(toGraphSelection("edge", event.target.data(), event.target.id()));
      });

      graphRef.current = cy;
    } catch (error) {
      setGraphError(error instanceof Error ? error.message : "Unable to render this graph.");
    }

    return () => {
      graphRef.current?.destroy();
      graphRef.current = null;
    };
  }, [props.sample]);

  useEffect(() => {
    const cy = graphRef.current;
    if (!cy) {
      return;
    }

    cy.elements().unselect();
    cy.elements().removeClass("dimmed related focus");
    if (!props.activeGraphItemId) {
      return;
    }

    const target = cy.getElementById(props.activeGraphItemId);
    if (target.nonempty()) {
      target.select();
      target.addClass("focus");

      if (target.isNode()) {
        const related = target.closedNeighborhood();
        cy.elements().difference(related).addClass("dimmed");
        related.nodes().addClass("related");
        related.edges().addClass("related");
      } else {
        const edge = target as cytoscape.EdgeSingular;
        const related = edge.connectedNodes().union(edge);
        cy.elements().difference(related).addClass("dimmed");
        edge.connectedNodes().addClass("related");
        edge.addClass("related");
      }
    }
  }, [props.activeGraphItemId]);

  return (
    <article className="content-card graph-card">
      <div className="section-top">
        <div>
          <p className="section-kicker">Graph</p>
          <h3>Sub Graph</h3>
        </div>
        <button
          type="button"
          className="ghost-button"
          onClick={() => {
            graphRef.current?.fit(undefined, 32);
          }}
        >
          Fit
        </button>
      </div>
      {graphIssues.invalidEdges.length ? (
          <div className="graph-warning">
            <strong>Broken graph references</strong>
            <p>
              {graphIssues.invalidEdges.length} edge
              {graphIssues.invalidEdges.length > 1
                ? "s reference missing nodes"
                : " references a missing node"}
              . Valid nodes are still shown below.
            </p>
            <div className="graph-issue-list">
              {graphIssues.invalidEdges.map((issue, index) => (
                <div key={`${issue.src}-${issue.tgt}-${index}`} className="graph-issue-chip">
                  <span>{`${issue.src} -> ${issue.tgt}`}</span>
                  <small>{`missing ${issue.missingNodes.join(", ")}`}</small>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      {props.sample?.sub_graph?.nodes?.length ? (
        <div className="graph-canvas" ref={containerRef} />
      ) : (
        <div className="empty-panel">
          {props.sample?.graph_parse_error
            ? `Graph parse error: ${props.sample.graph_parse_error}`
            : "No visualizable sub_graph for this sample."}
        </div>
      )}
      {graphError ? <div className="empty-panel">{`Graph render issue: ${graphError}`}</div> : null}
    </article>
  );
}

function ClampText(props: {
  text: string;
  className?: string;
  emptyLabel?: string;
  preserveWhitespace?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const text = props.text.trim();
  const shouldClamp = text.length > 180 || text.split("\n").length > 3;

  useEffect(() => {
    setExpanded(false);
  }, [props.text]);

  if (!text) {
    return <p className={props.className}>{props.emptyLabel || "-"}</p>;
  }

  return (
    <div className="clamp-block">
      <p
        className={[
          props.className || "",
          shouldClamp && !expanded ? "clamp-3" : "",
          props.preserveWhitespace ? "preserve-whitespace" : "",
        ]
          .filter(Boolean)
          .join(" ")}
      >
        {text}
      </p>
      {shouldClamp ? (
        <button type="button" className="text-toggle" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "Show less" : "Show more"}
        </button>
      ) : null}
    </div>
  );
}

function EvidenceViewer(props: {
  focusedContextId: string | null;
  selectedGraphItem: GraphSelection | null;
  sourceContexts: SourceContext[];
  activeEvidence: EvidenceItem | null;
  onSelectContext: (sourceId: string) => void;
}) {
  const visibleContexts = props.sourceContexts;
  const activeContextId =
    props.focusedContextId ||
    resolveEvidenceContextId(props.activeEvidence, visibleContexts) ||
    resolveGraphSelectionContextId(props.selectedGraphItem, visibleContexts);
  const matchedContextIds = resolveMatchedContextIds(
    props.activeEvidence,
    props.selectedGraphItem,
    visibleContexts,
  );
  const contextRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const activeContext =
    visibleContexts.find((context) => context.source_id === activeContextId) || visibleContexts[0] || null;

  useEffect(() => {
    if (!activeContextId) {
      return;
    }
    contextRefs.current[activeContextId]?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  }, [activeContextId]);

  return (
    <div className="evidence-viewer">
      <article className="sidebar-card">
        <div className="section-top compact-top">
          <div>
            <p className="section-kicker">Sources</p>
            <h3>Original Content</h3>
          </div>
          <span>
            {matchedContextIds.length
              ? `${matchedContextIds.length} matched / ${visibleContexts.length}`
              : visibleContexts.length}
          </span>
        </div>
        {visibleContexts.length > 1 ? (
          <div className="context-switcher">
            {visibleContexts.map((context, index) => (
              <button
                key={`switch-${context.source_id}`}
                type="button"
                className={`context-tab ${context.source_id === activeContextId ? "active" : ""} ${
                  matchedContextIds.includes(context.source_id) ? "matched" : ""
                }`}
                onClick={() => props.onSelectContext(context.source_id)}
              >
                <span>{`Source ${index + 1}`}</span>
              </button>
            ))}
          </div>
        ) : null}
        <div className="context-list">
          {activeContext ? (
            <div
              key={activeContext.source_id}
              ref={(node) => {
                contextRefs.current[activeContext.source_id] = node;
              }}
              className={`context-item active ${
                matchedContextIds.includes(activeContext.source_id) ? "matched" : ""
              }`}
            >
              <div className="context-top">
                <div>
                  <strong>{activeContext.title}</strong>
                  <span>{activeContext.content_type}</span>
                </div>
                <div className="context-meta">
                  {matchedContextIds.includes(activeContext.source_id) ? (
                    <small className="context-badge matched">Matched</small>
                  ) : (
                    <small className="context-badge">No direct match</small>
                  )}
                  <small>{activeContext.source_id}</small>
                </div>
              </div>
              <HighlightedClampText
                text={activeContext.content}
                evidence={props.activeEvidence}
                isMatched={matchedContextIds.includes(activeContext.source_id)}
              />
            </div>
          ) : (
            <div className="empty-panel">No source context extracted for this sample.</div>
          )}
        </div>
      </article>

    </div>
  );
}

function AllEvidencePanel(props: {
  sample: SampleDetail;
  activeEvidenceId: string | null;
  sourceContexts: SourceContext[];
  onSelectEvidence: (item: EvidenceItem) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const totalEvidence = props.sample.evidence_items.length;
  const visibleEvidence = expanded ? props.sample.evidence_items : props.sample.evidence_items.slice(0, 3);
  const matchedCount = props.sample.evidence_items.filter((item) =>
    resolveEvidenceContextId(item, props.sourceContexts),
  ).length;

  useEffect(() => {
    setExpanded(false);
  }, [props.sample.sample_id]);

  return (
    <article className="content-card">
      <div className="section-top compact-top">
        <div>
          <p className="section-kicker">Evidence</p>
          <h3>All Evidence</h3>
        </div>
        <span>{expanded ? totalEvidence : `${visibleEvidence.length} / ${totalEvidence}`}</span>
      </div>
      <div className="panel-inline-metrics">
        <span>{matchedCount} linked to source</span>
        <span>{totalEvidence ? `${Math.min(visibleEvidence.length, totalEvidence)} visible` : "0 visible"}</span>
      </div>
      <div className="evidence-list evidence-list-main">
        {totalEvidence ? (
          visibleEvidence.map((item) => {
            const matchedContextId = resolveEvidenceContextId(item, props.sourceContexts);
            return (
              <button
                key={item.id}
                type="button"
                className={`evidence-item ${props.activeEvidenceId === item.id ? "active" : ""}`}
                onClick={() => props.onSelectEvidence(item)}
              >
                <div className="evidence-top">
                  <strong>{item.label}</strong>
                  <span>{item.kind}</span>
                </div>
                <ClampText
                  text={item.evidence_span}
                  className="evidence-text"
                  emptyLabel="No evidence text"
                />
                <div className="evidence-meta">
                  <small>{item.source_id || "No source id"}</small>
                  <small>{matchedContextId ? "Matched in source" : "Not found in source"}</small>
                </div>
              </button>
            );
          })
        ) : (
          <div className="empty-panel">No evidence spans found for this sample.</div>
        )}
      </div>
      {totalEvidence > 3 ? (
        <button type="button" className="ghost-button panel-toggle" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "Show less evidence" : `View all evidence (${totalEvidence})`}
        </button>
      ) : null}
    </article>
  );
}

function HighlightedClampText(props: {
  text: string;
  evidence: EvidenceItem | null;
  isMatched: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const shouldClamp = props.text.length > 220 || props.text.split("\n").length > 3;
  const collapsedText = !expanded && props.isMatched ? buildMatchedExcerpt(props.text, props.evidence) : props.text;

  useEffect(() => {
    setExpanded(false);
  }, [props.text, props.evidence?.id]);

  return (
    <div className="clamp-block">
      <div className={`highlighted-text ${shouldClamp && !expanded ? "clamp-3" : ""}`}>
        {renderHighlightedText(collapsedText, props.evidence)}
      </div>
      {props.evidence && !props.isMatched ? <small className="match-note">Not located in this source block.</small> : null}
      {shouldClamp ? (
        <button type="button" className="text-toggle" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "Show less" : "Show more"}
        </button>
      ) : null}
    </div>
  );
}

export default function App() {
  const [rootPath, setRootPath] = useState(DEFAULT_ROOT_PATH);
  const [runs, setRuns] = useState<ImportedRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [samples, setSamples] = useState<SampleListItem[]>([]);
  const [selectedSample, setSelectedSample] = useState<SampleDetail | null>(null);
  const [selectedGraphItem, setSelectedGraphItem] = useState<GraphSelection | null>(null);
  const [activeEvidenceId, setActiveEvidenceId] = useState<string | null>(null);
  const [focusedContextId, setFocusedContextId] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [hasImageOnly, setHasImageOnly] = useState(false);
  const [hasGraphOnly, setHasGraphOnly] = useState(false);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState("");
  const sampleListRequestRef = useRef(0);
  const sampleDetailRequestRef = useRef(0);

  useEffect(() => {
    void handleScan(DEFAULT_ROOT_PATH);
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
    const requestId = sampleListRequestRef.current + 1;
    sampleListRequestRef.current = requestId;
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
      if (sampleListRequestRef.current !== requestId) {
        return;
      }

      const currentSampleId = selectedSample?.sample_id || "";
      const hasCurrentSample = result.items.some((item) => item.sample_id === currentSampleId);

      startTransition(() => {
        setSamples(result.items);
        setTotal(result.total);
      });

      if (hasCurrentSample) {
        return;
      }

      if (result.items[0]) {
        void handleSampleSelect(result.items[0].sample_id);
      } else {
        setSelectedSample(null);
        setSelectedGraphItem(null);
        setActiveEvidenceId(null);
        setFocusedContextId(null);
      }
    } catch (sampleError) {
      setError(sampleError instanceof Error ? sampleError.message : "Failed to load samples");
    } finally {
      if (sampleListRequestRef.current === requestId) {
        setLoadingSamples(false);
      }
    }
  }

  async function handleSampleSelect(sampleId: string) {
    if (selectedSample?.sample_id === sampleId && !loadingDetail) {
      return;
    }

    const requestId = sampleDetailRequestRef.current + 1;
    sampleDetailRequestRef.current = requestId;
    setLoadingDetail(true);
    setError("");
    try {
      const detail = await fetchSampleDetail(sampleId);
      if (sampleDetailRequestRef.current !== requestId) {
        return;
      }

      startTransition(() => {
        setSelectedSample(detail);
        setSelectedGraphItem(null);
        setActiveEvidenceId(null);
        setFocusedContextId(null);
      });
    } catch (detailError) {
      setError(detailError instanceof Error ? detailError.message : "Failed to load detail");
    } finally {
      if (sampleDetailRequestRef.current === requestId) {
        setLoadingDetail(false);
      }
    }
  }

  async function handleSearchSubmit() {
    if (!selectedRunId) {
      return;
    }
    setPage(1);
    await loadSamples(selectedRunId, 1);
  }

  function handleGraphSelect(selection: GraphSelection | null) {
    setSelectedGraphItem(selection);
    setFocusedContextId(resolvePrimarySourceId(selection?.sourceId));
    setActiveEvidenceId((current) => {
      if (!selection || !selectedSample) {
        return current;
      }
      const matchingEvidence = selectedSample.evidence_items.find(
        (item) => item.graph_item_id === selection.id,
      );
      return matchingEvidence?.id || current;
    });
  }

  function handleEvidenceSelect(item: EvidenceItem) {
    setActiveEvidenceId(item.id);
    setFocusedContextId(resolvePrimarySourceId(item.source_id));
    if (selectedSample && item.graph_item_id) {
      const selection = findGraphSelectionById(selectedSample, item.graph_item_id);
      if (selection) {
        setSelectedGraphItem(selection);
      }
    }
  }

  function handleSourceJump(sourceId: string) {
    setFocusedContextId(sourceId);
  }

  const selectedRun = runs.find((run) => run.run_id === selectedRunId) || null;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const visibleSourceContexts = useMemo(
    () => getVisibleSourceContexts(selectedSample?.source_contexts || []),
    [selectedSample],
  );
  const activeEvidence =
    selectedSample?.evidence_items.find((item) => item.id === activeEvidenceId) || null;
  const selectedSampleSummary = useMemo(() => {
    if (!selectedSample) {
      return null;
    }
    return {
      nodeCount: selectedSample.sub_graph_summary?.node_count || 0,
      edgeCount: selectedSample.sub_graph_summary?.edge_count || 0,
    };
  }, [selectedSample]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">GraphGen Data Platform</p>
          <h1>Evidence-first sample explorer for graph-grounded QA</h1>
          <p className="hero-copy">
            Browse runs, inspect grounded answers, and trace every evidence span back to its original source.
          </p>
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
            placeholder={DEFAULT_ROOT_PATH}
          />
          <button type="submit" disabled={loadingRuns}>
            {loadingRuns ? "Scanning..." : "Import Directory"}
          </button>
        </form>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <main className="workspace">
        <section className="panel rail-panel">
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
              <p>Broken graphs: {selectedRun.stats.invalid_graph_sample_count}</p>
              <p>Broken refs: {selectedRun.stats.invalid_graph_edge_count}</p>
            </div>
          ) : (
            <div className="empty-panel">Import a GraphGen output directory to begin.</div>
          )}
        </section>

        <section className="panel rail-panel">
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
            {loadingSamples && samples.length ? <div className="list-status">Refreshing samples...</div> : null}
            {!loadingSamples && !samples.length ? <div className="empty-panel">No samples found for this page.</div> : null}
            {loadingSamples && !samples.length ? <div className="empty-panel">Loading samples...</div> : null}
            {samples.map((sample) => (
                <button
                  key={sample.sample_id}
                  type="button"
                  className={`sample-card ${
                    selectedSample?.sample_id === sample.sample_id ? "active" : ""
                  }`}
                  onClick={() => void handleSampleSelect(sample.sample_id)}
                >
                  <strong>{sample.question || "Untitled sample"}</strong>
                  <ClampText text={sample.answer_preview || "No answer"} className="sample-preview" />
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

        <section className="panel content-panel">
          <div className="panel-heading">
            <h2>Sample</h2>
            <span>{loadingDetail ? "Loading..." : selectedSample?.run_id || "-"}</span>
          </div>
          {selectedSample ? (
            <div className="content-stack">
              <article className="content-card">
                <div className="section-top">
                  <div>
                    <h3>Question</h3>
                  </div>
                  {selectedSampleSummary ? <span>{selectedSampleSummary.nodeCount} nodes</span> : null}
                </div>
                <ClampText text={selectedSample.question} emptyLabel="No question text" />
              </article>

              <article className="content-card">
                <div className="section-top">
                  <div>
                    <h3>Answer</h3>
                  </div>
                  {selectedSampleSummary ? <span>{selectedSampleSummary.edgeCount} edges</span> : null}
                </div>
                <ClampText text={selectedSample.answer} emptyLabel="No answer text" />
              </article>

              {selectedSample.image_path ? (
                <article className="content-card image-card">
                  <div className="section-top">
                    <div>
                      <p className="section-kicker">Image</p>
                      <h3>Reference Asset</h3>
                    </div>
                    <span>{selectedSample.image_path.split("/").pop()}</span>
                  </div>
                  <img src={buildAssetUrl(selectedSample.image_path)} alt={selectedSample.question} />
                </article>
              ) : null}

              <GraphCanvas
                sample={selectedSample}
                activeGraphItemId={selectedGraphItem?.id || null}
                onSelect={handleGraphSelect}
              />

              <AllEvidencePanel
                sample={selectedSample}
                activeEvidenceId={activeEvidenceId}
                sourceContexts={visibleSourceContexts}
                onSelectEvidence={handleEvidenceSelect}
              />
            </div>
          ) : (
            <div className="empty-panel">Select a sample to inspect its QA pair and graph.</div>
          )}
        </section>

        <aside className="panel sidebar-panel">
          <div className="panel-heading">
            <h2>Inspector</h2>
            <span>{selectedGraphItem ? selectedGraphItem.kind : "Ready"}</span>
          </div>
          {selectedSample ? (
            <div className="sidebar-stack">
              <article className="sidebar-card">
                <div className="section-top compact-top">
                  <div>
                    <p className="section-kicker">Selection</p>
                    <h3>Node / Edge Content</h3>
                  </div>
                </div>
                {selectedGraphItem ? (
                  <div className="selection-grid">
                    <div className="selection-hero">
                      <div>
                        <span>Entity</span>
                        <p>{selectedGraphItem.title}</p>
                      </div>
                      <div>
                        <span>Type</span>
                        <p>{selectedGraphItem.entityType || selectedGraphItem.relationType || "-"}</p>
                      </div>
                    </div>
                    <div>
                      <span>Evidence</span>
                      <ClampText
                        text={selectedGraphItem.evidenceSpan || ""}
                        emptyLabel="-"
                        className="selection-text"
                      />
                    </div>
                    <div>
                      <span>Source</span>
                      {splitSourceIds(selectedGraphItem.sourceId).length ? (
                        <div className="source-chip-list">
                          {splitSourceIds(selectedGraphItem.sourceId).map((sourceId) => (
                            <button
                              key={`${selectedGraphItem.id}-${sourceId}`}
                              type="button"
                              className={`source-chip ${
                                focusedContextId === sourceId ? "active" : ""
                              }`}
                              onClick={() => handleSourceJump(sourceId)}
                            >
                              {sourceId}
                            </button>
                          ))}
                        </div>
                      ) : (
                        <p>-</p>
                      )}
                    </div>
                    <div>
                      <span>Connection</span>
                      <p>{selectedGraphItem.connectedTo || "-"}</p>
                    </div>
                    <div>
                      <span>Description</span>
                      <ClampText
                        text={selectedGraphItem.description || ""}
                        emptyLabel="-"
                        className="selection-text"
                      />
                    </div>
                    <div>
                      <span>Metadata</span>
                      {selectedGraphItem.metadata?.length ? (
                        <details className="metadata-disclosure">
                          <summary>Info</summary>
                          <pre className="metadata-json">
                            {JSON.stringify(metadataToObject(selectedGraphItem.metadata), null, 2)}
                          </pre>
                        </details>
                      ) : (
                        <p>-</p>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="empty-panel">Click a node or edge to inspect its actual content.</div>
                )}
              </article>

              <EvidenceViewer
                focusedContextId={focusedContextId}
                selectedGraphItem={selectedGraphItem}
                sourceContexts={visibleSourceContexts}
                activeEvidence={activeEvidence}
                onSelectContext={handleSourceJump}
              />
            </div>
          ) : (
            <div className="empty-panel">Evidence and node content will appear here.</div>
          )}
        </aside>
      </main>
    </div>
  );
}

function buildGraphElements(
  nodes: GraphNodeRecord[],
  edges: GraphEdgeRecord[],
  sampleImagePath?: string | null,
) {
  const elements: cytoscape.ElementDefinition[] = [];
  const nodeLabels = new Map<string, string>();

  for (const [nodeId, metadata] of nodes) {
    const entityType = String(metadata["entity_type"] || "unknown");
    const fullLabel = buildNodeLabel(nodeId, metadata, sampleImagePath);
    const displayLabel = truncateLabel(fullLabel, entityType === "IMAGE" ? 30 : 34);
    nodeLabels.set(nodeId, fullLabel);

    elements.push({
      data: {
        id: nodeId,
        label: displayLabel,
        title: fullLabel,
        fullLabel,
        entityType,
        description: coerceText(metadata["description"]),
        evidenceSpan: coerceText(metadata["evidence_span"]),
        sourceId: coerceText(metadata["source_id"]),
        metadataEntries: buildMetadataEntries(metadata),
        color: colorForEntityType(entityType),
      },
    });
  }

  edges.forEach(([src, tgt, metadata], index) => {
    const fullLabel = buildEdgeLabel(src, tgt, metadata, nodeLabels);
    elements.push({
      data: {
        id: `${src}-${tgt}-${index}`,
        source: src,
        target: tgt,
        label: truncateLabel(fullLabel, 24),
        title: fullLabel,
        fullLabel,
        relationType: coerceText(
          metadata["relation"] ||
            metadata["relation_type"] ||
            metadata["predicate"] ||
            metadata["edge_type"],
        ),
        description: coerceText(metadata["description"]),
        evidenceSpan: coerceText(metadata["evidence_span"]),
        sourceId: coerceText(metadata["source_id"]),
        connectedTo: `${nodeLabels.get(src) || src} -> ${nodeLabels.get(tgt) || tgt}`,
        metadataEntries: buildMetadataEntries(metadata),
      },
    });
  });

  return elements;
}

function inspectGraphReferences(nodes: GraphNodeRecord[], edges: GraphEdgeRecord[]) {
  const nodeIds = new Set(nodes.map(([nodeId]) => nodeId));
  const validEdges: GraphEdgeRecord[] = [];
  const invalidEdges: Array<{ src: string; tgt: string; missingNodes: string[]; metadata: Record<string, unknown> }> = [];

  edges.forEach((edge) => {
    const [src, tgt, metadata] = edge;
    const missingNodes = [src, tgt].filter((nodeId) => !nodeIds.has(nodeId));
    if (missingNodes.length) {
      invalidEdges.push({ src, tgt, missingNodes, metadata });
      return;
    }
    validEdges.push(edge);
  });

  return { validEdges, invalidEdges };
}

function toGraphSelection(kind: "node" | "edge", data: Record<string, unknown>, id: string): GraphSelection {
  return {
    kind,
    id,
    label: String(data.fullLabel || data.label || ""),
    title: String(data.title || data.fullLabel || data.label || ""),
    entityType: data.entityType as string | undefined,
    relationType: data.relationType as string | undefined,
    description: data.description as string | undefined,
    evidenceSpan: data.evidenceSpan as string | undefined,
    sourceId: data.sourceId as string | undefined,
    connectedTo: data.connectedTo as string | undefined,
    metadata: (data.metadataEntries as Array<{ key: string; value: string }> | undefined) || [],
  };
}

function findGraphSelectionById(sample: SampleDetail, graphItemId: string): GraphSelection | null {
  if (!sample.sub_graph) {
    return null;
  }

  for (const [nodeId, metadata] of sample.sub_graph.nodes || []) {
    if (nodeId === graphItemId) {
      const title = buildNodeLabel(nodeId, metadata, sample.image_path);
      return {
        kind: "node",
        id: nodeId,
        label: title,
        title,
        entityType: coerceText(metadata["entity_type"]),
        description: coerceText(metadata["description"]),
        evidenceSpan: coerceText(metadata["evidence_span"]),
        sourceId: coerceText(metadata["source_id"]),
        metadata: buildMetadataEntries(metadata),
      };
    }
  }

  for (let index = 0; index < (sample.sub_graph.edges || []).length; index += 1) {
    const edge = (sample.sub_graph.edges || [])[index];
    if (!edge) {
      continue;
    }
    const [src, tgt, metadata] = edge;
    const edgeId = `${src}-${tgt}-${index}`;
    if (edgeId === graphItemId) {
      const nodeLabels = new Map(
        (sample.sub_graph.nodes || []).map(([nodeId, nodeMetadata]) => [
          nodeId,
          buildNodeLabel(nodeId, nodeMetadata, sample.image_path),
        ]),
      );
      return {
        kind: "edge",
        id: edgeId,
        label: buildEdgeLabel(src, tgt, metadata, nodeLabels),
        title: buildEdgeLabel(src, tgt, metadata, nodeLabels),
        relationType: coerceText(
          metadata["relation"] ||
            metadata["relation_type"] ||
            metadata["predicate"] ||
            metadata["edge_type"],
        ),
        description: coerceText(metadata["description"]),
        evidenceSpan: coerceText(metadata["evidence_span"]),
        sourceId: coerceText(metadata["source_id"]),
        connectedTo: `${nodeLabels.get(src) || src} -> ${nodeLabels.get(tgt) || tgt}`,
        metadata: buildMetadataEntries(metadata),
      };
    }
  }

  return null;
}

function colorForEntityType(entityType: string) {
  const palette = [
    "#dbe8f7",
    "#e7edf6",
    "#dfece7",
    "#f3e7dc",
    "#e8e0f2",
    "#f1e0e4",
    "#dfeef1",
  ];

  let hash = 0;
  for (let index = 0; index < entityType.length; index += 1) {
    hash = entityType.charCodeAt(index) + ((hash << 5) - hash);
  }
  return palette[Math.abs(hash) % palette.length];
}

function buildNodeLabel(
  nodeId: string,
  metadata: Record<string, unknown>,
  sampleImagePath?: string | null,
) {
  const entityType = coerceText(metadata["entity_type"]).toUpperCase();
  const imageName = extractImageName(metadata, nodeId, sampleImagePath);
  if (entityType === "IMAGE" && imageName) {
    return imageName;
  }

  return (
    coerceText(metadata["entity_name"]) ||
    imageName ||
    coerceText(metadata["name"]) ||
    coerceText(metadata["label"]) ||
    nodeId
  );
}

function buildEdgeLabel(
  src: string,
  tgt: string,
  metadata: Record<string, unknown>,
  nodeLabels: Map<string, string>,
) {
  const relation =
    coerceText(metadata["relation"]) ||
    coerceText(metadata["relation_type"]) ||
    coerceText(metadata["predicate"]) ||
    coerceText(metadata["edge_type"]) ||
    coerceText(metadata["label"]);

  if (relation) {
    return relation;
  }

  const srcLabel = nodeLabels.get(src) || src;
  const tgtLabel = nodeLabels.get(tgt) || tgt;
  return `${truncateLabel(srcLabel, 18)} -> ${truncateLabel(tgtLabel, 18)}`;
}

function extractImageName(
  metadata: Record<string, unknown>,
  fallback: string,
  sampleImagePath?: string | null,
) {
  const candidates = [
    metadata["image_name"],
    metadata["image_file"],
    metadata["image_path"],
    metadata["asset_path"],
    metadata["uri"],
    metadata["source_id"],
    metadata["entity_name"],
    metadata["description"],
    sampleImagePath,
    fallback,
  ];

  for (const candidate of candidates) {
    const text = coerceText(candidate);
    if (!text) {
      continue;
    }

    const match = text.match(/([^/\\]+(?:\.(?:png|jpe?g|webp|bmp|gif|svg)))$/i);
    if (match) {
      return match[1];
    }
  }

  return "";
}

function buildMetadataEntries(metadata: Record<string, unknown>) {
  return Object.entries(metadata)
    .map(([key, value]) => ({
      key,
      value:
        typeof value === "string"
          ? value.trim()
          : value === undefined || value === null
            ? ""
            : JSON.stringify(value),
    }))
    .filter((entry) => entry.value && entry.value !== '""')
    .slice(0, 8);
}

function coerceText(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function truncateLabel(label: string, maxLength: number) {
  const normalized = label.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxLength - 1))}…`;
}

function resolveEvidenceContextId(evidence: EvidenceItem | null, contexts: SourceContext[]) {
  if (!evidence) {
    return null;
  }

  const preferred = contexts.find((context) => context.source_id === evidence.source_id);
  if (preferred && hasHighlightedMatch(preferred.content, evidence)) {
    return preferred.source_id;
  }

  const fallback = contexts.find((context) => hasHighlightedMatch(context.content, evidence));
  return fallback?.source_id || evidence.source_id || null;
}

function resolveGraphSelectionContextId(
  selection: GraphSelection | null,
  contexts: SourceContext[],
) {
  const sourceIds = splitSourceIds(selection?.sourceId);
  for (const sourceId of sourceIds) {
    const match = contexts.find((context) => context.source_id === sourceId);
    if (match) {
      return match.source_id;
    }
  }
  return null;
}

function resolveMatchedContextIds(
  evidence: EvidenceItem | null,
  selection: GraphSelection | null,
  contexts: SourceContext[],
) {
  const matchedIds: string[] = [];
  const addId = (sourceId: string | null | undefined) => {
    for (const item of splitSourceIds(sourceId)) {
      if (contexts.some((context) => context.source_id === item) && !matchedIds.includes(item)) {
        matchedIds.push(item);
      }
    }
  };

  addId(evidence?.source_id);
  addId(selection?.sourceId);

  if (evidence) {
    for (const context of contexts) {
      if (hasHighlightedMatch(context.content, evidence) && !matchedIds.includes(context.source_id)) {
        matchedIds.push(context.source_id);
      }
    }
  }

  return matchedIds;
}

function getVisibleSourceContexts(contexts: SourceContext[]) {
  const filtered = contexts.filter((context) => !/^context-\d+$/.test(context.source_id));
  return filtered.length ? filtered : contexts;
}

function metadataToObject(entries: Array<{ key: string; value: string }>) {
  return Object.fromEntries(entries.map((entry) => [entry.key, entry.value]));
}

function splitSourceIds(sourceId: string | null | undefined) {
  return String(sourceId || "")
    .split("<SEP>")
    .map((item) => item.trim())
    .filter(Boolean);
}

function resolvePrimarySourceId(sourceId: string | null | undefined) {
  return splitSourceIds(sourceId)[0] || null;
}

function splitEvidenceParts(evidenceSpan: string) {
  return evidenceSpan
    .split("<SEP>")
    .map((part) => part.trim())
    .filter(Boolean);
}

function hasHighlightedMatch(content: string, evidence: EvidenceItem | null) {
  if (!evidence) {
    return false;
  }
  const normalizedContent = content.toLowerCase();
  return splitEvidenceParts(evidence.evidence_span).some((part) =>
    normalizedContent.includes(part.toLowerCase()),
  );
}

function renderHighlightedText(text: string, evidence: EvidenceItem | null) {
  const parts = evidence ? splitEvidenceParts(evidence.evidence_span) : [];
  if (!parts.length) {
    return text;
  }

  let cursor = 0;
  const nodes: Array<string | ReactNode> = [];
  const lowerText = text.toLowerCase();

  while (cursor < text.length) {
    let nextMatch: { index: number; length: number } | null = null;
    for (const part of parts) {
      const index = lowerText.indexOf(part.toLowerCase(), cursor);
      if (index === -1) {
        continue;
      }
      if (!nextMatch || index < nextMatch.index) {
        nextMatch = { index, length: part.length };
      }
    }

    if (!nextMatch) {
      nodes.push(text.slice(cursor));
      break;
    }

    if (nextMatch.index > cursor) {
      nodes.push(text.slice(cursor, nextMatch.index));
    }

    const matchText = text.slice(nextMatch.index, nextMatch.index + nextMatch.length);
    nodes.push(
      <mark key={`${nextMatch.index}-${matchText}`} className="evidence-highlight">
        {matchText}
      </mark>,
    );
    cursor = nextMatch.index + nextMatch.length;
  }

  return nodes;
}

function buildMatchedExcerpt(text: string, evidence: EvidenceItem | null) {
  if (!evidence) {
    return text;
  }

  const normalizedText = text.toLowerCase();
  let firstMatchIndex = -1;
  let firstMatchLength = 0;

  for (const part of splitEvidenceParts(evidence.evidence_span)) {
    const index = normalizedText.indexOf(part.toLowerCase());
    if (index === -1) {
      continue;
    }
    if (firstMatchIndex === -1 || index < firstMatchIndex) {
      firstMatchIndex = index;
      firstMatchLength = part.length;
    }
  }

  if (firstMatchIndex === -1) {
    return text;
  }

  const leadingWindow = 120;
  const trailingWindow = 180;
  const start = Math.max(0, firstMatchIndex - leadingWindow);
  const end = Math.min(text.length, firstMatchIndex + firstMatchLength + trailingWindow);
  const prefix = start > 0 ? "..." : "";
  const suffix = end < text.length ? "..." : "";

  return `${prefix}${text.slice(start, end).trim()}${suffix}`;
}

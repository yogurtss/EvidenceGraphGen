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
  VisualizationEvent,
  VisualizationTrace,
} from "./types";

const PAGE_SIZE = 5;
const DEFAULT_ROOT_PATH = "/home/lukashe/data/projects/EvidenceGraphGen/cache";
const MAX_VISIBLE_SOURCE_IDS = 6;
const TIMELINE_ANIMATION_MS = 420;

type TimelineEdgeEntry = {
  src: string;
  tgt: string;
  candidate?: Record<string, unknown>;
};

function GraphCanvas(props: {
  sample: SampleDetail | null;
  activeGraphItemId: string | null;
  onSelect: (selection: GraphSelection | null) => void;
  viewSwitcher?: ReactNode;
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
        <div className="graph-actions">
          {props.viewSwitcher}
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

function TimelineGraphCanvas(props: {
  sample: SampleDetail;
  trace: VisualizationTrace;
  activeGraphItemId: string | null;
  onSelect: (selection: GraphSelection | null) => void;
  viewSwitcher?: ReactNode;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<cytoscape.Core | null>(null);
  const [familyFilter, setFamilyFilter] = useState("all");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);

  const events = props.trace.events || [];
  const families = useMemo(
    () => Array.from(new Set(events.map((event) => event.qa_family).filter(Boolean))),
    [events],
  );
  const filteredEvents = useMemo(
    () =>
      familyFilter === "all"
        ? events
        : events.filter((event) => event.qa_family === familyFilter),
    [events, familyFilter],
  );
  const activeEvent = filteredEvents[Math.min(currentIndex, filteredEvents.length - 1)] || null;

  useEffect(() => {
    setCurrentIndex(0);
    setPlaying(false);
  }, [props.sample.sample_id, familyFilter, events.length]);

  useEffect(() => {
    if (!playing || filteredEvents.length <= 1) {
      return;
    }
    const timer = window.setInterval(() => {
      setCurrentIndex((index) => (index + 1) % filteredEvents.length);
    }, 1200);
    return () => window.clearInterval(timer);
  }, [playing, filteredEvents.length]);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }
    graphRef.current?.destroy();
    setGraphError(null);

    if (!filteredEvents.length) {
      return;
    }

    try {
      const elements = buildTimelineGraphElements(
        props.trace,
        filteredEvents,
        props.sample.image_path,
        containerRef.current.clientWidth,
        containerRef.current.clientHeight,
      );
      const cy = cytoscape({
        container: containerRef.current,
        elements,
        layout: { name: "preset", fit: false },
        style: [
          {
            selector: "node",
            style: {
              label: "data(label)",
              "background-color": "data(color)",
              shape: "ellipse",
              color: "#24364c",
              "text-wrap": "wrap",
              "text-max-width": "66px",
              "font-size": 8,
              "font-weight": 600,
              "text-valign": "center",
              "text-halign": "center",
              width: 50,
              height: 50,
              "border-width": 1.5,
              "border-color": "#f8fbff",
              "transition-property": "opacity, background-color, border-color, border-width",
              "transition-duration": TIMELINE_ANIMATION_MS,
              "transition-timing-function": "ease-in-out",
              opacity: 1,
            },
          },
          {
            selector: "node.timeline-hidden",
            style: {
              opacity: 0.04,
              "border-width": 0,
            },
          },
          {
            selector: "node.timeline-anchor",
            style: {
              width: 62,
              height: 62,
              "border-width": 3,
              "border-color": "#f8fbff",
            },
          },
          {
            selector: "node.candidate",
            style: {
              opacity: 0.42,
              "border-style": "dashed",
              "border-color": "#9ac7ec",
              "background-color": "#dfeef1",
            },
          },
          {
            selector: ".chosen",
            style: {
              opacity: 1,
              "border-color": "#ffcf7a",
              "border-width": 4,
              "z-index": 999,
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
              "transition-property": "opacity, width, line-color, target-arrow-color",
              "transition-duration": TIMELINE_ANIMATION_MS,
              "transition-timing-function": "ease-in-out",
              opacity: 0.76,
            },
          },
          {
            selector: "edge.timeline-hidden",
            style: {
              opacity: 0.02,
              width: 0.4,
            },
          },
          {
            selector: "edge.candidate",
            style: {
              opacity: 0.36,
              width: 1.4,
              "line-style": "dashed",
              "line-color": "#9ab8d8",
              "target-arrow-color": "#9ab8d8",
            },
          },
          {
            selector: "edge.chosen",
            style: {
              opacity: 1,
              width: 3,
              "line-color": "#ffcf7a",
              "target-arrow-color": "#ffcf7a",
              color: "#ffe4aa",
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
        if (event.target.hasClass("timeline-hidden")) {
          return;
        }
        props.onSelect(toGraphSelection("node", event.target.data(), event.target.id()));
      });
      cy.on("tap", "edge", (event) => {
        if (event.target.hasClass("timeline-hidden")) {
          return;
        }
        props.onSelect(toGraphSelection("edge", event.target.data(), event.target.id()));
      });

      const anchorNode = cy.nodes(".timeline-anchor");
      anchorNode.lock();
      centerTimelineAnchor(cy);
      graphRef.current = cy;
    } catch (error) {
      setGraphError(error instanceof Error ? error.message : "Unable to render this timeline.");
    }

    return () => {
      graphRef.current?.destroy();
      graphRef.current = null;
    };
  }, [props.trace, props.sample.image_path, props.sample.sample_id, filteredEvents]);

  useEffect(() => {
    const cy = graphRef.current;
    if (!cy || !activeEvent) {
      return;
    }
    applyTimelineEventState(cy, activeEvent);
    centerTimelineAnchor(cy);
  }, [activeEvent]);

  useEffect(() => {
    const cy = graphRef.current;
    if (!cy) {
      return;
    }
    cy.elements().unselect();
    if (!props.activeGraphItemId) {
      return;
    }
    const target = cy.getElementById(props.activeGraphItemId);
    if (target.nonempty()) {
      target.select();
    }
  }, [props.activeGraphItemId, activeEvent?.event_id]);

  const canStep = filteredEvents.length > 1;

  return (
    <article className="content-card graph-card">
      <div className="section-top">
        <div>
          <p className="section-kicker">Timeline</p>
          <h3>Build Timeline</h3>
        </div>
        <div className="graph-actions">
          {props.viewSwitcher}
          <button
            type="button"
            className="ghost-button"
            onClick={() => {
              if (graphRef.current) {
                centerTimelineAnchor(graphRef.current);
              }
            }}
          >
            Center
          </button>
        </div>
      </div>

      <div className="timeline-toolbar">
        <div className="timeline-buttons">
          <button
            type="button"
            className="ghost-button"
            disabled={!canStep}
            onClick={() => setCurrentIndex((index) => Math.max(0, index - 1))}
          >
            Prev
          </button>
          <button
            type="button"
            className="ghost-button"
            disabled={!canStep}
            onClick={() =>
              setCurrentIndex((index) => Math.min(filteredEvents.length - 1, index + 1))
            }
          >
            Next
          </button>
          <button
            type="button"
            className="ghost-button"
            disabled={!canStep}
            onClick={() => setPlaying((value) => !value)}
          >
            {playing ? "Pause" : "Play"}
          </button>
        </div>
        <div className="timeline-family-filter">
          {["all", ...families].map((family) => (
            <button
              key={family}
              type="button"
              className={`timeline-chip ${familyFilter === family ? "active" : ""}`}
              onClick={() => setFamilyFilter(family)}
            >
              {family === "all" ? "All" : family}
            </button>
          ))}
        </div>
      </div>

      {activeEvent ? (
        <>
          <div className="timeline-event-summary">
            <div>
              <span>Event</span>
              <strong>{activeEvent.event_type}</strong>
            </div>
            <div>
              <span>Family</span>
              <strong>{activeEvent.qa_family || "-"}</strong>
            </div>
            <div>
              <span>Status</span>
              <strong>{activeEvent.status || "-"}</strong>
            </div>
            <div>
              <span>Step</span>
              <strong>{`${currentIndex + 1} / ${filteredEvents.length}`}</strong>
            </div>
          </div>
          {activeEvent.event_type === "rollback" ? (
            <div className="timeline-rollback-badge">Rollback applied</div>
          ) : null}
          <div className="graph-canvas timeline-canvas" ref={containerRef} />
          {graphError ? <div className="empty-panel">{`Timeline render issue: ${graphError}`}</div> : null}
          <div className="timeline-reason">
            <ClampText text={activeEvent.reason || ""} emptyLabel="No event reason" />
            {activeEvent.termination_reason ? (
              <small>{`Termination: ${activeEvent.termination_reason}`}</small>
            ) : null}
          </div>
          {activeEvent.event_type === "qa_generated" ? (
            <div className="timeline-qa-card">
              <span>{activeEvent.generator_key || "generator"}</span>
              <ClampText text={activeEvent.question || ""} emptyLabel="No question" />
              <ClampText text={activeEvent.answer || ""} emptyLabel="No answer" />
            </div>
          ) : null}
        </>
      ) : (
        <div className="empty-panel">No timeline events for this sample.</div>
      )}
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
  const [showAllContextTabs, setShowAllContextTabs] = useState(false);
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
  const CONTEXT_TAB_VISIBLE_COUNT = 4;
  const displayedContextTabs = showAllContextTabs
    ? visibleContexts
    : visibleContexts.slice(0, CONTEXT_TAB_VISIBLE_COUNT);
  const hiddenContextTabCount = Math.max(visibleContexts.length - displayedContextTabs.length, 0);

  useEffect(() => {
    setShowAllContextTabs(false);
  }, [props.selectedGraphItem?.id, props.activeEvidence?.id, visibleContexts.length]);

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
            {displayedContextTabs.map((context, index) => (
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
            {hiddenContextTabCount > 0 ? (
              <button
                type="button"
                className="context-tab"
                onClick={() => setShowAllContextTabs((value) => !value)}
              >
                <span>{showAllContextTabs ? "Show hidden" : `+${hiddenContextTabCount} hidden`}</span>
              </button>
            ) : null}
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
  const chunkPreview = useMemo(() => buildChunkPreview(props.text), [props.text]);
  const shouldClamp =
    props.text.length > 220 ||
    props.text.split("\n").length > 3 ||
    chunkPreview.truncated;
  const collapsedText = useMemo(() => {
    if (expanded) {
      return props.text;
    }
    if (props.isMatched) {
      return buildMatchedExcerpt(props.text, props.evidence);
    }
    return chunkPreview.text;
  }, [expanded, props.evidence, props.isMatched, props.text, chunkPreview.text]);

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
  const [showAllSourceIds, setShowAllSourceIds] = useState(false);
  const [graphViewMode, setGraphViewMode] = useState<"final" | "timeline">("final");
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
        setGraphViewMode("final");
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
        setShowAllSourceIds(false);
        setGraphViewMode("final");
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
    setShowAllSourceIds(false);
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
    setShowAllSourceIds(false);
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
  const selectedSourceIds = useMemo(
    () => splitSourceIds(selectedGraphItem?.sourceId),
    [selectedGraphItem?.sourceId],
  );
  const hiddenSourceCount = Math.max(0, selectedSourceIds.length - MAX_VISIBLE_SOURCE_IDS);
  const displayedSourceIds =
    showAllSourceIds || hiddenSourceCount === 0
      ? selectedSourceIds
      : selectedSourceIds.slice(0, MAX_VISIBLE_SOURCE_IDS);
  const hasTimeline = Boolean(selectedSample?.visualization_trace?.events?.length);
  const activeGraphViewMode = hasTimeline ? graphViewMode : "final";
  const graphViewSwitcher = hasTimeline ? (
    <div className="graph-view-toggle">
      <button
        type="button"
        className={activeGraphViewMode === "final" ? "active" : ""}
        onClick={() => setGraphViewMode("final")}
      >
        Final Graph
      </button>
      <button
        type="button"
        className={activeGraphViewMode === "timeline" ? "active" : ""}
        onClick={() => setGraphViewMode("timeline")}
      >
        Build Timeline
      </button>
    </div>
  ) : null;

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

              {activeGraphViewMode === "timeline" && selectedSample.visualization_trace ? (
                <TimelineGraphCanvas
                  sample={selectedSample}
                  trace={selectedSample.visualization_trace}
                  activeGraphItemId={selectedGraphItem?.id || null}
                  onSelect={handleGraphSelect}
                  viewSwitcher={graphViewSwitcher}
                />
              ) : (
                <GraphCanvas
                  sample={selectedSample}
                  activeGraphItemId={selectedGraphItem?.id || null}
                  onSelect={handleGraphSelect}
                  viewSwitcher={graphViewSwitcher}
                />
              )}

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
                      {selectedSourceIds.length ? (
                        <div className="source-chip-list">
                          {displayedSourceIds.map((sourceId) => (
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
                          {hiddenSourceCount > 0 ? (
                            <button
                              type="button"
                              className="source-chip source-chip-toggle"
                              onClick={() => setShowAllSourceIds((value) => !value)}
                            >
                              {showAllSourceIds ? "Show hidden" : `+${hiddenSourceCount} hidden`}
                            </button>
                          ) : null}
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

function buildTimelineGraphElements(
  trace: VisualizationTrace,
  events: VisualizationEvent[],
  sampleImagePath?: string | null,
  canvasWidth = 760,
  canvasHeight = 336,
) {
  const elements: cytoscape.ElementDefinition[] = [];
  const catalogNodes = trace.graph_catalog?.nodes || {};
  const catalogEdges = trace.graph_catalog?.edges || {};
  const nodeIds = new Set<string>();
  const edgeEntries = new Map<string, TimelineEdgeEntry>();
  const candidateByNodeId = new Map<string, Record<string, unknown>>();
  const candidateUidsByNodeId = new Map<string, Set<string>>();
  const candidateUidsByEdgeKey = new Map<string, Set<string>>();

  events.forEach((event) => {
    (event.selected_node_ids || []).forEach((nodeId) => {
      nodeIds.add(String(nodeId));
    });
    (event.selected_edge_pairs || []).forEach((pair) => {
      const edgePair = normalizeTimelineEdgePair(pair);
      if (!edgePair) {
        return;
      }
      const [src, tgt] = edgePair;
      nodeIds.add(src);
      nodeIds.add(tgt);
      edgeEntries.set(pairKey(src, tgt), { src, tgt });
    });
    (event.candidate_pool || []).forEach((candidate) => {
      const candidateNodeId = coerceText(candidate.candidate_node_id);
      if (candidateNodeId) {
        nodeIds.add(candidateNodeId);
        candidateByNodeId.set(candidateNodeId, candidate);
        const candidateUid = coerceText(candidate.candidate_uid);
        if (candidateUid) {
          const existingUids = candidateUidsByNodeId.get(candidateNodeId) || new Set<string>();
          existingUids.add(candidateUid);
          candidateUidsByNodeId.set(candidateNodeId, existingUids);
        }
      }
      const edgePair = normalizeTimelineEdgePair(candidate.bound_edge_pair);
      if (!edgePair) {
        return;
      }
      const [src, tgt] = edgePair;
      nodeIds.add(src);
      nodeIds.add(tgt);
      const key = pairKey(src, tgt);
      edgeEntries.set(key, { src, tgt, candidate });
      const candidateUid = coerceText(candidate.candidate_uid);
      if (candidateUid) {
        const existingUids = candidateUidsByEdgeKey.get(key) || new Set<string>();
        existingUids.add(candidateUid);
        candidateUidsByEdgeKey.set(key, existingUids);
      }
    });
  });

  if (!nodeIds.size && trace.seed_node_id) {
    nodeIds.add(trace.seed_node_id);
  }

  const anchorNodeId = getTimelineAnchorNodeId(trace, catalogNodes, nodeIds);
  const positions = buildTimelineNodePositions(
    Array.from(nodeIds),
    Array.from(edgeEntries.values()),
    anchorNodeId,
    canvasWidth,
    canvasHeight,
  );

  for (const nodeId of nodeIds) {
    const metadata = normalizeTimelineNodeMetadata(
      nodeId,
      catalogNodes[nodeId],
      candidateByNodeId.get(nodeId),
    );
    const entityType = coerceText(metadata["entity_type"]) || "unknown";
    const fullLabel = buildNodeLabel(nodeId, metadata, sampleImagePath);
    const classes = [
      "timeline-hidden",
      nodeId === anchorNodeId ? "timeline-anchor" : "",
    ]
      .filter(Boolean)
      .join(" ");
    elements.push({
      data: {
        id: nodeId,
        label: truncateLabel(fullLabel, entityType === "IMAGE" ? 30 : 34),
        title: fullLabel,
        fullLabel,
        entityType,
        description: coerceText(metadata["description"]),
        evidenceSpan: coerceText(metadata["evidence_span"]),
        sourceId: coerceText(metadata["source_id"]),
        metadataEntries: buildMetadataEntries(metadata),
        color: colorForEntityType(entityType),
        candidateUids: Array.from(candidateUidsByNodeId.get(nodeId) || []),
      },
      position: positions.get(nodeId),
      classes,
    });
  }

  Array.from(edgeEntries.values()).forEach((entry, index) => {
    const key = pairKey(entry.src, entry.tgt);
    const edgeId = `timeline-edge-${index}`;
    const metadata = normalizeTimelineEdgeMetadata(
      entry.src,
      entry.tgt,
      catalogEdges[key] || catalogEdges[`${entry.tgt}->${entry.src}`],
      entry.candidate,
    );
    const fullLabel = buildEdgeLabel(entry.src, entry.tgt, metadata, new Map());
    elements.push({
      data: {
        id: edgeId,
        source: entry.src,
        target: entry.tgt,
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
        connectedTo: `${entry.src} -> ${entry.tgt}`,
        metadataEntries: buildMetadataEntries(metadata),
        pairKey: key,
        candidateUids: Array.from(candidateUidsByEdgeKey.get(key) || []),
      },
      classes: "timeline-hidden",
    });
  });

  return elements;
}

function normalizeTimelineNodeMetadata(
  nodeId: string,
  catalogPayload: Record<string, unknown> | undefined,
  candidate?: Record<string, unknown>,
): Record<string, unknown> {
  return {
    entity_name: nodeId,
    ...(catalogPayload || {}),
    entity_type:
      catalogPayload?.["entity_type"] || candidate?.["entity_type"] || "unknown",
    description:
      catalogPayload?.["description"] ||
      candidate?.["evidence_summary"] ||
      candidate?.["reason"] ||
      "",
  };
}

function applyTimelineEventState(cy: cytoscape.Core, event: VisualizationEvent) {
  const selectedNodeIds = new Set((event.selected_node_ids || []).map(String));
  const selectedEdgeKeys = new Set<string>();
  const candidateNodeIds = new Set<string>();
  const candidateEdgeKeys = new Set<string>();
  const chosenCandidate = isRecord(event.chosen_candidate) ? event.chosen_candidate : {};
  const chosenNodeId = coerceText(chosenCandidate["candidate_node_id"]);
  const chosenCandidateUid = coerceText(chosenCandidate["candidate_uid"]);
  const chosenEdgePair = normalizeTimelineEdgePair(chosenCandidate["bound_edge_pair"]);
  const chosenEdgeKey = chosenEdgePair ? pairKey(chosenEdgePair[0], chosenEdgePair[1]) : "";

  (event.selected_edge_pairs || []).forEach((pair) => {
    const edgePair = normalizeTimelineEdgePair(pair);
    if (edgePair) {
      selectedEdgeKeys.add(pairKey(edgePair[0], edgePair[1]));
    }
  });

  (event.candidate_pool || []).forEach((candidate) => {
    const candidateNodeId = coerceText(candidate.candidate_node_id);
    if (candidateNodeId) {
      candidateNodeIds.add(candidateNodeId);
    }
    const edgePair = normalizeTimelineEdgePair(candidate.bound_edge_pair);
    if (edgePair) {
      candidateEdgeKeys.add(pairKey(edgePair[0], edgePair[1]));
    }
  });

  cy.batch(() => {
    cy.elements()
      .removeClass("timeline-hidden selected-node selected-edge candidate chosen")
      .addClass("timeline-hidden");

    cy.nodes().forEach((node) => {
      const nodeId = node.id();
      const candidateUids = timelineCandidateUids(node.data("candidateUids"));
      const isSelected = selectedNodeIds.has(nodeId);
      const isCandidate = !isSelected && candidateNodeIds.has(nodeId);
      const isChosen =
        nodeId === chosenNodeId ||
        (chosenCandidateUid ? candidateUids.has(chosenCandidateUid) : false);

      if (isSelected || isCandidate) {
        node.removeClass("timeline-hidden");
      }
      if (isSelected) {
        node.addClass("selected-node");
      }
      if (isCandidate) {
        node.addClass("candidate");
      }
      if (isChosen) {
        node.addClass("chosen");
      }
    });

    cy.edges().forEach((edge) => {
      const edgeKey = coerceText(edge.data("pairKey"));
      const candidateUids = timelineCandidateUids(edge.data("candidateUids"));
      const isSelected = selectedEdgeKeys.has(edgeKey);
      const isCandidate = !isSelected && candidateEdgeKeys.has(edgeKey);
      const isChosen =
        edgeKey === chosenEdgeKey ||
        (chosenCandidateUid ? candidateUids.has(chosenCandidateUid) : false);

      if (isSelected || isCandidate) {
        edge.removeClass("timeline-hidden");
      }
      if (isSelected) {
        edge.addClass("selected-edge");
      }
      if (isCandidate) {
        edge.addClass("candidate");
      }
      if (isChosen) {
        edge.addClass("chosen");
      }
    });
  });
}

function centerTimelineAnchor(cy: cytoscape.Core) {
  const anchor = cy.nodes(".timeline-anchor").first();
  if (anchor.empty()) {
    return;
  }
  const renderedPosition = anchor.renderedPosition();
  cy.panBy({
    x: cy.width() / 2 - renderedPosition.x,
    y: cy.height() / 2 - renderedPosition.y,
  });
}

function getTimelineAnchorNodeId(
  trace: VisualizationTrace,
  catalogNodes: Record<string, Record<string, unknown>>,
  nodeIds: Set<string>,
) {
  if (
    trace.seed_node_id &&
    nodeIds.has(trace.seed_node_id) &&
    coerceText(catalogNodes[trace.seed_node_id]?.["entity_type"]).toUpperCase() === "IMAGE"
  ) {
    return trace.seed_node_id;
  }

  if (trace.seed_node_id) {
    const virtualImageNodeId = `${trace.seed_node_id}::virtual_image`;
    if (nodeIds.has(virtualImageNodeId)) {
      return virtualImageNodeId;
    }
  }

  for (const nodeId of nodeIds) {
    if (coerceText(catalogNodes[nodeId]?.["entity_type"]).toUpperCase() === "IMAGE") {
      return nodeId;
    }
  }

  if (trace.seed_node_id && nodeIds.has(trace.seed_node_id)) {
    return trace.seed_node_id;
  }

  return Array.from(nodeIds)[0] || trace.seed_node_id || "";
}

function buildTimelineNodePositions(
  nodeIds: string[],
  edgeEntries: TimelineEdgeEntry[],
  anchorNodeId: string,
  canvasWidth: number,
  canvasHeight: number,
) {
  const positions = new Map<string, { x: number; y: number }>();
  const center = { x: Math.max(canvasWidth, 360) / 2, y: Math.max(canvasHeight, 280) / 2 };
  if (!nodeIds.length) {
    return positions;
  }

  const orderedNodeIds = [...nodeIds].sort((left, right) => {
    if (left === anchorNodeId) {
      return -1;
    }
    if (right === anchorNodeId) {
      return 1;
    }
    return left.localeCompare(right);
  });
  const adjacency = new Map<string, Set<string>>();
  orderedNodeIds.forEach((nodeId) => adjacency.set(nodeId, new Set<string>()));
  edgeEntries.forEach((entry) => {
    adjacency.get(entry.src)?.add(entry.tgt);
    adjacency.get(entry.tgt)?.add(entry.src);
  });

  const distances = new Map<string, number>();
  const queue = anchorNodeId ? [anchorNodeId] : [orderedNodeIds[0]];
  if (queue[0]) {
    distances.set(queue[0], 0);
  }

  for (let queueIndex = 0; queueIndex < queue.length; queueIndex += 1) {
    const nodeId = queue[queueIndex];
    const distance = distances.get(nodeId) || 0;
    const neighbors = Array.from(adjacency.get(nodeId) || []).sort();
    neighbors.forEach((neighborId) => {
      if (distances.has(neighborId)) {
        return;
      }
      distances.set(neighborId, distance + 1);
      queue.push(neighborId);
    });
  }

  orderedNodeIds.forEach((nodeId) => {
    if (!distances.has(nodeId)) {
      distances.set(nodeId, 2);
    }
  });

  const rings = new Map<number, string[]>();
  orderedNodeIds.forEach((nodeId) => {
    const distance = Math.min(distances.get(nodeId) || 0, 3);
    const ringNodes = rings.get(distance) || [];
    ringNodes.push(nodeId);
    rings.set(distance, ringNodes);
  });

  const ringGap = Math.max(82, Math.min(canvasWidth, canvasHeight) / 3.25);
  rings.forEach((ringNodes, distance) => {
    if (distance === 0) {
      ringNodes.forEach((nodeId) => positions.set(nodeId, center));
      return;
    }

    const radius = ringGap * distance;
    const angleOffset = -Math.PI / 2 + distance * 0.36;
    ringNodes.forEach((nodeId, index) => {
      const angle = angleOffset + (2 * Math.PI * index) / Math.max(ringNodes.length, 1);
      positions.set(nodeId, {
        x: center.x + Math.cos(angle) * radius,
        y: center.y + Math.sin(angle) * radius,
      });
    });
  });

  return positions;
}

function normalizeTimelineEdgePair(value: unknown): [string, string] | null {
  if (!Array.isArray(value) || value.length < 2) {
    return null;
  }
  return [String(value[0]), String(value[1])];
}

function pairKey(src: string, tgt: string) {
  return `${src}->${tgt}`;
}

function timelineCandidateUids(value: unknown) {
  if (!Array.isArray(value)) {
    return new Set<string>();
  }
  return new Set(value.map(String).filter(Boolean));
}

function normalizeTimelineEdgeMetadata(
  src: string,
  tgt: string,
  catalogPayload: Record<string, unknown> | undefined,
  candidate?: Record<string, unknown>,
): Record<string, unknown> {
  return {
    source: src,
    target: tgt,
    ...(catalogPayload || {}),
    relation_type:
      catalogPayload?.["relation_type"] || candidate?.["relation_type"] || "candidate",
    description:
      catalogPayload?.["description"] ||
      candidate?.["evidence_summary"] ||
      candidate?.["reason"] ||
      "",
  };
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
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

function buildChunkPreview(text: string, maxChunks = 2, maxChars = 480) {
  const chunks = text
    .split(/\n{2,}/)
    .map((chunk) => chunk.trim())
    .filter(Boolean);

  if (!chunks.length) {
    const trimmed = text.trim();
    if (trimmed.length <= maxChars) {
      return { text: trimmed, truncated: false };
    }
    return { text: `${trimmed.slice(0, maxChars).trim()}...`, truncated: true };
  }

  const selected: string[] = [];
  let totalChars = 0;
  for (let index = 0; index < chunks.length; index += 1) {
    const chunk = chunks[index];
    const joinedLength = totalChars + chunk.length + (selected.length ? 2 : 0);
    if (selected.length >= maxChunks || joinedLength > maxChars) {
      break;
    }
    selected.push(chunk);
    totalChars = joinedLength;
  }

  if (!selected.length) {
    const firstChunk = chunks[0];
    return {
      text: `${firstChunk.slice(0, maxChars).trim()}...`,
      truncated: firstChunk.length > maxChars || chunks.length > 1,
    };
  }

  const previewText = selected.join("\n\n");
  const truncated = selected.length < chunks.length || previewText.length < text.trim().length;
  return {
    text: truncated ? `${previewText}...` : previewText,
    truncated,
  };
}

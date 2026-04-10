import json
from pathlib import Path

from fastapi.testclient import TestClient

from data_platform.backend.main import app


def _write_run(root: Path) -> tuple[Path, Path]:
    image_path = root / "demo.png"
    image_path.write_bytes(b"fake-image")

    run_dir = root / "output" / "1774019999"
    generate_dir = run_dir / "generate"
    generate_dir.mkdir(parents=True)

    (run_dir / "config.yaml").write_text(
        """
nodes:
  - id: generate
    params:
      method: vqa
""".strip(),
        encoding="utf-8",
    )

    valid_record = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "What does the graph show?"},
                    {
                        "type": "image",
                        "image": str(image_path),
                        "image_caption": ["Figure caption with caption span and grounded relation."],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It shows a grounded relation."}],
            },
        ],
        "sub_graph": json.dumps(
            {
                "nodes": [
                    [
                        "NODE_A",
                        {
                            "entity_type": "IMAGE",
                            "entity_name": "NODE_A",
                            "description": "An image anchor",
                            "evidence_span": "caption span",
                            "source_id": "context-2",
                        },
                    ]
                ],
                "edges": [
                    [
                        "NODE_A",
                        "NODE_B",
                        {
                            "relation_type": "supports",
                            "description": "A grounded relation",
                            "evidence_span": "edge span",
                        },
                    ]
                ],
            },
            ensure_ascii=False,
        ),
        "visualization_trace": json.dumps(
            {
                "schema_version": "visual_core_family_timeline_v1",
                "sampler_version": "family_llm_v2",
                "seed_node_id": "NODE_A",
                "seed_image_path": str(image_path),
                "graph_catalog": {
                    "nodes": {
                        "NODE_A": {
                            "node_id": "NODE_A",
                            "entity_type": "IMAGE",
                            "entity_name": "NODE_A",
                        },
                        "NODE_B": {
                            "node_id": "NODE_B",
                            "entity_type": "TEXT",
                            "entity_name": "NODE_B",
                        },
                    },
                    "edges": {
                        "NODE_A->NODE_B": {
                            "source": "NODE_A",
                            "target": "NODE_B",
                            "relation_type": "supports",
                        }
                    },
                },
                "events": [
                    {
                        "event_id": "NODE_A:atomic:1:bootstrap_state_created",
                        "order": 1,
                        "qa_family": "atomic",
                        "phase": "bootstrap",
                        "event_type": "bootstrap_state_created",
                        "status": "ok",
                        "selected_node_ids": ["NODE_A"],
                        "selected_edge_pairs": [],
                        "candidate_pool": [],
                        "chosen_candidate": {},
                        "judge": {},
                        "reason": "Start from image.",
                        "termination_reason": "",
                    },
                    {
                        "event_id": "NODE_A:atomic:2:qa_generated",
                        "order": 2,
                        "qa_family": "atomic",
                        "phase": "generation",
                        "event_type": "qa_generated",
                        "status": "generated",
                        "selected_node_ids": ["NODE_A", "NODE_B"],
                        "selected_edge_pairs": [["NODE_A", "NODE_B"]],
                        "candidate_pool": [],
                        "chosen_candidate": {},
                        "judge": {},
                        "reason": "",
                        "termination_reason": "",
                        "subgraph_id": "demo-subgraph",
                        "generator_key": "atomic",
                        "question": "What does the graph show?",
                        "answer": "It shows a grounded relation.",
                        "qa_trace_id": "trace-valid",
                        "sub_graph_summary": {"node_count": 1, "edge_count": 1},
                    },
                ],
            },
            ensure_ascii=False,
        ),
        "_trace_id": "trace-valid",
    }
    invalid_graph_record = {
        "messages": [
            {"role": "user", "content": [{"text": "Second question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Second answer"}]},
        ],
        "sub_graph": "{not-json}",
        "_trace_id": "trace-invalid",
    }

    output_file = generate_dir / "generate_demo.jsonl"
    output_file.write_text(
        "\n".join(
            [
                json.dumps(valid_record, ensure_ascii=False),
                json.dumps(invalid_graph_record, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )
    return output_file, image_path


def test_scan_list_and_detail_endpoints(tmp_path: Path):
    _write_run(tmp_path)
    client = TestClient(app)

    scan_response = client.post("/api/imports/scan", json={"root_path": str(tmp_path)})
    assert scan_response.status_code == 200
    scan_payload = scan_response.json()
    assert scan_payload["run_count"] == 1
    assert scan_payload["sample_count"] == 2
    assert scan_payload["runs"][0]["task_type"] == "vqa"
    assert scan_payload["runs"][0]["has_image"] is True
    assert scan_payload["runs"][0]["has_sub_graph"] is True

    run_id = scan_payload["runs"][0]["run_id"]
    samples_response = client.get(
        f"/api/runs/{run_id}/samples",
        params={"page": 1, "page_size": 10, "search": "graph", "has_graph": True},
    )
    assert samples_response.status_code == 200
    samples_payload = samples_response.json()
    assert samples_payload["total"] == 1
    assert samples_payload["items"][0]["question"] == "What does the graph show?"

    sample_id = samples_payload["items"][0]["sample_id"]
    detail_response = client.get(f"/api/samples/{sample_id}")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["answer"] == "It shows a grounded relation."
    assert detail_payload["sub_graph_summary"]["node_count"] == 1
    assert len(detail_payload["evidence_items"]) == 2
    assert detail_payload["image_path"].endswith("demo.png")
    assert len(detail_payload["source_contexts"]) == 3
    assert detail_payload["source_contexts"][0]["content"] == "What does the graph show?"
    assert detail_payload["source_contexts"][1]["content_type"] == "image_caption"
    assert detail_payload["source_contexts"][1]["content"] == (
        "Figure caption with caption span and grounded relation."
    )
    assert detail_payload["evidence_items"][0]["id"] == "node-1-NODE_A"
    assert detail_payload["evidence_items"][0]["graph_item_id"] == "NODE_A"
    assert detail_payload["evidence_items"][1]["source_id"] == "context-1"
    assert detail_payload["visualization_trace"]["schema_version"] == (
        "visual_core_family_timeline_v1"
    )
    assert detail_payload["visualization_trace"]["events"][-1]["event_type"] == (
        "qa_generated"
    )


def test_invalid_graph_is_preserved_without_breaking_browse(tmp_path: Path):
    _write_run(tmp_path)
    client = TestClient(app)
    client.post("/api/imports/scan", json={"root_path": str(tmp_path)})

    samples_response = client.get(
        "/api/runs/1774019999/samples",
        params={"page": 1, "page_size": 10, "search": "Second"},
    )
    sample_id = samples_response.json()["items"][0]["sample_id"]

    detail_response = client.get(f"/api/samples/{sample_id}")
    assert detail_response.status_code == 200
    payload = detail_response.json()
    assert payload["question"] == "Second question"
    assert payload["sub_graph"] is None
    assert payload["graph_parse_error"]
    assert payload["source_contexts"][0]["source_id"] == "context-1"


def test_assets_endpoint_allows_only_indexed_paths(tmp_path: Path):
    _, image_path = _write_run(tmp_path)
    client = TestClient(app)
    client.post("/api/imports/scan", json={"root_path": str(tmp_path)})

    asset_response = client.get("/api/assets", params={"path": str(image_path)})
    assert asset_response.status_code == 200
    assert asset_response.content == b"fake-image"

    forbidden_response = client.get("/api/assets", params={"path": str(tmp_path / "other.png")})
    assert forbidden_response.status_code == 403

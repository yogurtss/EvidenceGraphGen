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
                "content": [{"text": "What does the graph show?"}, {"image": str(image_path)}],
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
                            "source_id": "chunk-1",
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
                            "source_id": "chunk-2",
                        },
                    ]
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


def test_assets_endpoint_allows_only_indexed_paths(tmp_path: Path):
    _, image_path = _write_run(tmp_path)
    client = TestClient(app)
    client.post("/api/imports/scan", json={"root_path": str(tmp_path)})

    asset_response = client.get("/api/assets", params={"path": str(image_path)})
    assert asset_response.status_code == 200
    assert asset_response.content == b"fake-image"

    forbidden_response = client.get("/api/assets", params={"path": str(tmp_path / "other.png")})
    assert forbidden_response.status_code == 403

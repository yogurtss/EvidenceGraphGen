from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .models import RunRecord, SamplePage, SampleRecord, ScanRequest, ScanResponse
from .store import DataPlatformStore

app = FastAPI(title="GraphGen Data Platform API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = DataPlatformStore(base_dir=Path.cwd())


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/imports/scan", response_model=ScanResponse)
def scan_imports(request: ScanRequest) -> ScanResponse:
    try:
        runs, sample_count = store.scan(request.root_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ScanResponse(
        root_path=request.root_path,
        run_count=len(runs),
        sample_count=sample_count,
        runs=runs,
    )


@app.get("/api/runs", response_model=list[RunRecord])
def list_runs() -> list[RunRecord]:
    return store.list_runs()


@app.get("/api/runs/{run_id}/samples", response_model=SamplePage)
def list_samples(
    run_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search: str | None = None,
    has_image: bool | None = None,
    has_graph: bool | None = None,
) -> SamplePage:
    try:
        return store.list_samples(
            run_id,
            page=page,
            page_size=page_size,
            search=search,
            has_image=has_image,
            has_graph=has_graph,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc


@app.get("/api/samples/{sample_id}", response_model=SampleRecord)
def get_sample(sample_id: str) -> SampleRecord:
    try:
        return store.get_sample(sample_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}") from exc


@app.get("/api/assets")
def get_asset(path: str = Query(..., min_length=1)) -> FileResponse:
    asset_path = Path(path).resolve()
    if not store.is_asset_allowed(str(asset_path)):
        raise HTTPException(status_code=403, detail="Asset path is not indexed")
    if not asset_path.exists() or not asset_path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(asset_path)

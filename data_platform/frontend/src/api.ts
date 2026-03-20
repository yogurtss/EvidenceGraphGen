import type { ImportedRun, SampleDetail, SamplePage } from "./types";

export async function scanRuns(rootPath: string) {
  const response = await fetch("/api/imports/scan", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ root_path: rootPath }),
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  return (await response.json()) as {
    root_path: string;
    run_count: number;
    sample_count: number;
    runs: ImportedRun[];
  };
}

export async function fetchRuns() {
  const response = await fetch("/api/runs");
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as ImportedRun[];
}

export async function fetchSamples(params: {
  runId: string;
  page: number;
  pageSize: number;
  search: string;
  hasImage: boolean | undefined;
  hasGraph: boolean | undefined;
}) {
  const query = new URLSearchParams({
    page: String(params.page),
    page_size: String(params.pageSize),
  });
  if (params.search.trim()) {
    query.set("search", params.search.trim());
  }
  if (params.hasImage !== undefined) {
    query.set("has_image", String(params.hasImage));
  }
  if (params.hasGraph !== undefined) {
    query.set("has_graph", String(params.hasGraph));
  }

  const response = await fetch(`/api/runs/${params.runId}/samples?${query.toString()}`);
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as SamplePage;
}

export async function fetchSampleDetail(sampleId: string) {
  const response = await fetch(`/api/samples/${sampleId}`);
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as SampleDetail;
}

export function buildAssetUrl(path: string) {
  return `/api/assets?path=${encodeURIComponent(path)}`;
}

async function readError(response: Response) {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail || `Request failed with status ${response.status}`;
  } catch {
    return `Request failed with status ${response.status}`;
  }
}

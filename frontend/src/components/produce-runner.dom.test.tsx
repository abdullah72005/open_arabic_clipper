// @vitest-environment jsdom
import React from "react";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  Candidate,
  CandidateRefinement,
  Job,
  Materialization,
  MaterializationQueueResponse,
  RightsStatus,
  Source
} from "@/lib/api-client";
import type { ProduceApi, QueuedJob } from "@/lib/produce-runner";

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) =>
    React.createElement("a", { href }, children)
}));

// The component imports the module-level `api` singleton. Expose a mutable holder
// through a Proxy so each test can install its own deterministic fake without
// re-importing the module.
const holder = vi.hoisted(() => ({ current: null as unknown as ProduceApi }));
vi.mock("@/lib/api-client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api-client")>();
  return {
    ...actual,
    api: new Proxy({} as ProduceApi, {
      get: (_target, prop) => {
        const value = (holder.current as unknown as Record<string | symbol, unknown>)[prop];
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(holder.current)
          : value;
      }
    })
  };
});

import { ProduceRunner } from "./produce-runner";

type Deferred<T> = { promise: Promise<T>; resolve: (value: T) => void; reject: (error: unknown) => void };

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const source: Source = {
  id: "src-1",
  source_uri: "https://example.org/v",
  original_filename: null,
  rights_status: "UNKNOWN",
  lifecycle_state: "READY_FOR_ANALYSIS",
  created_at: "2026-10-01T00:00:00Z"
};

const succeededJob: Job = {
  id: "job-analysis",
  source_video_id: "src-1",
  kind: "CANDIDATE_ANALYSIS",
  status: "SUCCEEDED",
  retry_count: 0,
  error_code: null,
  error_message: null,
  created_at: "2026-10-01T00:00:00Z",
  started_at: "2026-10-01T00:00:01Z",
  completed_at: "2026-10-01T00:00:02Z"
};

const candidate: Candidate = {
  id: "cand-1",
  source_video_id: "src-1",
  candidate_key: "key-1",
  disposition: "CANDIDATE",
  start_time: 0,
  end_time: 30,
  transcript_excerpt: "text",
  primary_content_type: "STORY",
  clip_score: 0.7,
  transcript_confidence: 0.9,
  uncertainty_severity: 0.1,
  refinement_reasons: [],
  dialect_profile: "EGYPTIAN",
  code_switch_suspected: false
};

const refinement: CandidateRefinement = {
  id: "ref-1",
  source_video_id: "src-1",
  clip_candidate_id: "cand-1",
  priority: "FINAL_CLIP",
  status: "FINAL_TRANSCRIPT_READY",
  quality_level: "FINAL_CLIP",
  coarse_start: 0,
  coarse_end: 30,
  context_start: 0,
  context_end: 30,
  refined_start: 0,
  refined_end: 30,
  automatic_transcript: "text",
  manual_transcript: null,
  final_transcript: "text",
  word_timestamps: [],
  confidence: 0.9,
  dialect_profile: "EGYPTIAN",
  dialect_confidence: 0.9,
  code_switch_evidence: {},
  transcript_evidence: [],
  entity_evidence: [],
  unresolved_spans: [],
  provider_evidence: {},
  routing_evidence: {},
  input_fingerprint: "in",
  output_fingerprint: "out",
  cache_eligible: true,
  metrics: {},
  processing_duration: 1,
  created_at: "2026-10-01T00:00:00Z",
  updated_at: "2026-10-01T00:00:00Z"
};

function materialization(overrides: Partial<Materialization> = {}): Materialization {
  return {
    id: "mat-1",
    source_video_id: "src-1",
    clip_candidate_id: "cand-1",
    profile_key: "default",
    priority: "CANDIDATE",
    lifecycle: "AWAITING_RENDER",
    semantic_outcome: null,
    human_review_required: false,
    publication_ready: false,
    is_current: true,
    repair_count: 0,
    reason_codes: [],
    input_fingerprint: "in",
    output_fingerprint: "out",
    policy_version: "stage6-v1",
    schema_version: "stage6-schema-v1",
    render_execution_id: null,
    selection_id: "sel-1",
    render_contract_id: "con-1",
    is_effective: false,
    live_freshness: "CURRENT",
    readiness: { review_ready: false, reasons: [], publication_ready: false },
    component_statuses: {},
    findings: [],
    provider_attempts: {},
    artifact_available: false,
    ...overrides
  };
}

function queueResponse(overrides: Partial<MaterializationQueueResponse> = {}): MaterializationQueueResponse {
  return {
    job_id: "job-prep",
    status: "QUEUED",
    queued: true,
    cached: false,
    active: true,
    materialization_id: "mat-1",
    render_execution_id: null,
    phase: "PREPARE",
    lifecycle: "QUEUED",
    semantic_outcome: null,
    human_review_required: false,
    publication_ready: false,
    review_ready: false,
    reasons: [],
    skipped_reason: null,
    ...overrides
  };
}

class FakeApi implements ProduceApi {
  listJobsResult: Job[] = [succeededJob];
  activeIngestJob: Job | null = null;
  queueMaterializationResult: MaterializationQueueResponse = queueResponse();
  cancelMaterializationFailures = 0;

  readonly cancelMaterializationCalls: string[] = [];
  readonly cancelJobCalls: string[] = [];
  private cancelAttempts = 0;

  queueTransformationAnalysisGate: Deferred<QueuedJob> | null = null;
  queueMaterializationGate: Deferred<MaterializationQueueResponse> | null = null;
  getMaterializationGate: Deferred<Materialization> | null = null;

  readonly transformationAnalysisStarted = deferred<void>();
  readonly queueMaterializationStarted = deferred<void>();
  readonly getMaterializationStarted = deferred<void>();

  submitUrls(): Promise<Source[]> {
    return Promise.resolve([source]);
  }
  listJobs(): Promise<Job[]> {
    return Promise.resolve(this.activeIngestJob ? [this.activeIngestJob] : this.listJobsResult);
  }
  getJob(): Promise<Job> {
    return Promise.resolve(succeededJob);
  }
  getCandidateAnalysis(): Promise<unknown> {
    return Promise.resolve({});
  }
  listCandidates(): Promise<Candidate[]> {
    return Promise.resolve([candidate]);
  }
  queueCandidateRefinement(): Promise<QueuedJob> {
    return Promise.resolve({ job_id: null, status: "SUCCEEDED", cached: true });
  }
  listCandidateRefinements(): Promise<CandidateRefinement[]> {
    return Promise.resolve([refinement]);
  }
  queueTransformationAnalysis(): Promise<QueuedJob> {
    this.transformationAnalysisStarted.resolve();
    if (this.queueTransformationAnalysisGate) return this.queueTransformationAnalysisGate.promise;
    return Promise.resolve({ job_id: null, status: "SUCCEEDED", cached: true });
  }
  getCandidateTransformationAnalysis() {
    return Promise.resolve({ recommended_strategies: [{}], eligibility_outcome: "ELIGIBLE" });
  }
  queueTransformationPlans(): Promise<QueuedJob> {
    return Promise.resolve({ job_id: null, status: "SUCCEEDED", cached: true });
  }
  getCandidateTransformationPlans() {
    return Promise.resolve({ plans: [{}], planning_outcome: "PLANNED" });
  }
  queueTransformationGovernance(): Promise<QueuedJob> {
    return Promise.resolve({ job_id: null, status: "SUCCEEDED", cached: true });
  }
  getCandidateTransformationGovernance(): Promise<unknown> {
    return Promise.resolve({});
  }
  createTransformationSelection() {
    return Promise.resolve({
      selected_plan_id: "plan-1",
      effective: true,
      selection_reason_codes: [],
      status: "PLAN_SELECTED"
    });
  }
  createRenderContract() {
    return Promise.resolve({
      contract_ready: true,
      effective: true,
      reason_codes: [],
      status: "EXECUTABLE"
    });
  }
  queueVisualComposition(): Promise<QueuedJob> {
    return Promise.resolve({ job_id: null, status: "SUCCEEDED", cached: true });
  }
  getCandidateVisualComposition() {
    return Promise.resolve({ plan_ready: true, reason_codes: [], status: "READY" });
  }
  queueMaterialization(): Promise<MaterializationQueueResponse> {
    this.queueMaterializationStarted.resolve();
    if (this.queueMaterializationGate) return this.queueMaterializationGate.promise;
    return Promise.resolve(this.queueMaterializationResult);
  }
  getMaterialization(): Promise<Materialization> {
    this.getMaterializationStarted.resolve();
    if (this.getMaterializationGate) return this.getMaterializationGate.promise;
    return Promise.resolve(materialization());
  }
  resumeMaterialization(): Promise<MaterializationQueueResponse> {
    return Promise.resolve(queueResponse({ phase: "FINALIZE" }));
  }
  cancelMaterialization(id: string): Promise<Materialization> {
    this.cancelMaterializationCalls.push(id);
    this.cancelAttempts += 1;
    if (this.cancelAttempts <= this.cancelMaterializationFailures) {
      return Promise.reject(new Error("cancel unavailable"));
    }
    return Promise.resolve(materialization({ id, lifecycle: "CANCELLED" }));
  }
  cancelJob(id: string): Promise<Job> {
    this.cancelJobCalls.push(id);
    return Promise.resolve({ ...succeededJob, id, status: "CANCELLED" });
  }
  materializationArtifactUrl(id: string): string {
    return `/api/materializations/${id}/artifact`;
  }
}

function produce() {
  const api = new FakeApi();
  holder.current = api;
  const utils = render(<ProduceRunner />);
  const input = screen.getByPlaceholderText("https://example.org/video");
  fireEvent.change(input, { target: { value: "https://example.org/v" } });
  fireEvent.click(screen.getByRole("button", { name: "Produce final clip" }));
  return { api, ...utils };
}

afterEach(() => {
  cleanup();
  holder.current = null as unknown as ProduceApi;
  vi.restoreAllMocks();
});

beforeEach(() => {
  vi.clearAllMocks();
});

describe("ProduceRunner mounted cancellation ownership", () => {
  it("surfaces an unconfirmed cancellation, settles polling, then retries for real", async () => {
    const { api } = produce();
    api.cancelMaterializationFailures = 1;
    api.getMaterializationGate = deferred<Materialization>();
    await api.getMaterializationStarted.promise;

    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    await screen.findByText(/Cancellation could not be confirmed/);
    // Produce stays disabled while cancellation is unresolved.
    expect(
      (screen.getByRole("button", { name: /Producing/ }) as HTMLButtonElement).disabled
    ).toBe(true);
    expect(api.cancelMaterializationCalls).toEqual(["mat-1"]);

    // Polling settles (the pending read returns) without falsely confirming.
    await act(async () => {
      api.getMaterializationGate?.resolve(materialization());
    });
    await waitFor(() => expect(screen.queryByText(/Cancellation could not be confirmed/)).toBeTruthy());

    // A real retry performs a fresh cancellation and confirms.
    fireEvent.click(screen.getByRole("button", { name: "Retry cancel" }));
    await screen.findByText("Cancelled.");
    expect(api.cancelMaterializationCalls).toEqual(["mat-1", "mat-1"]);
    expect(await screen.findByRole("button", { name: "Produce final clip" })).toBeTruthy();
  });

  it("reconciles a late pre-Stage-6 queue response and cancels its exact job", async () => {
    const { api } = produce();
    api.queueTransformationAnalysisGate = deferred<QueuedJob>();
    await api.transformationAnalysisStarted.promise;

    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    await act(async () => {
      api.queueTransformationAnalysisGate?.resolve({
        job_id: "job-analysis-late",
        status: "QUEUED",
        cached: false
      });
    });
    await screen.findByText("Cancelled.");
    expect(api.cancelJobCalls).toEqual(["job-analysis-late"]);
    expect(api.cancelMaterializationCalls).toEqual([]);
  });

  it("cancels the exact active ingest job observed for the submitted source", async () => {
    const { api } = produce();
    api.activeIngestJob = {
      ...succeededJob,
      id: "job-ingest-active",
      kind: "INGEST",
      status: "RUNNING"
    };
    await screen.findByText(/Processing: INGEST/);

    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    await screen.findByText("Cancelled.");
    expect(api.cancelJobCalls).toEqual(["job-ingest-active"]);
    expect(api.cancelMaterializationCalls).toEqual([]);
  });

  it("cancels a successful late Stage 6 submission by its returned identity", async () => {
    const { api } = produce();
    api.queueMaterializationGate = deferred<MaterializationQueueResponse>();
    await api.queueMaterializationStarted.promise;

    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    await act(async () => {
      api.queueMaterializationGate?.resolve(queueResponse({ materialization_id: "mat-late" }));
    });
    await screen.findByText("Cancelled.");
    expect(api.cancelMaterializationCalls).toEqual(["mat-late"]);
    expect(api.cancelJobCalls).toEqual([]);
  });

  it("does not expose a later result when stopped during a pending GET", async () => {
    const { api, container } = produce();
    api.getMaterializationGate = deferred<Materialization>();
    await api.getMaterializationStarted.promise;

    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    await screen.findByText("Cancelled.");

    // A review-ready response arriving after stop must never be exposed.
    await act(async () => {
      api.getMaterializationGate?.resolve(
        materialization({
          lifecycle: "COMPLETE",
          semantic_outcome: "READY_FOR_HUMAN_REVIEW",
          human_review_required: true,
          artifact_available: true,
          readiness: { review_ready: true, reasons: [], publication_ready: false }
        })
      );
    });
    await waitFor(() => expect(container.querySelector("video")).toBeNull());
    expect(screen.queryByText(/ready for human review/i)).toBeNull();
  });
});

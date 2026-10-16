import type { Candidate, CandidateRefinement } from "@/lib/api-client";

export const PRODUCE_STEP_KEYS = [
  "ingest",
  "refine",
  "analyze",
  "plan",
  "govern",
  "select",
  "contract",
  "compose",
  "materialize"
] as const;

export type ProduceStepKey = (typeof PRODUCE_STEP_KEYS)[number];

export interface ProduceStep {
  key: ProduceStepKey;
  label: string;
  detail: string;
}

export const PRODUCE_STEPS: ProduceStep[] = [
  { key: "ingest", label: "Ingest & transcribe", detail: "Download/probe, transcribe, and find clip candidates." },
  { key: "refine", label: "Refine transcript", detail: "Targeted audio-verified transcript and boundary refinement." },
  { key: "analyze", label: "Transformation analysis", detail: "Stage 4.0 eligibility and recommended strategies." },
  { key: "plan", label: "Generate plan", detail: "Stage 4.1 concrete structured transformation plan." },
  { key: "govern", label: "Govern plan", detail: "Stage 4.2 policy and originality governance." },
  { key: "select", label: "Select final plan", detail: "Stage 4.3 deterministic final-plan selection." },
  { key: "contract", label: "Render contract", detail: "Stage 5.0 executable render contract." },
  { key: "compose", label: "Visual composition", detail: "Stage 5.1 framing, captions, and overlays." },
  { key: "materialize", label: "Materialize & assess", detail: "Stage 6 transformed short: render, narration, QC, and final review assessment." }
];

export type StepState = "pending" | "active" | "done" | "error";

export type StepStates = Record<ProduceStepKey, StepState>;

export function initialStepStates(): StepStates {
  return PRODUCE_STEP_KEYS.reduce((accumulator, key) => {
    accumulator[key] = "pending";
    return accumulator;
  }, {} as StepStates);
}

const USABLE_DISPOSITIONS = new Set<Candidate["disposition"]>([
  "CANDIDATE",
  "CANDIDATE_NEEDS_REFINEMENT"
]);

/**
 * Deterministic auto-pick: highest `clip_score` among retained `CANDIDATE`
 * proposals, falling back to `CANDIDATE_NEEDS_REFINEMENT` only when no plain
 * candidate exists. Rejected proposals are never eligible.
 */
export function selectTopCandidate(candidates: Candidate[]): Candidate | null {
  const preferred = candidates.filter((candidate) => candidate.disposition === "CANDIDATE");
  const fallback = candidates.filter(
    (candidate) => candidate.disposition === "CANDIDATE_NEEDS_REFINEMENT"
  );
  const pool = preferred.length > 0 ? preferred : fallback;
  if (pool.length === 0) return null;
  return [...pool].sort((left, right) => right.clip_score - left.clip_score)[0] ?? null;
}

export function isRetainedDisposition(disposition: Candidate["disposition"]): boolean {
  return USABLE_DISPOSITIONS.has(disposition);
}

const FINAL_READY = new Set(["FINAL_TRANSCRIPT_READY"]);
const CANDIDATE_USABLE = new Set([
  "CANDIDATE_REFINED",
  "NEEDS_MANUAL_TRANSCRIPT_REVIEW",
  "PROVIDER_DEGRADED"
]);

function newest(rows: CandidateRefinement[]): CandidateRefinement | undefined {
  return [...rows].sort(
    (left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at)
  )[0];
}

/**
 * Mirror the backend's effective-refinement resolution: prefer a usable
 * `FINAL_CLIP` row (publication grade), otherwise a usable `CANDIDATE` row.
 */
export function pickEffectiveRefinement(
  refinements: CandidateRefinement[]
): CandidateRefinement | null {
  const final = newest(refinements.filter((row) => row.priority === "FINAL_CLIP"));
  if (final && FINAL_READY.has(final.status)) return final;
  const candidate = newest(refinements.filter((row) => row.priority === "CANDIDATE"));
  if (candidate && CANDIDATE_USABLE.has(candidate.status)) return candidate;
  return null;
}

export function isTerminalJobStatus(status: string): boolean {
  return status === "SUCCEEDED" || status === "FAILED" || status === "CANCELLED";
}

export function isActiveJobStatus(status: string): boolean {
  return status === "QUEUED" || status === "RUNNING";
}

export function refinementStatusLabel(status: string): string {
  return status.replaceAll("_", " ").toLowerCase();
}

import { describe, expect, it } from "vitest";

import type { Candidate, CandidateRefinement } from "@/lib/api-client";
import {
  initialStepStates,
  isTerminalJobStatus,
  pickEffectiveRefinement,
  PRODUCE_STEP_KEYS,
  selectTopCandidate
} from "@/lib/produce";

function candidate(overrides: Partial<Candidate>): Candidate {
  return {
    id: "c1",
    source_video_id: "s1",
    candidate_key: "candidate-1",
    disposition: "CANDIDATE",
    start_time: 0,
    end_time: 5,
    transcript_excerpt: "excerpt",
    primary_content_type: "STORY",
    clip_score: 0.5,
    transcript_confidence: 0.8,
    uncertainty_severity: 0,
    refinement_reasons: [],
    dialect_profile: null,
    code_switch_suspected: false,
    ...overrides
  };
}

function refinement(overrides: Partial<CandidateRefinement>): CandidateRefinement {
  return {
    id: "r1",
    source_video_id: "s1",
    clip_candidate_id: "c1",
    priority: "CANDIDATE",
    status: "CANDIDATE_REFINED",
    quality_level: "MEDIUM",
    coarse_start: 0,
    coarse_end: 5,
    context_start: 0,
    context_end: 5,
    refined_start: 0,
    refined_end: 5,
    automatic_transcript: "text",
    manual_transcript: null,
    final_transcript: "text",
    word_timestamps: [],
    confidence: 0.8,
    dialect_profile: null,
    dialect_confidence: 0,
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
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides
  };
}

describe("initialStepStates", () => {
  it("marks every step pending", () => {
    const states = initialStepStates();
    expect(Object.keys(states).sort()).toEqual([...PRODUCE_STEP_KEYS].sort());
    expect(Object.values(states).every((value) => value === "pending")).toBe(true);
  });
});

describe("selectTopCandidate", () => {
  it("picks the highest-scoring retained candidate", () => {
    const picked = selectTopCandidate([
      candidate({ id: "a", clip_score: 0.4 }),
      candidate({ id: "b", clip_score: 0.9 }),
      candidate({ id: "c", clip_score: 0.7 })
    ]);
    expect(picked?.id).toBe("b");
  });

  it("prefers a plain CANDIDATE over a higher-scoring needs-refinement row", () => {
    const picked = selectTopCandidate([
      candidate({ id: "a", clip_score: 0.4, disposition: "CANDIDATE" }),
      candidate({ id: "b", clip_score: 0.99, disposition: "CANDIDATE_NEEDS_REFINEMENT" })
    ]);
    expect(picked?.id).toBe("a");
  });

  it("falls back to CANDIDATE_NEEDS_REFINEMENT when no plain candidate exists", () => {
    const picked = selectTopCandidate([
      candidate({ id: "a", clip_score: 0.4, disposition: "CANDIDATE_NEEDS_REFINEMENT" }),
      candidate({ id: "b", clip_score: 0.6, disposition: "CANDIDATE_NEEDS_REFINEMENT" })
    ]);
    expect(picked?.id).toBe("b");
  });

  it("never selects a rejected proposal", () => {
    const picked = selectTopCandidate([
      candidate({ id: "a", clip_score: 0.99, disposition: "DO_NOT_CLIP" }),
      candidate({ id: "b", clip_score: 0.99, disposition: "DO_NOT_CLIP_RECENTLY_REDUNDANT" })
    ]);
    expect(picked).toBeNull();
  });

  it("returns null for an empty list", () => {
    expect(selectTopCandidate([])).toBeNull();
  });
});

describe("pickEffectiveRefinement", () => {
  it("prefers a ready final-clip refinement", () => {
    const picked = pickEffectiveRefinement([
      refinement({ id: "c", priority: "CANDIDATE", status: "CANDIDATE_REFINED" }),
      refinement({ id: "f", priority: "FINAL_CLIP", status: "FINAL_TRANSCRIPT_READY" })
    ]);
    expect(picked?.id).toBe("f");
  });

  it("falls back to a usable candidate row when the final row is unresolved", () => {
    const picked = pickEffectiveRefinement([
      refinement({ id: "f", priority: "FINAL_CLIP", status: "NEEDS_MANUAL_TRANSCRIPT_REVIEW" }),
      refinement({ id: "c", priority: "CANDIDATE", status: "CANDIDATE_REFINED" })
    ]);
    expect(picked?.id).toBe("c");
  });

  it("treats needs-manual-review candidate text as usable, matching the backend", () => {
    const picked = pickEffectiveRefinement([
      refinement({ id: "c", priority: "CANDIDATE", status: "NEEDS_MANUAL_TRANSCRIPT_REVIEW" })
    ]);
    expect(picked?.id).toBe("c");
  });

  it("returns null when nothing is usable", () => {
    expect(pickEffectiveRefinement([])).toBeNull();
    expect(
      pickEffectiveRefinement([refinement({ priority: "FINAL_CLIP", status: "FAILED" })])
    ).toBeNull();
  });

  it("uses the newest row per priority", () => {
    const picked = pickEffectiveRefinement([
      refinement({
        id: "old",
        priority: "CANDIDATE",
        status: "CANDIDATE_REFINED",
        updated_at: "2026-01-01T00:00:00Z"
      }),
      refinement({
        id: "new",
        priority: "CANDIDATE",
        status: "CANDIDATE_REFINED",
        updated_at: "2026-02-01T00:00:00Z"
      })
    ]);
    expect(picked?.id).toBe("new");
  });
});

describe("isTerminalJobStatus", () => {
  it("recognizes terminal states only", () => {
    expect(isTerminalJobStatus("SUCCEEDED")).toBe(true);
    expect(isTerminalJobStatus("FAILED")).toBe(true);
    expect(isTerminalJobStatus("CANCELLED")).toBe(true);
    expect(isTerminalJobStatus("RUNNING")).toBe(false);
    expect(isTerminalJobStatus("QUEUED")).toBe(false);
  });
});

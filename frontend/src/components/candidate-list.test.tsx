import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { CandidateList } from "./candidate-list";
import type { Candidate, CandidateAnalysis, CandidateRefinement } from "@/lib/api-client";

const analysis: CandidateAnalysis = {
  provider_status: "DETERMINISTIC",
  semantic_provider_mode: "deterministic",
  cache_eligible: true,
  metrics: { candidates_retained: 2, refinement_needed_count: 1, redundant_count: 1 }
};

const candidates: Candidate[] = [
  {
    id: "candidate-1",
    source_video_id: "source-1",
    candidate_key: "key-1",
    disposition: "CANDIDATE",
    start_time: 65,
    end_time: 95,
    transcript_excerpt: "معلومة غريبة ومفاجأة",
    primary_content_type: "SURPRISING_FACT",
    clip_score: 0.62,
    transcript_confidence: 0.9,
    uncertainty_severity: 0.1,
    refinement_reasons: [],
    dialect_profile: "EGYPTIAN",
    code_switch_suspected: false
  },
  {
    id: "candidate-2",
    source_video_id: "source-1",
    candidate_key: "key-2",
    disposition: "CANDIDATE_NEEDS_REFINEMENT",
    start_time: 5,
    end_time: 40,
    transcript_excerpt: "كلام غير مؤكد",
    primary_content_type: "STORY",
    clip_score: 0.5,
    transcript_confidence: 0.4,
    uncertainty_severity: 0.5,
    refinement_reasons: ["UNRESOLVED_INDEX_TEXT", "LOW_CONFIDENCE_WORD_SPAN"],
    dialect_profile: "EGYPTIAN",
    code_switch_suspected: true
  }
];

const refinements: CandidateRefinement[] = [
  {
    id: "refinement-1",
    source_video_id: "source-1",
    clip_candidate_id: "candidate-2",
    priority: "CANDIDATE",
    status: "CANDIDATE_REFINED",
    quality_level: "CANDIDATE",
    coarse_start: 5,
    coarse_end: 40,
    context_start: 0,
    context_end: 45,
    refined_start: 7.2,
    refined_end: 38.5,
    automatic_transcript: "أنا عملت deploy للbackend امبارح",
    manual_transcript: null,
    final_transcript: "أنا عملت deploy للbackend امبارح",
    word_timestamps: [],
    confidence: 0.91,
    dialect_profile: "EGYPTIAN",
    dialect_confidence: 0.92,
    code_switch_evidence: { code_switch_recovered: ["deploy", "backend"] },
    transcript_evidence: [],
    entity_evidence: [],
    unresolved_spans: [],
    provider_evidence: {},
    routing_evidence: {},
    input_fingerprint: "input",
    output_fingerprint: "output",
    cache_eligible: true,
    metrics: {},
    processing_duration: 2.3,
    created_at: "2026-09-12T00:00:00Z",
    updated_at: "2026-09-12T00:00:00Z"
  }
];

describe("CandidateList", () => {
  it("renders compact candidates with time, score, disposition and refinement reasons", () => {
    const markup = renderToStaticMarkup(
      <CandidateList analysis={analysis} candidates={candidates} onSeek={() => {}} />
    );

    expect(markup).toContain("1:05–1:35");
    expect(markup).toContain("معلومة غريبة ومفاجأة");
    expect(markup).toContain("62%");
    expect(markup).toContain("2 retained");
    expect(markup).toContain("Needs refinement");
    expect(markup).toContain("UNRESOLVED_INDEX_TEXT");
    expect(markup).toContain("SURPRISING FACT");
  });

  it("shows empty and not-yet-analyzed states", () => {
    const empty = renderToStaticMarkup(
      <CandidateList analysis={analysis} candidates={[]} onSeek={() => {}} />
    );
    expect(empty).toContain("No candidates");

    const none = renderToStaticMarkup(
      <CandidateList analysis={null} candidates={[]} onSeek={() => {}} />
    );
    expect(none).toContain("No candidate analysis yet");
  });

  it("renders the automatically loaded Stage 3.5 result beside its candidate", () => {
    const markup = renderToStaticMarkup(
      <CandidateList
        analysis={analysis}
        candidates={candidates}
        onSeek={() => {}}
        refinements={refinements}
      />
    );

    expect(markup).toContain("Candidate refinement");
    expect(markup).toContain("CANDIDATE REFINED");
    expect(markup).toContain("0:07–0:38");
    expect(markup).toContain("أنا عملت deploy للbackend امبارح");
    expect(markup).toContain("deploy, backend");
  });

  it("exposes unresolved evidence and a manual-review form only when final refinement needs it", () => {
    const reviewRefinement: CandidateRefinement = {
      ...refinements[0],
      id: "refinement-review",
      priority: "FINAL_CLIP",
      status: "NEEDS_MANUAL_TRANSCRIPT_REVIEW",
      unresolved_spans: [{ span_id: "entity-0", readings: ["25", "95"], reason: "entity_conflict" }]
    };
    const markup = renderToStaticMarkup(
      <CandidateList analysis={analysis} candidates={candidates} onSeek={() => {}} refinements={[reviewRefinement]} />
    );

    expect(markup).toContain("Manual transcript review");
    expect(markup).toContain("entity conflict");
    expect(markup).toContain("25, 95");
  });

  it("does not offer Stage 3.5 work for a rejected Stage 3 proposal", () => {
    const rejected: Candidate = { ...candidates[0], id: "rejected", disposition: "DO_NOT_CLIP" };
    const markup = renderToStaticMarkup(
      <CandidateList analysis={analysis} candidates={[rejected]} onSeek={() => {}} />
    );

    expect(markup).not.toContain("Refine candidate");
  });
});

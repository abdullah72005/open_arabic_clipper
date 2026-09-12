import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { CandidateList } from "./candidate-list";
import type { Candidate, CandidateAnalysis } from "@/lib/api-client";

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
});

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { TranscriptStatus } from "./transcript-status";
import type { QualityResponse, Transcript } from "@/lib/api-client";

const transcript: Transcript = {
  source_video_id: "source-1",
  language: "ar",
  detected_language_probability: 0.98,
  whisper_model: "large-v3-turbo",
  transcription_options: {},
  raw_text: "دقل",
  normalized_text: "دقل",
  corrected_text: "دقل",
  final_text: "دقل",
  reconstruction_status: "PROVIDER_UNAVAILABLE",
  reconstruction_metadata: {
    provider_availability: "UNAVAILABLE",
    model: "qwen3:8b"
  },
  segments: [],
  duration: 1
};

const quality: QualityResponse = {
  reconstruction_status: "PROVIDER_UNAVAILABLE",
  quality: {
    audio_quality_score: 0.985,
    transcript_quality_score: 0.4,
    low_confidence_word_ratio: 0.2,
    unresolved_segment_ratio: 0.5,
    manual_review_required: true,
    conservative_source_floor: 0.4
  }
};

describe("TranscriptStatus", () => {
  it("makes unavailable reconstruction and split quality visible", () => {
    const markup = renderToStaticMarkup(
      <TranscriptStatus quality={quality} transcript={transcript} />
    );

    expect(markup).toContain("Reconstruction unavailable");
    expect(markup).toContain("Transcript quality 40%");
    expect(markup).toContain("Audio quality 99%");
    expect(markup).toContain("Manual review required");
  });

  it("shows routed focus spans with rounded probability", () => {
    const markup = renderToStaticMarkup(
      <TranscriptStatus
        quality={quality}
        transcript={{
          ...transcript,
          segments: [{
            start: 0,
            end: 1,
            text: "دقل",
            reconstruction_status: "LOW_CONFIDENCE_UNRESOLVED",
            focus_spans: [{ word: "دقل", start: 0, end: 0.4, probability: 0.54 }]
          }]
        }}
      />
    );

    expect(markup).toContain("دقل · 54%");
  });
});

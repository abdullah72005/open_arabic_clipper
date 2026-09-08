import { describe, expect, it } from "vitest";

import { ApiError, createApiClient } from "./api-client";

describe("API client", () => {
  it("builds a local source-media URL for transcript timestamp playback", () => {
    const client = createApiClient("http://api.test");

    expect(client.sourceMediaUrl("source-1")).toBe("http://api.test/api/sources/source-1/media");
  });

  it("loads timestamped auto-detected transcripts from the Stage 2 API", async () => {
    const fetcher = async (input: RequestInfo | URL) => {
      expect(String(input)).toBe("http://api.test/api/sources/source-1/transcript");
      return new Response(
        JSON.stringify({
          source_video_id: "source-1",
          language: "ar",
          detected_language_probability: 0.98,
          raw_text: "أهلا hello",
          normalized_text: "أهلا hello",
          segments: [{ start: 0, end: 1, text: "أهلا hello" }],
          duration: 1,
        }),
        { headers: { "content-type": "application/json" } },
      );
    };

    await expect(createApiClient("http://api.test", fetcher).getTranscript("source-1")).resolves.toMatchObject({
      language: "ar",
      segments: [{ start: 0, text: "أهلا hello" }],
    });
  });

  it("sends a manual transcript override with the stable segment index", async () => {
    const fetcher = async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(String(input)).toBe("http://api.test/api/sources/source-1/transcript/segments/0/override");
      expect(init?.method).toBe("POST");
      expect(init?.body).toBe(JSON.stringify({ text: "خلي بالك" }));
      return new Response(JSON.stringify({ start: 0, end: 1, text: "خطي بالك", final_text: "خلي بالك" }), {
        headers: { "content-type": "application/json" }
      });
    };

    await expect(
      createApiClient("http://api.test", fetcher).overrideTranscriptSegment("source-1", 0, "خلي بالك")
    ).resolves.toMatchObject({ final_text: "خلي بالك" });
  });

  it("queues an encoded contextual reconstruction request with force", async () => {
    const fetcher = async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(String(input)).toBe("http://api.test/api/sources/source%2Fid/reconstruct?force=true");
      expect(init?.method).toBe("POST");
      return new Response(JSON.stringify({ id: "job-1", kind: "RECONSTRUCTION", status: "QUEUED" }), {
        headers: { "content-type": "application/json" }
      });
    };

    await expect(createApiClient("http://api.test", fetcher).reconstructTranscript("source/id", true))
      .resolves.toMatchObject({ kind: "RECONSTRUCTION" });
  });

  it("loads split transcript quality and reconstruction status", async () => {
    const fetcher = async (input: RequestInfo | URL) => {
      expect(String(input)).toBe("http://api.test/api/sources/source-1/quality");
      return new Response(JSON.stringify({
        reconstruction_status: "PROVIDER_UNAVAILABLE",
        quality: {
          audio_quality_score: 0.985,
          transcript_quality_score: 0.4,
          low_confidence_word_ratio: 0.2,
          unresolved_segment_ratio: 0.5,
          manual_review_required: true,
          conservative_source_floor: 0.4
        }
      }), { headers: { "content-type": "application/json" } });
    };

    await expect(createApiClient("http://api.test", fetcher).getQuality("source-1"))
      .resolves.toMatchObject({ reconstruction_status: "PROVIDER_UNAVAILABLE" });
  });

  it("forces retranscription by default", async () => {
    const fetcher = async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(String(input)).toBe("http://api.test/api/sources/source%2Fid/retranscribe?force=true");
      expect(init?.method).toBe("POST");
      return new Response(JSON.stringify({ id: "job-1", kind: "TRANSCRIPTION", status: "QUEUED" }), {
        headers: { "content-type": "application/json" }
      });
    };

    await expect(createApiClient("http://api.test", fetcher).retranscribeTranscript("source/id"))
      .resolves.toMatchObject({ kind: "TRANSCRIPTION" });
  });

  it("uses the configured base URL and returns typed source data", async () => {
    const fetcher = async (input: RequestInfo | URL) => {
      expect(String(input)).toBe("http://api.test/sources");
      return new Response(
        JSON.stringify([
          {
            id: "0d9f0117-739f-4f34-b0cf-b3d0f1f5ebd1",
            source_uri: "https://example.test/video",
            original_filename: null,
            rights_status: "UNKNOWN",
            lifecycle_state: "INGEST",
            created_at: "2026-09-04T00:00:00Z"
          }
        ]),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    };

    const client = createApiClient("http://api.test", fetcher);
    await expect(client.listSources()).resolves.toMatchObject([{ rights_status: "UNKNOWN" }]);
  });

  it("turns failed API responses into a useful ApiError", async () => {
    const client = createApiClient("http://api.test", async () =>
      new Response(JSON.stringify({ detail: "not found" }), { status: 404 })
    );

    await expect(client.getSource("missing")).rejects.toEqual(
      expect.objectContaining<ApiError>({ name: "ApiError", status: 404, message: "not found" })
    );
  });

  it("includes the selected rights status with a local-file upload", async () => {
    const fetcher = async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(init?.method).toBe("POST");
      expect(init?.body).toBeInstanceOf(FormData);
      expect((init?.body as FormData).get("rights_status")).toBe("LICENSED");
      return new Response(
        JSON.stringify({
          id: "0d9f0117-739f-4f34-b0cf-b3d0f1f5ebd1",
          source_uri: "/storage/sources/example/licensed.mp4",
          original_filename: "licensed.mp4",
          rights_status: "LICENSED",
          lifecycle_state: "INGEST",
          created_at: "2026-09-04T00:00:00Z"
        }),
        { status: 201, headers: { "content-type": "application/json" } }
      );
    };

    const client = createApiClient("http://api.test", fetcher);
    await expect(client.upload(new File(["video"], "licensed.mp4"), "LICENSED")).resolves.toMatchObject({
      rights_status: "LICENSED"
    });
  });
});

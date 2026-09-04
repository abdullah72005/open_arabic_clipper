import { describe, expect, it } from "vitest";

import { ApiError, createApiClient } from "./api-client";

describe("API client", () => {
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
      expect.objectContaining<ApiError>({ status: 404, message: "not found" })
    );
  });
});

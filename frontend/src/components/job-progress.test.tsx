import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { JobProgress, formatElapsed } from "./job-progress";
import type { Job } from "@/lib/api-client";

const runningJob: Job = {
  id: "job-1",
  source_video_id: "s1",
  kind: "RECONSTRUCTION",
  status: "RUNNING",
  retry_count: 0,
  error_code: null,
  error_message: null,
  created_at: "2026-09-08T00:00:00Z",
  started_at: "2026-09-08T00:00:00Z",
  completed_at: null,
};

const queuedJob: Job = { ...runningJob, status: "QUEUED", started_at: null };

describe("formatElapsed", () => {
  it("labels an unstarted queued job", () => {
    expect(formatElapsed(null, Date.now())).toBe("waiting to start");
  });

  it("formats sub-minute elapsed time", () => {
    const started = new Date("2026-09-08T00:00:00Z").getTime();
    expect(formatElapsed(new Date(started).toISOString(), started + 45000)).toBe("45s");
  });

  it("formats minute elapsed time", () => {
    const started = new Date("2026-09-08T00:00:00Z").getTime();
    expect(formatElapsed(new Date(started).toISOString(), started + 125000)).toBe("2m 5s");
  });
});

describe("JobProgress", () => {
  it("shows what stage is running", () => {
    const markup = renderToStaticMarkup(<JobProgress job={runningJob} />);
    expect(markup).toContain("Reconstructing transcript");
    expect(markup).toContain("running");
    expect(markup).toContain("Elapsed");
  });

  it("calls out a queued job waiting for a worker", () => {
    const markup = renderToStaticMarkup(<JobProgress job={queuedJob} />);
    expect(markup).toContain("queued");
    expect(markup).toContain("Waiting for a worker");
    expect(markup).toContain("waiting to start");
  });
});
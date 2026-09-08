import { describe, expect, it } from "vitest";
import type { Job } from "@/lib/api-client";
import { nextJobTransition } from "./job-poller";

const running: Job = {
  id: "job-1",
  source_video_id: "s1",
  kind: "RECONSTRUCTION",
  status: "RUNNING",
  retry_count: 0,
  error_code: null,
  error_message: null,
};

const succeeded: Job = { ...running, status: "SUCCEEDED" };

describe("nextJobTransition", () => {
  it("does not fire on the first observation of a job", () => {
    const result = nextJobTransition({}, [running], "s1");
    expect(result.completed).toBe(false);
    expect(result.active).toBe(true);
  });

  it("stays active while the job keeps running", () => {
    const result = nextJobTransition({ "job-1": "RUNNING" }, [running], "s1");
    expect(result.completed).toBe(false);
    expect(result.active).toBe(true);
  });

  it("fires once when a running job succeeds", () => {
    const result = nextJobTransition({ "job-1": "RUNNING" }, [succeeded], "s1");
    expect(result.completed).toBe(true);
    expect(result.active).toBe(false);
  });

  it("fires when a queued job fails or is cancelled", () => {
    expect(nextJobTransition({ "job-1": "QUEUED" }, [{ ...running, status: "FAILED" }], "s1").completed).toBe(true);
    expect(nextJobTransition({ "job-1": "QUEUED" }, [{ ...running, status: "CANCELLED" }], "s1").completed).toBe(true);
  });

  it("ignores jobs belonging to other sources", () => {
    const result = nextJobTransition({ "job-1": "RUNNING" }, [succeeded], "s2");
    expect(result.completed).toBe(false);
    expect(result.active).toBe(false);
  });
});
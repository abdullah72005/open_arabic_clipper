"use client";
import { useEffect, useRef } from "react";
import { api, type Job, type JobStatus } from "@/lib/api-client";

export const ACTIVE_JOB_STATUSES = new Set<string>(["QUEUED", "RUNNING"]);

export interface JobTransition {
  completed: boolean;
  active: boolean;
  next: Record<string, string>;
}

export function nextJobTransition(
  previous: Record<string, string>,
  jobs: Job[],
  sourceId: string,
): JobTransition {
  const next: Record<string, string> = {};
  let completed = false;
  let active = false;
  for (const job of jobs) {
    if (job.source_video_id !== sourceId) continue;
    next[job.id] = job.status;
    const prior = previous[job.id];
    if (prior && ACTIVE_JOB_STATUSES.has(prior) && !ACTIVE_JOB_STATUSES.has(job.status)) {
      completed = true;
    }
    if (ACTIVE_JOB_STATUSES.has(job.status)) active = true;
  }
  return { completed, active, next };
}

export function JobPoller({
  sourceId,
  jobStatus,
  onComplete
}: {
  sourceId: string;
  jobStatus: JobStatus;
  onComplete: () => void;
}) {
  const callbackRef = useRef(onComplete);
  callbackRef.current = onComplete;

  useEffect(() => {
    if (!ACTIVE_JOB_STATUSES.has(jobStatus)) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let previous: Record<string, string> = {};
    const poll = async () => {
      try {
        const jobs = await api.listJobs();
        if (cancelled) return;
        const transition = nextJobTransition(previous, jobs, sourceId);
        previous = transition.next;
        if (transition.completed) callbackRef.current();
        if (transition.active) timer = setTimeout(poll, 2000);
      } catch {
        if (!cancelled) timer = setTimeout(poll, 2000);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [sourceId, jobStatus]);

  return null;
}
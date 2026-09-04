"use client";
import Link from "next/link";
import { ApiState } from "@/components/api-state";
import { api } from "@/lib/api-client";

export default function Dashboard() {
  return <ApiState load={() => Promise.all([api.listSources(), api.listJobs(), api.health()])}>{([sources, jobs, health]) => <><div className="grid"><section className="card"><p className="muted">Sources</p><h2>{sources.length}</h2></section><section className="card"><p className="muted">Active jobs</p><h2>{jobs.filter((job) => ["QUEUED", "RUNNING"].includes(job.status)).length}</h2></section><section className="card"><p className="muted">System health</p><h2 className={health.status === "HEALTHY" ? "ok" : "error"}>{health.status}</h2></section></div><section className="card" style={{ marginTop: 16 }}><h2>Start safely</h2><p className="muted">Add media you own or are authorized to process. Stage 1 only acquires and probes media; it does not transcribe, select clips, render, or publish.</p><Link className="button" href="/sources/add">Add a source</Link></section></>}</ApiState>;
}

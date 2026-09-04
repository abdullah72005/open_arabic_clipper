"use client";
import { useState } from "react";
import { ApiState } from "@/components/api-state";
import { api } from "@/lib/api-client";
export default function JobsPage() { const [message, setMessage] = useState(""); return <><h2>Jobs</h2>{message && <p className="error">{message}</p>}<ApiState load={api.listJobs}>{(jobs) => jobs.length ? <table><thead><tr><th>Kind</th><th>Status</th><th>Retries</th><th>Error</th><th /></tr></thead><tbody>{jobs.map((job) => <tr key={job.id}><td>{job.kind}</td><td>{job.status}</td><td>{job.retry_count}</td><td>{job.error_message ?? "—"}</td><td>{["QUEUED", "RUNNING"].includes(job.status) && <button className="button danger" onClick={() => api.cancelJob(job.id).catch((err: Error) => setMessage(err.message))}>Cancel</button>}</td></tr>)}</tbody></table> : <p className="muted">No jobs recorded.</p>}</ApiState></>; }

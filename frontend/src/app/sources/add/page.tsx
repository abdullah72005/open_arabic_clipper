"use client";
import { FormEvent, useState } from "react";
import { ApiError, api } from "@/lib/api-client";

export default function AddSourcePage() {
  const [urls, setUrls] = useState(""); const [file, setFile] = useState<File | null>(null); const [message, setMessage] = useState(""); const [busy, setBusy] = useState(false);
  const submit = async (event: FormEvent) => { event.preventDefault(); const entries = urls.split("\n").map((url) => url.trim()).filter(Boolean); if (!entries.length && !file) { setMessage("Enter one or more URLs, or select a file."); return; } setBusy(true); setMessage(""); try { if (entries.length) await api.submitUrls(entries); if (file) await api.upload(file); setUrls(""); setFile(null); setMessage("Submitted for ingest and probing."); } catch (error) { setMessage(error instanceof ApiError ? error.message : "Submission failed."); } finally { setBusy(false); } };
  return <><h2>Add source</h2><p className="muted">Only submit media you own or have permission to process. URL sources must be public and permitted; this app never bypasses platform protections.</p><form className="card" onSubmit={submit}><label>One URL per line<textarea rows={8} value={urls} onChange={(event) => setUrls(event.target.value)} placeholder="https://example.org/video" /></label><p className="muted">or</p><label>Local media file<input type="file" accept="video/*,audio/*" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /></label><p><button className="button" disabled={busy}>{busy ? "Submitting…" : "Queue source"}</button></p>{message && <p className={message.startsWith("Submitted") ? "ok" : "error"}>{message}</p>}</form></>;
}

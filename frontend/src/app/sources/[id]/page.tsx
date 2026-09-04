"use client";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import { ApiState } from "@/components/api-state";
import { api } from "@/lib/api-client";

export default function SourceDetail() { const params = useParams<{ id: string }>(); const router = useRouter(); const [error, setError] = useState(""); const load = useCallback(() => api.getSource(params.id), [params.id]); const remove = async () => { if (!confirm("Delete this source and its stored local files?")) return; try { await api.deleteSource(params.id); router.push("/sources"); } catch (err) { setError(err instanceof Error ? err.message : "Delete failed"); } }; return <ApiState load={load}>{(source) => <section className="card"><h2>{source.original_filename ?? "URL source"}</h2><dl><dt>Origin</dt><dd>{source.source_uri}</dd><dt>Rights</dt><dd>{source.rights_status}</dd><dt>Pipeline state</dt><dd>{source.lifecycle_state}</dd></dl><p className="muted">Future transcription and output workflows are unavailable in Stage 1.</p><button className="button danger" onClick={remove}>Delete source</button>{error && <p className="error">{error}</p>}</section>}</ApiState>; }

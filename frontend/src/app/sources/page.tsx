"use client";
import Link from "next/link";
import { ApiState } from "@/components/api-state";
import { api } from "@/lib/api-client";

export default function SourcesPage() { return <><h2>Sources</h2><ApiState load={api.listSources}>{(sources) => sources.length ? <table><thead><tr><th>Name / origin</th><th>Rights</th><th>State</th><th>Added</th></tr></thead><tbody>{sources.map((source) => <tr key={source.id}><td><Link href={`/sources/${source.id}`}>{source.original_filename ?? source.source_uri}</Link></td><td>{source.rights_status}</td><td>{source.lifecycle_state}</td><td>{new Date(source.created_at).toLocaleString()}</td></tr>)}</tbody></table> : <p className="muted">No sources yet. <Link href="/sources/add">Add one</Link>.</p>}</ApiState></>; }

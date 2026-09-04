"use client";
import { ApiState } from "@/components/api-state";
import { api } from "@/lib/api-client";
const bytes = (value: number) => `${(value / 1024 / 1024 / 1024).toFixed(2)} GB`;
export default function SettingsPage() { return <><h2>Local settings</h2><ApiState load={() => Promise.all([api.health(), api.storage()])}>{([health, storage]) => <div className="grid"><section className="card"><h3>Health: {health.status}</h3>{health.checks.map((check) => <p key={check.name}>{check.name}: {check.status}{check.detail ? ` — ${check.detail}` : ""}</p>)}</section><section className="card"><h3>Storage</h3><p>Total: {bytes(storage.total_bytes)}</p><p>Used: {bytes(storage.used_bytes)}</p><p>Free: {bytes(storage.free_bytes)}</p></section></div>}</ApiState><p className="muted">Configure service URLs, storage limits, and CORS via the deployment environment; see <code>.env.example</code>. Settings are intentionally not edited from the web UI.</p></>; }

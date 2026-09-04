import "./globals.css";
import Link from "next/link";

export const metadata = { title: "ClipFactory", description: "Local-first media ingest" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><main className="shell"><header><h1>ClipFactory</h1><p className="muted">Stage 1: local ingest and media probing</p></header><nav className="nav"><Link href="/">Dashboard</Link><Link href="/sources">Sources</Link><Link href="/sources/add">Add source</Link><Link href="/jobs">Jobs</Link><Link href="/settings">Settings</Link></nav>{children}</main></body></html>;
}

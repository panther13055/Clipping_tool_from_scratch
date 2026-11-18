import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "ClipForge Studio — Video Editor", description: "A creator-first video editing studio." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }

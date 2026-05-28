import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TensorQ-Engine | Quantum Circuit Simulator",
  description:
    "High-performance quantum circuit simulator with state-vector evolution and tensor-network contraction.",
};

export const viewport: Viewport = {
  themeColor: "#000000",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  readonly children: React.ReactNode;
}): JSX.Element {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-void font-mono antialiased">{children}</body>
    </html>
  );
}

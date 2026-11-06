import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Centralized Federated Learning Framework",
    template: "%s | CFL Framework",
  },
  description:
    "A cross-silo federated learning framework with conformance checking, fault tolerance and observability. COMP 491 capstone, Koç University.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}

import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

// Medium also covers 600, so `font-semibold` and `font-medium` both render as Medium. Anything
// heavier (the 650 of headings, `font-bold`) renders as Bold.
const stolzl = localFont({
  src: [
    { path: "../fonts/stolzl_book.otf", weight: "400", style: "normal" },
    { path: "../fonts/stolzl_medium.otf", weight: "500 600", style: "normal" },
    { path: "../fonts/stolzl_bold.otf", weight: "700", style: "normal" },
  ],
  variable: "--font-stolzl",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "FedUp: Centralized Federated Learning Framework",
    template: "%s | FedUp",
  },
  description:
    "A cross-silo federated learning framework with conformance checking, fault tolerance and observability. COMP 491 capstone by Team FedUp, Koç University.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${stolzl.variable} h-full`}>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}

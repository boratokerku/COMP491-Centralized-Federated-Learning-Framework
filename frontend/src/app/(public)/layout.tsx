import { SiteHeader } from "@/components/site-header";

export default function PublicLayout({ children }: LayoutProps<"/">) {
  return (
    <>
      <SiteHeader />
      <main className="wrap w-full flex-1 py-12">{children}</main>
      <footer className="border-t border-border">
        <div className="wrap flex flex-wrap items-center justify-between gap-2 py-6 text-sm text-muted">
          <span>COMP 491 capstone project by Team FedUp, Koç University.</span>
          <span>Cross-silo federated learning with conformance checking and observability.</span>
        </div>
      </footer>
    </>
  );
}

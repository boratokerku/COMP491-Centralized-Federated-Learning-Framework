import Link from "next/link";

export default function LoginLayout({ children }: LayoutProps<"/login">) {
  return (
    <main className="mx-auto flex w-full max-w-sm flex-1 flex-col justify-center px-6 py-16">
      <Link href="/" className="mb-8 text-sm text-muted hover:text-foreground">
        &larr; Back to the project site
      </Link>
      {children}
    </main>
  );
}

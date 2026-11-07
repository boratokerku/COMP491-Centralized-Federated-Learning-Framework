import type { Metadata } from "next";
import Image, { type StaticImageData } from "next/image";
import { Card, PageHeader } from "@/components/ui";
import { REPO_URL, advisors, team } from "@/content/site";
import barisAkgun from "../../../../team/baris_akgun.png";
import boraToker from "../../../../team/bora_toker.png";
import emreGursoy from "../../../../team/emre_gursoy.jpeg";
import enesTunc from "../../../../team/enes_tunc.png";
import hilalAygul from "../../../../team/hilal_aygul.png";
import merveKoybasi from "../../../../team/merve_koybasi.png";

export const metadata: Metadata = { title: "Team" };

const PHOTOS: Record<string, { src: StaticImageData; className?: string }> = {
  "Bora Toker": { src: boraToker, className: "object-cover object-top" },
  "Enes Tunç": { src: enesTunc, className: "object-cover object-top scale-105" },
  "Hilal Aygül": { src: hilalAygul, className: "object-cover object-[center_30%] scale-[1.22]" },
  "Merve Köybaşı": { src: merveKoybasi, className: "object-cover object-top scale-105" },
  "Prof. Mehmet Emre Gürsoy": { src: emreGursoy, className: "object-cover object-top" },
  "Prof. Barış Akgün": { src: barisAkgun, className: "object-cover object-top" },
};

const initials = (name: string) =>
  name
    .split(" ")
    .map((p) => p[0])
    .join("");

function Avatar({ name }: { name: string }) {
  const photo = PHOTOS[name];
  const dims = "h-36 w-36 sm:h-44 sm:w-44 lg:h-48 lg:w-48";
  if (!photo) {
    return (
      <span
        className={`grid ${dims} shrink-0 place-items-center rounded-full bg-accent text-2xl font-semibold text-background`}
      >
        {initials(name)}
      </span>
    );
  }
  return (
    <span
      className={`relative ${dims} shrink-0 overflow-hidden rounded-full border border-border bg-surface`}
    >
      <Image
        src={photo.src}
        alt={name}
        fill
        sizes="192px"
        className={photo.className ?? "object-cover"}
      />
    </span>
  );
}

export default function Team() {
  return (
    <div className="space-y-8">
      <PageHeader
        title="Team FedUp"
        description="We are FedUp, the team behind this COMP 491 capstone project at Koç University."
      >
        <a
          href={REPO_URL}
          className="rounded-lg border border-border px-4 py-2 text-sm hover:bg-surface"
        >
          Source code on GitHub &rarr;
        </a>
      </PageHeader>

      <Card className="space-y-8 p-8 sm:p-10">
        <div className="flex items-center gap-3">
          <span
            aria-hidden
            className="grid h-11 w-11 shrink-0 place-items-center rounded-lg border border-border bg-accent-soft text-accent"
          >
            <svg
              viewBox="0 0 24 24"
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.8}
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M22 10 12 5 2 10l10 5 10-5Z" />
              <path d="M6 12v5c0 1.7 2.7 3 6 3s6-1.3 6-3v-5" />
              <path d="M22 10v6" />
            </svg>
          </span>
          <h2 className="text-2xl font-semibold sm:text-3xl">Project Advisors</h2>
        </div>
        <ul className="grid gap-10 sm:grid-cols-2 lg:gap-x-16">
          {advisors.map((a) => (
            <li key={a} className="flex items-center gap-6 sm:gap-8">
              <Avatar name={a} />
              <div className="space-y-1.5">
                <span className="block text-xl font-semibold sm:text-2xl">{a}</span>
                <span className="block text-sm text-muted sm:text-base">
                  Computer Engineering · Koç University
                </span>
              </div>
            </li>
          ))}
        </ul>
      </Card>

      <Card className="space-y-8 p-8 sm:p-10">
        <div className="flex items-center gap-3">
          <span
            aria-hidden
            className="grid h-11 w-11 shrink-0 place-items-center rounded-lg border border-border bg-background text-mint"
          >
            <svg
              viewBox="0 0 24 24"
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.8}
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
              <circle cx="9" cy="7" r="4" />
              <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
              <path d="M16 3.13a4 4 0 0 1 0 7.75" />
            </svg>
          </span>
          <h2 className="text-2xl font-semibold sm:text-3xl">Team Members</h2>
        </div>
        <div className="grid gap-10 sm:grid-cols-2 lg:gap-x-16 lg:gap-y-12">
          {team.map((m) => (
            <div key={m.name} className="flex items-center gap-6 sm:gap-8">
              <Avatar name={m.name} />
              <div className="min-w-0 space-y-2">
                <h3 className="text-xl font-semibold sm:text-2xl">{m.name}</h3>
                <a
                  href={`mailto:${m.email}`}
                  className="inline-block text-base text-foreground/90 underline underline-offset-4 transition-colors hover:text-accent sm:text-lg"
                >
                  {m.email}
                </a>
                <p className="text-sm leading-relaxed text-muted sm:text-base">{m.role}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

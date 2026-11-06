"use client";

import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";

/** The marker of the open tab, a highlight or an underline. It slides to the new tab instead of
 * jumping there. */
export function NavPill({ underline = false }: { underline?: boolean }) {
  const pathname = usePathname();
  const pill = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const self = pill.current;
    const nav = self?.parentElement;
    if (!self || !nav) return;

    const place = () => {
      const tab = nav.querySelector<HTMLElement>('[aria-current="page"]');
      self.style.opacity = tab ? "1" : "0";
      if (!tab) return;
      self.style.width = `${tab.offsetWidth}px`;
      if (underline) {
        self.style.transform = `translateX(${tab.offsetLeft}px)`;
        return;
      }
      self.style.height = `${tab.offsetHeight}px`;
      self.style.transform = `translate(${tab.offsetLeft}px, ${tab.offsetTop}px)`;
      self.style.setProperty("--pill", getComputedStyle(tab).getPropertyValue("--hover"));
    };

    // On first load it appears in place; only later changes slide.
    if (!nav.hasAttribute("data-pill")) {
      self.style.transition = "none";
      place();
      void self.offsetWidth;
      self.style.transition = "";
      nav.setAttribute("data-pill", "");
    } else {
      place();
    }
    const sizes = new ResizeObserver(place);
    sizes.observe(nav);
    return () => sizes.disconnect();
  }, [pathname, underline]);

  return <span ref={pill} aria-hidden className={underline ? "nav-underline" : "nav-pill"} />;
}

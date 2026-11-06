"use client";

import { useEffect, useRef } from "react";

// A federated round as a backdrop: clients drift around and send updates to the coordinator, and
// every few updates a new global model spreads back out as a wave. The cursor links to nearby
// clients, and a click on an empty spot adds one.
const LINK_DISTANCE = 130;
const CURSOR_DISTANCE = 170;
const UPDATES_PER_ROUND = 6;
const MAX_CLIENTS = 40;
const WAVE_SPEED = 0.2; // px per ms

type Client = { x: number; y: number; vx: number; vy: number; r: number; color: string; glow: number };
type Update = { from: Client; t: number; speed: number };

function start(canvas: HTMLCanvasElement, moving: boolean) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return () => {};

  const css = getComputedStyle(canvas);
  const [danger, warning, blue, mint] = ["--danger", "--warning", "--accent", "--mint"].map((v) =>
    css.getPropertyValue(v).trim(),
  );
  const tones = [blue, danger, warning, mint];

  let w = 0;
  let h = 0;
  let hub = { x: 0, y: 0 };
  let clients: Client[] = [];
  let updates: Update[] = [];
  let received = 0;
  let wave: number | null = null;
  let hubGlow = 0;
  let nextUpdate = 300;
  let cursor: { x: number; y: number } | null = null;
  let frame = 0;
  let last = 0;

  const addClient = (x: number, y: number): Client => {
    const angle = Math.random() * Math.PI * 2;
    const speed = 0.006 + Math.random() * 0.012;
    const client = {
      x,
      y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      r: 2.5 + Math.random() * 2,
      color: tones[clients.length % tones.length],
      glow: 0,
    };
    clients.push(client);
    return client;
  };

  const resize = () => {
    const box = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio, 2);
    const first = w === 0;
    w = box.width;
    h = box.height;
    canvas.width = w * ratio;
    canvas.height = h * ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    hub = { x: w * (w >= 1024 ? 0.74 : 0.5), y: h * 0.45 };
    if (first) {
      const count = Math.round(Math.min(36, Math.max(14, (w * h) / 16000)));
      for (let i = 0; i < count; i++) addClient(Math.random() * w, Math.random() * h);
    } else {
      clients = clients.filter((c) => c.x <= w && c.y <= h);
      updates = updates.filter((u) => clients.includes(u.from));
    }
  };

  const line = (x1: number, y1: number, x2: number, y2: number, color: string, alpha: number) => {
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  };

  const dot = (x: number, y: number, r: number, color: string, alpha: number) => {
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  };

  const update = (dt: number) => {
    for (const c of clients) {
      c.x += c.vx * dt;
      c.y += c.vy * dt;
      if (c.x < 0 || c.x > w) c.vx *= -1;
      if (c.y < 0 || c.y > h) c.vy *= -1;
      c.glow = Math.max(0, c.glow - dt / 900);
      if (wave !== null && Math.abs(Math.hypot(c.x - hub.x, c.y - hub.y) - wave) < 18) c.glow = 1;
    }

    nextUpdate -= dt;
    if (nextUpdate <= 0 && clients.length > 0) {
      nextUpdate = 500 + Math.random() * 500;
      const from = clients[Math.floor(Math.random() * clients.length)];
      updates.push({ from, t: 0, speed: 0.0005 + Math.random() * 0.0003 });
    }
    updates = updates.filter((u) => {
      u.t += dt * u.speed;
      if (u.t < 1) return true;
      received += 1;
      hubGlow = Math.max(hubGlow, 0.5);
      return false;
    });
    if (received >= UPDATES_PER_ROUND && wave === null) {
      received = 0;
      wave = 0;
      hubGlow = 1;
    }
    if (wave !== null) {
      wave += dt * WAVE_SPEED;
      if (wave > Math.hypot(w, h)) wave = null;
    }
    hubGlow = Math.max(0, hubGlow - dt / 1200);
  };

  const draw = () => {
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 1;

    for (let i = 0; i < clients.length; i++) {
      const a = clients[i];
      line(a.x, a.y, hub.x, hub.y, a.color, 0.1);
      for (let j = i + 1; j < clients.length; j++) {
        const b = clients[j];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < LINK_DISTANCE) line(a.x, a.y, b.x, b.y, "#ffffff", 0.14 * (1 - d / LINK_DISTANCE));
      }
    }

    if (cursor) {
      for (const c of clients) {
        const d = Math.hypot(c.x - cursor.x, c.y - cursor.y);
        if (d < CURSOR_DISTANCE) {
          const near = 1 - d / CURSOR_DISTANCE;
          line(c.x, c.y, cursor.x, cursor.y, c.color, 0.6 * near);
          c.glow = Math.max(c.glow, near);
        }
      }
    }

    if (wave !== null) {
      ctx.globalAlpha = 0.35 * Math.max(0, 1 - wave / Math.max(w, h));
      ctx.strokeStyle = mint;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(hub.x, hub.y, wave, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 1;
    }

    for (const u of updates) {
      const x = u.from.x + (hub.x - u.from.x) * u.t;
      const y = u.from.y + (hub.y - u.from.y) * u.t;
      dot(x, y, 2.2, u.from.color, 0.9);
    }

    for (const c of clients) {
      if (c.glow > 0) dot(c.x, c.y, c.r * 3.5, c.color, 0.18 * c.glow);
      dot(c.x, c.y, c.r, c.color, 0.85 + 0.15 * c.glow);
    }

    dot(hub.x, hub.y, 16 + 14 * hubGlow, mint, 0.08 + 0.14 * hubGlow);
    dot(hub.x, hub.y, 6, mint, 0.9);
    ctx.globalAlpha = 1;
  };

  const tick = (now: number) => {
    update(Math.min(now - last, 50));
    last = now;
    draw();
    frame = requestAnimationFrame(tick);
  };
  const play = () => {
    cancelAnimationFrame(frame);
    last = performance.now();
    frame = requestAnimationFrame(tick);
  };

  resize();
  draw();

  const sizes = new ResizeObserver(() => {
    resize();
    draw();
  });
  sizes.observe(canvas);
  if (!moving) return () => sizes.disconnect();

  const inside = (e: PointerEvent) => {
    const box = canvas.getBoundingClientRect();
    const x = e.clientX - box.left;
    const y = e.clientY - box.top;
    return x >= 0 && y >= 0 && x <= w && y <= h ? { x, y } : null;
  };
  const onMove = (e: PointerEvent) => {
    cursor = e.pointerType === "mouse" ? inside(e) : null;
  };
  const onDown = (e: PointerEvent) => {
    const at = inside(e);
    if (!at || clients.length >= MAX_CLIENTS) return;
    if ((e.target as Element).closest("a, button, input, summary, h1, p")) return;
    const client = addClient(at.x, at.y);
    client.glow = 1;
    updates.push({ from: client, t: 0, speed: 0.0007 });
  };
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerdown", onDown);

  // Drawing stops once the backdrop has scrolled away.
  const seen = new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting) play();
    else cancelAnimationFrame(frame);
  });
  seen.observe(canvas);

  return () => {
    cancelAnimationFrame(frame);
    sizes.disconnect();
    seen.disconnect();
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerdown", onDown);
  };
}

/** Fills its section across the whole window, behind the content. */
export function FederatedBackground() {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvas.current) return;
    return start(canvas.current, !window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }, []);

  return (
    <div aria-hidden className="pointer-events-none absolute -top-16 -bottom-16 left-1/2 -z-10 w-screen -translate-x-1/2">
      <canvas ref={canvas} className="h-full w-full" />
      {/* Keeps the text readable and melts the edges into the page. */}
      <div className="absolute inset-0 bg-background/50 lg:bg-transparent lg:bg-linear-to-r lg:from-background/85 lg:via-background/45 lg:to-transparent" />
      <div className="absolute inset-x-0 bottom-0 h-24 bg-linear-to-b from-transparent to-background" />
    </div>
  );
}

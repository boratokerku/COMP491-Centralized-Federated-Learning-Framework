"use client";

import { useEffect, useRef } from "react";

// The landing screen's centrepiece: a slowly turning sphere of nodes around a glowing
// coordinator. A few larger nodes are clients: they send updates inward, and every few updates
// the coordinator sends a new global model back out as a wave that lights the nodes it passes.
// The sphere leans towards the cursor, and nodes near the cursor move aside and link to it.

const NODES = 120;
const CLIENTS = 8;
const LINK = 0.4; // neighbour distance on the unit sphere
const CURSOR = 150; // px
const UPDATES_PER_ROUND = 4;

type Node = {
  x: number;
  y: number;
  z: number;
  r: number;
  color: string;
  client: boolean;
  glow: number;
  push: number;
  // Projected, per frame.
  sx: number;
  sy: number;
  sz: number;
  scale: number;
};
type Packet = { from: Node; t: number; dur: number };

function rgba(hex: string, a: number) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? [...h].map((c) => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/** A soft round glow, drawn once per colour and stamped for every node. */
function sprite(hex: string) {
  const size = 64;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  const grd = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grd.addColorStop(0, rgba(hex, 1));
  grd.addColorStop(0.14, rgba(hex, 0.95));
  grd.addColorStop(0.32, rgba(hex, 0.28));
  grd.addColorStop(1, rgba(hex, 0));
  g.fillStyle = grd;
  g.fillRect(0, 0, size, size);
  return c;
}

function start(canvas: HTMLCanvasElement, moving: boolean) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return () => {};

  const css = getComputedStyle(canvas);
  const v = (name: string) => css.getPropertyValue(name).trim();
  const [danger, warning, accent, mint, lilac, fg] = ["--danger", "--warning", "--accent", "--mint", "--lilac", "--foreground"].map(v);
  const brand = [danger, warning, accent, lilac];
  const sprites = new Map([danger, warning, accent, mint, lilac, fg].map((c) => [c, sprite(c)]));

  // Points spread evenly over a sphere (a Fibonacci lattice), a little thick so it has depth.
  const golden = Math.PI * (3 - Math.sqrt(5));
  const clientEvery = Math.floor(NODES / CLIENTS);
  const nodes: Node[] = Array.from({ length: NODES }, (_, i) => {
    const y = 1 - (i / (NODES - 1)) * 2;
    const ring = Math.sqrt(1 - y * y);
    const shell = 0.88 + Math.random() * 0.17;
    const client = i % clientEvery === Math.floor(clientEvery / 2) && i / clientEvery < CLIENTS;
    const tint = Math.random();
    return {
      x: Math.cos(i * golden) * ring * shell,
      y: y * shell,
      z: Math.sin(i * golden) * ring * shell,
      r: client ? 6 : 1 + Math.random() * 1.3,
      color: client ? brand[Math.floor(i / clientEvery) % brand.length] : tint < 0.7 ? fg : brand[Math.floor(tint * 40) % brand.length],
      client,
      glow: 0,
      push: 0,
      sx: 0,
      sy: 0,
      sz: 0,
      scale: 1,
    };
  });
  const clients = nodes.filter((n) => n.client);
  const links: [Node, Node][] = [];
  for (let i = 0; i < NODES; i++)
    for (let j = i + 1; j < NODES; j++) {
      const a = nodes[i];
      const b = nodes[j];
      if (Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) < LINK) links.push([a, b]);
    }

  let w = 0;
  let h = 0;
  let R = 0;
  let cx = 0;
  let cy = 0;
  let spin = 0;
  let yaw = 0;
  let pitch = 0;
  let aimYaw = 0;
  let aimPitch = 0;
  let cursor: { x: number; y: number } | null = null;
  let packets: Packet[] = [];
  let waves: number[] = [];
  let arrivals = 0;
  let pulse = 0;
  let nextPacket = 400;
  let frame = 0;
  let last = 0;

  const resize = () => {
    const box = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio, 2);
    w = box.width;
    h = box.height;
    canvas.width = w * ratio;
    canvas.height = h * ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    cx = w / 2;
    cy = h / 2;
    R = Math.min(w, h) * 0.34;
    if (!moving) draw(0);
  };

  const project = () => {
    const cy_ = Math.cos(spin + yaw);
    const sy_ = Math.sin(spin + yaw);
    const cx_ = Math.cos(0.32 + pitch);
    const sx_ = Math.sin(0.32 + pitch);
    const fov = R * 2.8;
    for (const n of nodes) {
      const x1 = n.x * cy_ - n.z * sy_;
      const z1 = n.x * sy_ + n.z * cy_;
      const y2 = n.y * cx_ - z1 * sx_;
      const z2 = n.y * sx_ + z1 * cx_;
      const scale = fov / (fov + z2 * R);
      let px = cx + x1 * R * scale;
      let py = cy + y2 * R * scale;
      // Move aside for the cursor.
      if (cursor) {
        const dx = px - cursor.x;
        const dy = py - cursor.y;
        const d = Math.hypot(dx, dy) || 1;
        const want = d < CURSOR ? 1 - d / CURSOR : 0;
        n.push += (want - n.push) * 0.12;
        px += (dx / d) * n.push * 22;
        py += (dy / d) * n.push * 22;
      } else {
        n.push *= 0.9;
      }
      n.sx = px;
      n.sy = py;
      n.sz = z2;
      n.scale = scale;
    }
  };

  // 1 at the front of the sphere, 0 at the back.
  const near = (n: Node) => (1 - n.sz) / 2;

  const stamp = (color: string, x: number, y: number, radius: number, alpha: number) => {
    const s = sprites.get(color);
    if (!s || alpha <= 0) return;
    ctx.globalAlpha = Math.min(1, alpha);
    ctx.drawImage(s, x - radius, y - radius, radius * 2, radius * 2);
  };

  const draw = (dt: number) => {
    if (moving) {
      spin += dt * 0.00011;
      yaw += (aimYaw - yaw) * 0.04;
      pitch += (aimPitch - pitch) * 0.04;

      nextPacket -= dt;
      if (nextPacket <= 0) {
        const from = clients[Math.floor(Math.random() * clients.length)];
        packets.push({ from, t: 0, dur: 1300 + Math.random() * 600 });
        from.glow = 1;
        nextPacket = 380 + Math.random() * 380;
      }
      packets = packets.filter((p) => {
        p.t += dt / p.dur;
        if (p.t < 1) return true;
        pulse = 1;
        arrivals += 1;
        if (arrivals % UPDATES_PER_ROUND === 0) waves.push(0);
        return false;
      });
      waves = waves.map((r) => r + dt * 0.24).filter((r) => r < R * 1.5);
      pulse = Math.max(0, pulse - dt * 0.0025);
      for (const n of nodes) n.glow = Math.max(0, n.glow - dt * 0.0012);
    }

    project();
    for (const n of nodes) {
      for (const r of waves) {
        const lit = 1 - Math.abs(Math.hypot(n.sx - cx, n.sy - cy) - r) / 28;
        if (lit > n.glow) n.glow = lit;
      }
    }

    ctx.clearRect(0, 0, w, h);
    ctx.globalCompositeOperation = "lighter";

    // The mesh, fainter at the back.
    ctx.lineWidth = 1;
    for (const [a, b] of links) {
      const depth = (near(a) + near(b)) / 2;
      const alpha = 0.03 + depth * depth * 0.16 + (a.glow + b.glow) * 0.1 + (a.push + b.push) * 0.25;
      ctx.globalAlpha = Math.min(0.6, alpha);
      ctx.strokeStyle = fg;
      ctx.beginPath();
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.stroke();
    }

    // Spokes from the clients to the coordinator.
    for (const c of clients) {
      const g = ctx.createLinearGradient(c.sx, c.sy, cx, cy);
      g.addColorStop(0, rgba(c.color, 0.9));
      g.addColorStop(1, rgba(mint, 0.3));
      ctx.globalAlpha = 0.3 + near(c) * 0.45 + c.glow * 0.4;
      ctx.strokeStyle = g;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(c.sx, c.sy);
      ctx.lineTo(cx, cy);
      ctx.stroke();
    }

    // The cursor joins in.
    if (cursor) {
      for (const n of nodes) {
        if (n.push < 0.05) continue;
        ctx.globalAlpha = n.push * 0.45;
        ctx.strokeStyle = accent;
        ctx.beginPath();
        ctx.moveTo(cursor.x, cursor.y);
        ctx.lineTo(n.sx, n.sy);
        ctx.stroke();
      }
    }

    // Updates on their way in, each with a short tail.
    for (const p of packets) {
      for (let k = 4; k >= 0; k--) {
        const t = Math.max(0, p.t - k * 0.025);
        const e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
        const x = p.from.sx + (cx - p.from.sx) * e;
        const y = p.from.sy + (cy - p.from.sy) * e;
        stamp(p.from.color, x, y, k === 0 ? 13 : 9 - k, k === 0 ? 1 : 0.5 - k * 0.1);
      }
    }

    const nodeAt = (n: Node) => {
      const depth = near(n);
      const size = n.r * n.scale * (n.client ? 3.6 : 3.2) * (1 + n.glow * 0.5 + n.push * 0.6);
      stamp(n.color, n.sx, n.sy, size, 0.18 + depth * 0.82 + n.glow * 0.6);
    };
    const sorted = [...nodes].sort((a, b) => b.sz - a.sz);
    for (const n of sorted) if (n.sz > 0) nodeAt(n);

    // The coordinator: a halo, waves for each new global model, and a bright core.
    stamp(mint, cx, cy, R * (0.6 + pulse * 0.12), 0.45 + pulse * 0.3);
    for (const r of waves) {
      ctx.globalAlpha = Math.max(0, 0.7 * (1 - r / (R * 1.5)));
      ctx.strokeStyle = mint;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.globalCompositeOperation = "source-over";
    const core = 16 + pulse * 4;
    const g = ctx.createRadialGradient(cx - core * 0.3, cy - core * 0.35, 0, cx, cy, core);
    g.addColorStop(0, "#ffffff");
    g.addColorStop(0.45, mint);
    g.addColorStop(1, rgba(mint, 0.6));
    ctx.globalAlpha = 1;
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, core, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = "lighter";

    for (const n of sorted) if (n.sz <= 0) nodeAt(n);
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
  };

  const loop = (now: number) => {
    const dt = last ? Math.min(now - last, 50) : 16;
    last = now;
    draw(dt);
    frame = requestAnimationFrame(loop);
  };
  const play = () => {
    if (!moving) return;
    cancelAnimationFrame(frame);
    last = 0;
    frame = requestAnimationFrame(loop);
  };

  const sizes = new ResizeObserver(resize);
  sizes.observe(canvas);
  resize();

  const onMove = (e: PointerEvent) => {
    const box = canvas.getBoundingClientRect();
    const x = e.clientX - box.left;
    const y = e.clientY - box.top;
    // The whole window steers the lean; only the canvas itself has the cursor links.
    aimYaw = Math.max(-1, Math.min(1, (x - box.width / 2) / (window.innerWidth / 2))) * 0.55;
    aimPitch = Math.max(-1, Math.min(1, (y - box.height / 2) / (window.innerHeight / 2))) * 0.35;
    cursor = e.pointerType === "mouse" && x >= 0 && y >= 0 && x <= box.width && y <= box.height ? { x, y } : null;
    if (!moving) draw(0);
  };
  const onLeave = () => {
    cursor = null;
  };
  window.addEventListener("pointermove", onMove);
  document.documentElement.addEventListener("pointerleave", onLeave);

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
    document.documentElement.removeEventListener("pointerleave", onLeave);
  };
}

export function HeroNetwork() {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvas.current) return;
    return start(canvas.current, !window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }, []);

  return (
    <div aria-hidden className="relative mx-auto aspect-square w-full max-w-[min(760px,86svh)]">
      <canvas ref={canvas} className="absolute -inset-[8%] h-[116%] w-[116%]" />
    </div>
  );
}

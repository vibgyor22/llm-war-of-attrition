const $ = s => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent && parent.appendChild(e); return e; };
const COND = ["A", "B", "C", "D", "E"];
const ACT = { HOLD: "", SIGNAL_FLEXIBILITY: "sig", CONCEDE: "con" };
const ACTNAME = { HOLD: "Hold", SIGNAL_FLEXIBILITY: "Signal flexibility", CONCEDE: "Concede" };
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
let S;

/* ---------- tooltips ---------- */
function tip(fig) { let t = fig.querySelector(".tip"); if (!t) { t = document.createElement("div"); t.className = "tip"; fig.appendChild(t); } return t; }
function showTip(fig, html, x, y) { const t = tip(fig); t.innerHTML = html; t.style.opacity = 1; const w = fig.clientWidth; t.style.left = Math.min(Math.max(0, x - t.offsetWidth / 2), w - t.offsetWidth) + "px"; t.style.top = (y - t.offsetHeight - 12) + "px"; }
const hideTip = fig => tip(fig).style.opacity = 0;
const hover = (node, fig, html, x, y) => { node.style.cursor = "pointer"; node.addEventListener("mouseenter", () => showTip(fig, html, x, y)); node.addEventListener("mouseleave", () => hideTip(fig)); };

/* ---------- scroll reveal, count-up, parallax ---------- */
function countUp(n) {
  const to = +n.dataset.to, t0 = performance.now(), D = reduce ? 0 : 1400;
  const step = now => { const k = D ? Math.min(1, (now - t0) / D) : 1; n.textContent = Math.round(to * (1 - Math.pow(1 - k, 3))); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
function drawPaths(root) { root.querySelectorAll(".draw").forEach(p => p.style.strokeDashoffset = 0); }
const io = new IntersectionObserver(es => es.forEach(e => {
  if (!e.isIntersecting) return; const t = e.target; t.classList.add("in"); io.unobserve(t);
  t.querySelectorAll?.(".count").forEach(countUp); if (t.classList.contains("count")) countUp(t);
  if (t.classList.contains("dots")) t.classList.add("in");
  drawPaths(t);
}), { threshold: .25, rootMargin: "0px 0px -8% 0px" });
const watch = () => document.querySelectorAll(".reveal:not(.in),.stagger:not(.in),.figure:not(.in),.dots:not(.in),.count").forEach(n => io.observe(n));
function prepDraw(path) { const L = path.getTotalLength(); path.style.strokeDasharray = L; path.style.strokeDashoffset = path.closest(".figure.in") ? 0 : L; }
function parallax() {
  if (reduce) return;
  const items = [...document.querySelectorAll(".px")];
  let ticking = false;
  addEventListener("scroll", () => { if (ticking) return; ticking = true; requestAnimationFrame(() => { const y = scrollY; items.forEach(n => n.style.transform = `translate3d(0,${y * +n.dataset.speed}px,0)`); ticking = false; }); }, { passive: true });
}

/* ---------- dot grids ---------- */
function dotGrid(box, runs) {
  box.innerHTML = ""; [...runs].sort((a, b) => (a.conceder === "D") - (b.conceder === "D")).forEach((r, i) => {
    const s = document.createElement("span"); s.className = r.conceder === "R" ? "r" : "d"; s.style.transitionDelay = (i * 14) + "ms"; box.appendChild(s);
  });
}
function hero() {
  dotGrid($("#dots"), S.runs); $("#bigN").textContent = S.n; $("#bigR").dataset.to = S.rConcede;
  dotGrid($("#dotsH"), S.runs.filter(r => !r.masked)); dotGrid($("#dotsM"), S.runs.filter(r => r.masked));
  $("#cH").dataset.to = S.masked.historical; $("#cM").dataset.to = S.masked.masked;
}

/* ---------- 02 strip chart ---------- */
function strip() {
  const fig = $("#strip"); fig.querySelector("svg")?.remove();
  const W = fig.clientWidth, small = W < 600, L = small ? 78 : 140, R = 12, rowH = small ? 64 : 76, H = rowH * 5 + 40;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, height: H }, fig);
  const x = d => L + (d / 30) * (W - L - R);
  [0, 10, 20, 30].forEach(d => { el("line", { x1: x(d), x2: x(d), y1: 0, y2: rowH * 5, class: "tick" }, svg); el("text", { x: x(d), y: rowH * 5 + 22, "text-anchor": d === 30 ? "end" : "middle" }, svg).textContent = d === 30 ? "X-date" : "Day " + d; });
  let k = 0;
  COND.forEach((c, i) => {
    const cy = i * rowH + rowH / 2;
    el("line", { x1: L, x2: W - R, y1: cy, y2: cy, class: "axis" }, svg);
    el("text", { x: 0, y: cy - 2, class: "lbl" }, svg).textContent = S.conditions[c];
    el("text", { x: 0, y: cy + 15 }, svg).textContent = "stress " + c;
    const seen = {};
    S.runs.filter(r => r.cond === c).sort((a, b) => a.day - b.day).forEach(r => {
      const n = seen[r.day] = (seen[r.day] || 0) + 1, off = (n % 2 ? 1 : -1) * Math.floor(n / 2) * 9;
      const dot = el("circle", { cx: x(r.day), cy: cy + off, r: small ? 4 : 5, fill: r.conceder === "R" ? css("--rep") : css("--dem"), "fill-opacity": .85, class: "slide", style: `transition-delay:${(i * 120 + k++ * 8)}ms` }, svg);
      hover(dot, fig, `${S.episodes[r.ep]}${r.masked ? " · masked" : ""}<br>${r.conceder === "R" ? "Republican" : "Democrat"} conceded on day ${r.day + 1}`, x(r.day), cy + off);
      dot.addEventListener("click", () => { pick.ep = r.ep; pick.cond = r.cond; pick.run = r.id; loadRun(true); });
    });
    const m = S.byCondition[c].medianDay;
    el("line", { x1: x(m), x2: x(m), y1: cy - 24, y2: cy + 24, stroke: css("--fg"), "stroke-width": 2, class: "fade" }, svg);
  });
  const a = S.byCondition.A, e = S.byCondition.E;
  $("#stats").innerHTML = `<div><b>Day <span class="count" data-to="${a.medianDay + 1}">0</span></b><span>typical deal in calm markets</span></div>
    <div><b>Day <span class="count" data-to="${e.medianDay + 1}">0</span></b><span>typical deal under extreme stress</span></div>
    <div><b><span class="count" data-to="${Math.round(100 * S.rConcede / S.n)}">0</span>%</b><span>of simulations, the Republican agent conceded</span></div>`;
}

/* ---------- 03 survival curves ---------- */
function survival() {
  const fig = $("#survival"); fig.querySelector("svg")?.remove();
  const W = fig.clientWidth, H = Math.max(260, Math.min(440, W * .6)), L = 44, B = 36, T = 10, R = 12;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, height: H }, fig);
  const x = d => L + d / 30 * (W - L - R), y = v => T + (1 - v) * (H - T - B);
  [0, .5, 1].forEach(v => { el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "tick" }, svg); el("text", { x: L - 10, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = v * 100 + "%"; });
  [0, 10, 20, 30].forEach(d => el("text", { x: x(d), y: H - 12, "text-anchor": "middle" }, svg).textContent = d === 30 ? "X-date" : "Day " + d);
  const leg = [];
  COND.forEach((c, i) => {
    const s = S.survival[c]; let d = `M${x(0)},${y(s[0])}`;
    for (let t = 1; t <= 30; t++) d += `H${x(t)}V${y(s[t])}`;
    const op = .25 + i * .18, sw = i === 4 || i === 0 ? 2.2 : 1.4;
    const p = el("path", { d, fill: "none", stroke: css("--fg"), "stroke-opacity": op, "stroke-width": sw, class: "draw", style: `transition-delay:${i * 180}ms` }, svg);
    prepDraw(p);
    const half = s.findIndex(v => v <= .5);
    leg.push(`<span><b style="background:var(--fg);opacity:${op};border-radius:0;height:2px;width:18px"></b>${S.conditions[c]} · half settled by day ${half + 1}</span>`);
    for (let t2 = 0; t2 < 30; t2++) { const hit = el("rect", { x: x(t2), y: y(s[t2]) - 6, width: x(1) - x(0), height: 12, fill: "transparent" }, svg); hover(hit, fig, `${S.conditions[c]} · day ${t2 + 1}: ${Math.round(s[t2] * 100)}% still deadlocked`, x(t2), y(s[t2])); }
  });
  $("#survLeg").innerHTML = leg.join("");
}

/* ---------- 04 pain over time + scatter ---------- */
function painline() {
  const fig = $("#painline"); fig.querySelector("svg")?.remove();
  const W = fig.clientWidth, H = Math.min(340, W * .7), L = 30, B = 36, T = 16, R = 10;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, height: H }, fig);
  const x = d => L + d / 29 * (W - L - R), y = v => T + (1 - (v - 5) / 5) * (H - T - B);
  [5, 7.5, 10].forEach(v => { el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "tick" }, svg); el("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = v; });
  [0, 29].forEach(d => el("text", { x: x(d), y: H - 12, "text-anchor": d ? "end" : "start" }, svg).textContent = d ? "X-date" : "Day 1");
  const P = S.pain, idx = P.R.map((v, i) => i).filter(i => P.R[i] != null);
  const line = k => idx.map((i, j) => `${j ? "L" : "M"}${x(i)},${y(P[k][i])}`).join("");
  el("path", { d: line("R") + idx.slice().reverse().map(i => `L${x(i)},${y(P.D[i])}`).join("") + "Z", fill: css("--rep"), "fill-opacity": .08, class: "fade" }, svg);
  [["R", "--rep"], ["D", "--dem"]].forEach(([k, c], i) => { const p = el("path", { d: line(k), fill: "none", stroke: css(c), "stroke-width": 2, class: "draw", style: `transition-delay:${i * 200}ms` }, svg); prepDraw(p); });
  const last = idx.at(-1);
  el("text", { x: x(last) - 12, y: y(P.R[last]) - 8, "text-anchor": "end", fill: css("--rep"), class: "fade" }, svg).textContent = "Republican " + P.R[last].toFixed(1);
  el("text", { x: x(last) - 22, y: y(P.D[last]) + 26, "text-anchor": "end", fill: css("--dem"), class: "fade" }, svg).textContent = "Democrat " + P.D[last].toFixed(1);
  idx.forEach(i => { const hit = el("rect", { x: x(i) - 4, y: T, width: 8, height: H - T - B, fill: "transparent" }, svg); hover(hit, fig, `Day ${i + 1} · R ${P.R[i].toFixed(1)} · D ${P.D[i].toFixed(1)} · ${P.n[i]} runs active`, x(i), y(P.R[i])); });
}
function scatter() {
  const fig = $("#scatter"); fig.querySelector("svg")?.remove();
  const W = fig.clientWidth, H = Math.min(340, W * .7), L = 30, B = 36, T = 16, R = 10;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, height: H }, fig);
  const xs = S.runs.map(r => r.ratio), x0 = Math.min(...xs) - .02, x1 = Math.max(...xs) + .02;
  const x = v => L + (v - x0) / (x1 - x0) * (W - L - R), y = d => T + (1 - d / 30) * (H - T - B);
  [0, 15, 30].forEach(d => { el("line", { x1: L, x2: W - R, y1: y(d), y2: y(d), class: "tick" }, svg); el("text", { x: L - 8, y: y(d) + 4, "text-anchor": "end" }, svg).textContent = d; });
  el("line", { x1: x(1), x2: x(1), y1: T, y2: H - B, stroke: css("--faint"), "stroke-dasharray": "3 3" }, svg);
  el("text", { x: x(1) + 6, y: T + 4 }, svg).textContent = "equal pain";
  el("text", { x: L, y: H - 12 }, svg).textContent = "← Dem. hurts more";
  el("text", { x: W - R, y: H - 12, "text-anchor": "end" }, svg).textContent = "Rep. hurts more →";
  S.runs.forEach((r, i) => {
    const c = el("circle", { cx: x(r.ratio), cy: y(r.day), r: 4.5, fill: r.conceder === "R" ? css("--rep") : css("--dem"), "fill-opacity": .8, class: "pop", style: `transition-delay:${i * 12}ms` }, svg);
    hover(c, fig, `Pain ratio ${r.ratio.toFixed(2)} · deal on day ${r.day + 1}`, x(r.ratio), y(r.day));
  });
  const g = S.regression.pooled;
  const p = el("path", { d: `M${x(x0)},${y(g.const + g.beta * x0)}L${x(x1)},${y(g.const + g.beta * x1)}`, stroke: css("--fg"), "stroke-width": 1.5, class: "draw", style: "transition-delay:1s" }, svg); prepDraw(p);
  const h = S.regression.historical;
  $("#regnote").innerHTML = `Left: average felt cost across simulations still negotiating that day. Right: each dot is one simulation; horizontal is Republican ÷ Democrat average felt cost. Each 0.1 rise in that ratio brings the deal about <b>${Math.abs(h.beta / 10).toFixed(1)} days</b> earlier (historical runs, n = ${h.n}, p = ${h.p.toFixed(3)}). <a href="method.html#results">Full regression →</a>`;
  $("#stats2").innerHTML = `<div><b><span class="count" data-to="${S.higherPain}">0</span>/${S.n}</b><span>simulations where the Republican agent reported more pain on average</span></div>
    <div><b>${S.pain.R[0].toFixed(1)} → ${S.pain.R.filter(v => v != null).at(-1).toFixed(1)}</b><span>Republican felt cost, first day to last</span></div>
    <div><b>${Math.abs(h.beta / 10).toFixed(1)} days</b><span>earlier deal per +0.1 in the pain ratio</span></div>`;
}

/* ---------- 06 theater ---------- */
const pick = { ep: "2023", cond: "C", run: null };
let RUN = null, DAY = -1, playing = null;
function pills(box, items, cur, on) {
  box.innerHTML = ""; items.forEach(([v, l]) => { const b = document.createElement("button"); b.textContent = l; b.className = v === cur ? "on" : ""; b.onclick = () => on(v); box.appendChild(b); });
}
function controls() {
  pills($("#pEp"), Object.entries(S.episodes).map(([k, v]) => [k, v.split(" · ")[0]]), pick.ep, v => { pick.ep = v; pick.run = null; loadRun(true); });
  pills($("#pCond"), COND.map(c => [c, S.conditions[c]]), pick.cond, v => { pick.cond = v; pick.run = null; loadRun(true); });
  const opts = S.runs.filter(r => r.ep === pick.ep && r.cond === pick.cond).sort((a, b) => a.masked - b.masked || a.rep - b.rep);
  if (!opts.find(r => r.id === pick.run)) pick.run = opts[0].id;
  pills($("#pRun"), opts.map(r => [r.id, (r.masked ? "Masked " : "Run ") + (r.rep + 1)]), pick.run, v => { pick.run = v; loadRun(true); });
}
const stepPx = () => Math.max(240, innerHeight * .45);
function sizeTheater() { if (RUN) $("#theater").style.height = (RUN.periods.length * stepPx() + innerHeight) + "px"; }
const theaterTop = () => $("#theater").getBoundingClientRect().top + scrollY - 56;
async function loadRun(jump) {
  stopPlay(); controls();
  RUN = await (await fetch(`data/runs/${pick.run}.json`)).json();
  sizeTheater(); buildArena(); DAY = -1;
  const who = RUN.conceder === "R" ? "Republican" : "Democrat";
  $("#stMeta").textContent = `${S.episodes[RUN.ep]} · ${S.conditions[RUN.cond]} markets${RUN.masked ? " · masked" : ""}`;
  $("#stVerdict").innerHTML = `<span class="${RUN.conceder === "R" ? "rep" : "dem"}">${who} concedes · day ${RUN.day + 1}</span>`;
  ["R", "D"].forEach(side => {
    const lane = $("#lane" + side); lane.innerHTML = "";
    for (let t = 0; t < 30; t++) {
      const i = RUN.periods.findIndex(q => q.t === t), p = RUN.periods[i], b = document.createElement("button");
      b.className = "cell " + (p ? ACT[p[side].a] : "void"); b.dataset.t = t;
      if (p) { b.title = `Day ${t + 1}: ${ACTNAME[p[side].a]}`; b.onclick = () => goDay(i); }
      lane.appendChild(b);
    }
  });
  if (jump) scrollTo({ top: theaterTop() + 2, behavior: reduce ? "auto" : "smooth" });
  onScroll(true);
}
function goDay(i, smooth = true) { scrollTo({ top: theaterTop() + i * stepPx() + 2, behavior: smooth && !reduce ? "smooth" : "auto" }); }
function onScroll(force) {
  if (!RUN) return;
  const p = (scrollY - theaterTop()) / stepPx();
  const i = Math.max(0, Math.min(RUN.periods.length - 1, Math.floor(p)));
  if (i !== DAY || force === true) { DAY = i; render(i); }
}
let A = {};
function buildArena() {
  const svg = $("#arena"); svg.innerHTML = "";
  const W = svg.clientWidth, H = svg.clientHeight; svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const cy = H * .46, terrainH = Math.min(70, H * .28);
  // market stress terrain along the bottom, one point per day
  const tx = t => 8 + t / 29 * (W - 16), ty = v => H - 4 - v * terrainH;
  const vs = RUN.periods.map(p => p.vix), lo = Math.min(...vs), hi = Math.max(...vs), nv = v => hi > lo ? .15 + .85 * (v - lo) / (hi - lo) : .5;
  A.nv = nv;
  const pts = RUN.periods.map(p => `${tx(p.t)},${ty(nv(p.vix))}`).join(" ");
  const first = tx(RUN.periods[0].t), lastX = tx(RUN.periods.at(-1).t);
  el("polygon", { points: `${first},${H - 4} ${pts} ${lastX},${H - 4}`, fill: css("--soft") }, svg);
  const clip = el("clipPath", { id: "past" }, el("defs", {}, svg)); A.clipR = el("rect", { x: 0, y: 0, width: 0, height: H, style: "transition:width .8s" }, clip);
  el("polygon", { points: `${first},${H - 4} ${pts} ${lastX},${H - 4}`, fill: css("--fg"), "fill-opacity": .12, "clip-path": "url(#past)" }, svg);
  el("polyline", { points: pts, fill: "none", stroke: css("--faint"), "stroke-width": 1 }, svg);
  A.mark = el("circle", { r: 3.5, fill: css("--fg"), style: "transition:all .8s" }, svg);
  el("text", { x: W - 8, y: H - 12, "text-anchor": "end" }, svg).textContent = `VIX · ${lo.toFixed(0)} → ${hi.toFixed(0)}`;
  // axis + center
  el("line", { x1: 0, x2: W, y1: cy, y2: cy, stroke: css("--line") }, svg);
  el("line", { x1: W / 2, x2: W / 2, y1: cy - 8, y2: cy + 8, stroke: css("--faint") }, svg);
  el("text", { x: W / 2, y: cy - 16, "text-anchor": "middle" }, svg).textContent = "DEAL";
  A.tether = el("line", { y1: cy, y2: cy, stroke: css("--fg"), "stroke-width": 1, class: "tether" }, svg);
  const orb = (col, label) => {
    const g = el("g", { class: "orb" }, svg);
    const halo = el("circle", { cx: 0, cy, r: 30, fill: "none", stroke: css(col), "stroke-width": 1.2, class: "halo" }, g);
    const core = el("circle", { cx: 0, cy, r: 30, fill: css(col), class: "core" }, g);
    const t = el("text", { x: 0, y: cy + 0, "text-anchor": "middle" }, g); t.textContent = label;
    return { g, halo, core, t };
  };
  A.R = orb("--rep", "REPUBLICAN"); A.D = orb("--dem", "DEMOCRAT");
  Object.assign(A, { W, H, cy, tx });
}
function render(i) {
  const p = RUN.periods[i], W = A.W, small = W < 600, max = Math.min(A.H * .3, small ? 34 : 54);
  const isLast = i === RUN.periods.length - 1;
  $("#stDay").textContent = "Day " + (p.t + 1); $("#stLeft").textContent = `${p.x} DAYS TO X-DATE · ${p.date}`;
  const base = W * (small ? .33 : .36);
  const pos = {};
  ["R", "D"].forEach(s => {
    const d = p[s], o = A[s], r = 8 + (d.c / 10) * (max - 8);
    let gap = base * (1 - .7 * Math.min(1, d.p + (d.a === "SIGNAL_FLEXIBILITY" ? .15 : 0)));
    if (isLast && RUN.conceder) gap = RUN.conceder === s ? r * .2 : r * 1.1;
    const x = s === "R" ? W / 2 - gap : W / 2 + gap; pos[s] = { x, r };
    o.g.style.transform = `translateX(${x}px)`;
    o.core.setAttribute("r", r); o.halo.setAttribute("r", r);
    o.core.style.opacity = d.a === "CONCEDE" ? .25 : 1;
    o.halo.style.setProperty("--sp", (reduce ? 999 : Math.max(.9, 4.4 - d.c * .38)) + "s");
    o.t.setAttribute("y", A.cy + r + 22); o.t.textContent = (s === "R" ? "REP " : "DEM ") + d.c.toFixed(1);
    o.t.style.opacity = isLast && RUN.conceder === s ? 0 : 1;
  });
  const soft = p.R.a !== "HOLD" || p.D.a !== "HOLD";
  Object.entries({ x1: pos.R.x + pos.R.r, x2: Math.max(pos.R.x + pos.R.r, pos.D.x - pos.D.r) }).forEach(([k, v]) => A.tether.setAttribute(k, v));
  A.tether.setAttribute("stroke-dasharray", soft ? "none" : "2 6"); A.tether.setAttribute("stroke-opacity", soft ? .6 : .35);
  A.clipR.setAttribute("width", A.tx(p.t)); A.mark.setAttribute("cx", A.tx(p.t)); A.mark.setAttribute("cy", A.H - 4 - A.nv(p.vix) * Math.min(70, A.H * .28));
  ["R", "D"].forEach(s => {
    const box = $("#say" + s), d = p[s];
    box.classList.add("swap");
    setTimeout(() => {
      box.innerHTML = `<div class="sayhead"><span class="act">${ACTNAME[d.a]}</span><details><summary>Private reasoning</summary><p>${d.r || ""}</p></details></div><q>${d.s || "…"}</q>`;
      box.classList.remove("swap");
    }, reduce ? 0 : 220);
  });
  document.querySelectorAll(".timeline .cell").forEach(c => c.classList.toggle("col-on", +c.dataset.t === p.t));
}
function stopPlay() { clearInterval(playing); playing = null; $("#play").textContent = "Play"; }
function togglePlay() {
  if (playing) return stopPlay();
  if (DAY >= RUN.periods.length - 1) goDay(0, false);
  $("#play").textContent = "Pause";
  goDay(Math.max(0, DAY), true);
  playing = setInterval(() => { if (DAY >= RUN.periods.length - 1) return stopPlay(); goDay(DAY + 1); }, 2600);
}

/* ---------- boot ---------- */
function charts() { strip(); survival(); painline(); scatter(); }
fetch("data/summary.json").then(r => r.json()).then(d => {
  S = d; hero(); charts(); watch(); parallax();
  $("#play").onclick = togglePlay;
  addEventListener("wheel", () => playing && stopPlay(), { passive: true });
  addEventListener("touchstart", () => playing && stopPlay(), { passive: true });
  addEventListener("scroll", () => requestAnimationFrame(onScroll), { passive: true });
  loadRun(false);
  let w = innerWidth;
  addEventListener("resize", () => { if (innerWidth === w) return; w = innerWidth; charts(); watch(); sizeTheater(); buildArena(); render(Math.max(0, DAY)); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { charts(); buildArena(); render(Math.max(0, DAY)); });
});

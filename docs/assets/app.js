const $ = s => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent && parent.appendChild(e); return e; };
const COND = ["A", "B", "C", "D", "E"];
const ACT = { HOLD: "", SIGNAL_FLEXIBILITY: "sig", CONCEDE: "con" };
const ACTNAME = { HOLD: "Hold", SIGNAL_FLEXIBILITY: "Signal flexibility", CONCEDE: "Concede" };
let S;

function tip(fig) { let t = fig.querySelector(".tip"); if (!t) { t = document.createElement("div"); t.className = "tip"; fig.appendChild(t); } return t; }
function showTip(fig, html, x, y) { const t = tip(fig); t.innerHTML = html; t.style.opacity = 1; const w = fig.clientWidth; t.style.left = Math.min(Math.max(0, x - t.offsetWidth / 2), w - t.offsetWidth) + "px"; t.style.top = (y - t.offsetHeight - 12) + "px"; }
const hideTip = fig => tip(fig).style.opacity = 0;

/* hero dots */
function hero() {
  const box = $("#dots"); const runs = [...S.runs].sort((a, b) => (a.conceder === "D") - (b.conceder === "D"));
  runs.forEach((r, i) => { const s = document.createElement("span"); s.className = r.conceder === "R" ? "r" : "d"; s.style.transitionDelay = (i * 12) + "ms"; box.appendChild(s); });
  $("#bigR").textContent = S.rConcede; $("#bigN").textContent = S.n;
  requestAnimationFrame(() => setTimeout(() => box.classList.add("in"), 120));
}

/* strip chart: concession day by stress level */
function strip() {
  const fig = $("#strip"); fig.innerHTML = "";
  const W = fig.clientWidth, small = W < 600, L = small ? 78 : 140, R = 12, rowH = small ? 64 : 76, H = rowH * 5 + 40;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, height: H }, fig);
  const x = d => L + (d / 30) * (W - L - R);
  [0, 10, 20, 30].forEach(d => { el("line", { x1: x(d), x2: x(d), y1: 0, y2: rowH * 5, class: "tick" }, svg); el("text", { x: x(d), y: rowH * 5 + 22, "text-anchor": "middle" }, svg).textContent = d === 30 ? "Day 30 · X-date" : "Day " + d; });
  COND.forEach((c, i) => {
    const cy = i * rowH + rowH / 2;
    el("line", { x1: L, x2: W - R, y1: cy, y2: cy, class: "axis" }, svg);
    const t = el("text", { x: 0, y: cy - 2, class: "lbl" }, svg); t.textContent = S.conditions[c];
    el("text", { x: 0, y: cy + 15 }, svg).textContent = "stress " + c;
    const rs = S.runs.filter(r => r.cond === c).sort((a, b) => a.day - b.day);
    const seen = {};
    rs.forEach(r => {
      const k = r.day; const n = seen[k] = (seen[k] || 0) + 1; const off = (n % 2 ? 1 : -1) * Math.floor(n / 2) * 9;
      const dot = el("circle", { cx: x(r.day), cy: cy + off, r: small ? 4 : 5, fill: r.conceder === "R" ? css("--rep") : css("--dem"), "fill-opacity": .85, style: "cursor:pointer" }, svg);
      dot.addEventListener("mouseenter", () => showTip(fig, `${S.episodes[r.ep]}${r.masked ? " · masked" : ""}<br>${r.conceder === "R" ? "Republican" : "Democrat"} conceded on day ${r.day + 1}`, x(r.day), cy + off));
      dot.addEventListener("mouseleave", () => hideTip(fig));
      dot.addEventListener("click", () => { pick.ep = r.ep; pick.cond = r.cond; pick.run = r.id; replay(); $("#replay").scrollIntoView(); });
    });
    const m = S.byCondition[c].medianDay;
    el("line", { x1: x(m), x2: x(m), y1: cy - 24, y2: cy + 24, stroke: css("--fg"), "stroke-width": 2 }, svg);
  });
  const a = S.byCondition.A, e = S.byCondition.E;
  $("#stats").innerHTML = `<div><b>Day ${a.medianDay + 1}</b><span>typical deal in calm markets</span></div>
    <div><b>Day ${e.medianDay + 1}</b><span>typical deal under extreme stress</span></div>
    <div><b>${Math.round(100 * S.rConcede / S.n)}%</b><span>of standoffs, the Republican agent conceded</span></div>`;
}

/* scatter: cost ratio vs concession day */
function scatter() {
  const fig = $("#scatter"); fig.innerHTML = "";
  const W = fig.clientWidth, H = Math.min(420, W * .62), L = 44, B = 46, T = 10, R = 12;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, height: H }, fig);
  const xs = S.runs.map(r => r.ratio), x0 = Math.min(...xs) - .02, x1 = Math.max(...xs) + .02;
  const x = v => L + (v - x0) / (x1 - x0) * (W - L - R), y = d => T + (1 - d / 30) * (H - T - B);
  [0, 10, 20, 30].forEach(d => { el("line", { x1: L, x2: W - R, y1: y(d), y2: y(d), class: "tick" }, svg); el("text", { x: L - 10, y: y(d) + 4, "text-anchor": "end" }, svg).textContent = d; });
  el("line", { x1: x(1), x2: x(1), y1: T, y2: H - B, stroke: css("--faint"), "stroke-dasharray": "3 3" }, svg);
  el("text", { x: x(1) + 6, y: T + 12 }, svg).textContent = "equal pain";
  el("text", { x: L, y: H - 12 }, svg).textContent = "← Democrat hurts more";
  el("text", { x: W - R, y: H - 12, "text-anchor": "end" }, svg).textContent = "Republican hurts more →";
  const g = S.regression.pooled;
  el("line", { x1: x(x0), y1: y(g.const + g.beta * x0), x2: x(x1), y2: y(g.const + g.beta * x1), stroke: css("--fg"), "stroke-width": 1.5 }, svg);
  S.runs.forEach(r => {
    const c = el("circle", { cx: x(r.ratio), cy: y(r.day), r: 5, fill: r.conceder === "R" ? css("--rep") : css("--dem"), "fill-opacity": .8 }, svg);
    c.addEventListener("mouseenter", () => showTip(fig, `Pain ratio ${r.ratio.toFixed(2)} · deal on day ${r.day + 1}`, x(r.ratio), y(r.day)));
    c.addEventListener("mouseleave", () => hideTip(fig));
  });
  const h = S.regression.historical;
  $("#regnote").innerHTML = `Each dot is one run; the line fits all 80. Vertical: day of concession. Horizontal: Republican ÷ Democrat average self-reported delay cost. Each 0.1 rise in the ratio brings the deal about <b>${Math.abs(h.beta / 10).toFixed(1)} days</b> earlier (historical runs, n = ${h.n}, p = ${h.p.toFixed(3)}). <a href="method.html#results">Full regression →</a>`;
}

/* replay */
const pick = { ep: "2023", cond: "C", run: null };
function pills(box, items, cur, on) {
  box.innerHTML = ""; items.forEach(([v, l]) => { const b = document.createElement("button"); b.textContent = l; b.className = v === cur ? "on" : ""; b.onclick = () => on(v); box.appendChild(b); });
}
async function replay() {
  pills($("#pEp"), Object.entries(S.episodes).map(([k, v]) => [k, v.split(" · ")[0]]), pick.ep, v => { pick.ep = v; pick.run = null; replay(); });
  pills($("#pCond"), COND.map(c => [c, S.conditions[c]]), pick.cond, v => { pick.cond = v; pick.run = null; replay(); });
  const opts = S.runs.filter(r => r.ep === pick.ep && r.cond === pick.cond).sort((a, b) => a.masked - b.masked || a.rep - b.rep);
  if (!opts.find(r => r.id === pick.run)) pick.run = opts[0].id;
  pills($("#pRun"), opts.map((r, i) => [r.id, (r.masked ? "Masked " : "Run ") + (r.rep + 1)]), pick.run, v => { pick.run = v; replay(); });
  const run = await (await fetch(`data/runs/${pick.run}.json`)).json();
  const who = run.conceder === "R" ? '<span class="rep">Republican</span>' : '<span class="dem">Democrat</span>';
  $("#verdict").innerHTML = `The ${who} conceded on day ${run.day + 1}.`;
  $("#vmeta").textContent = `${S.episodes[run.ep]} · ${S.conditions[run.cond]} markets${run.masked ? " · names & outcome masked" : ""}`;
  ["R", "D"].forEach(side => {
    const lane = $("#lane" + side); lane.innerHTML = "";
    for (let t = 0; t < 30; t++) {
      const p = run.periods.find(q => q.t === t); const b = document.createElement("button");
      b.className = "cell " + (p ? ACT[p[side].a] : "void"); b.dataset.t = t;
      if (p) { b.title = `Day ${t + 1}: ${ACTNAME[p[side].a]}`; b.onclick = () => showDay(run, t); }
      lane.appendChild(b);
    }
  });
  costs(run); showDay(run, run.periods[run.periods.length - 1].t);
}
function costs(run) {
  const fig = $("#costs"); fig.innerHTML = "";
  const W = fig.clientWidth, H = 90, cw = W / 30;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, height: H }, fig);
  const x = t => t * cw + cw / 2, y = v => H - 6 - (v / 10) * (H - 16);
  const st = run.periods.map(p => `${x(p.t)},${H - 6 - p.stress * (H - 16)}`);
  el("polygon", { points: `${x(run.periods[0].t)},${H - 6} ${st.join(" ")} ${x(run.periods.at(-1).t)},${H - 6}`, fill: css("--soft") }, svg);
  [["R", "--rep"], ["D", "--dem"]].forEach(([s, c]) => el("polyline", { points: run.periods.map(p => `${x(p.t)},${y(p[s].c)}`).join(" "), fill: "none", stroke: css(c), "stroke-width": 1.75 }, svg));
  el("text", { x: 0, y: 11 }, svg).textContent = "felt cost of delay (0–10) · grey: market stress";
}
function showDay(run, t) {
  document.querySelectorAll(".cell").forEach(c => c.classList.toggle("col-on", +c.dataset.t === t));
  const p = run.periods.find(q => q.t === t);
  const side = (s, name, col) => { const d = p[s]; return `<div>
    <div class="head"><h3 class="${col}">${name}</h3><span class="act" style="background:var(--${col}-soft);color:var(--${col})">${ACTNAME[d.a]}</span></div>
    <blockquote>“${d.s || "…"}”</blockquote>
    <div class="meter">Felt cost of delay · ${d.c}/10</div><div class="bar"><i style="width:${d.c * 10}%;background:var(--${col})"></i></div>
    <div class="meter">Chance it concedes now · ${Math.round(d.p * 100)}%</div><div class="bar"><i style="width:${d.p * 100}%;background:var(--${col})"></i></div>
    <details><summary>Private reasoning</summary><p>${d.r || ""}</p></details></div>`; };
  $("#day").innerHTML = `<div class="daymeta"><span>DAY ${t + 1} · ${p.date}</span><span>${p.x} DAYS TO X-DATE</span><span>VIX ${p.vix.toFixed(1)}</span><span>STRESS ${p.stress.toFixed(2)}</span><span>APPROVAL ${p.appr.toFixed(0)}%</span></div>`
    + side("R", "Republican agent", "rep") + side("D", "Democratic agent", "dem");
}

fetch("data/summary.json").then(r => r.json()).then(d => {
  S = d; hero(); strip(); scatter(); replay();
  let w = innerWidth; addEventListener("resize", () => { if (innerWidth !== w) { w = innerWidth; strip(); scatter(); replay(); } });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { strip(); scatter(); replay(); });
});

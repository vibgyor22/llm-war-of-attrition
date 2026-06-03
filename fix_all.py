"""
Fix all Plotly layout conflicts in dashboard/app.py.

Strategy:
  - _BASE contains ONLY paper_bgcolor, plot_bgcolor, font, hoverlabel
    (no xaxis, yaxis, legend — these always cause duplicate-keyword conflicts)
  - _lay() does two separate update_layout calls:
      1. Theme globals (from _BASE + height + margin)  — never conflicts
      2. Per-chart **kw (xaxis, yaxis, legend, title, barmode, etc.) — no overlap
  - After both calls: update_xaxes/update_yaxes sets grid/tick defaults
    on ALL axes including subplot axes (xaxis2, yaxis2, etc.)
"""
import ast, re
from pathlib import Path

src = Path("dashboard/app.py").read_text(encoding="utf-8")

# ── 1. Replace _BASE (remove xaxis, yaxis, legend) ──────────────────────────
OLD_BASE = '''\
_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG2,
    font=dict(color="#b0bcd4", family="Courier New, monospace", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0.38)", font=dict(size=10), borderwidth=0),
    hoverlabel=dict(bgcolor="#080820", font=dict(family="Courier New, monospace", size=11)),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
               tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
               tickfont=dict(size=10)),
)'''

NEW_BASE = '''\
_GRID = "rgba(255,255,255,0.05)"   # axis grid colour used everywhere
_BASE = dict(                       # ONLY global properties — no xaxis/yaxis/legend
    paper_bgcolor=_BG,
    plot_bgcolor=_BG2,
    font=dict(color="#b0bcd4", family="Courier New, monospace", size=11),
    hoverlabel=dict(bgcolor="#080820", font=dict(family="Courier New, monospace", size=11)),
)'''

assert OLD_BASE in src, "Could not find _BASE block"
src = src.replace(OLD_BASE, NEW_BASE)

# ── 2. Replace _lay() ────────────────────────────────────────────────────────
# Find current _lay() definition (any version) and replace
_lay_pattern = re.compile(
    r'def _lay\(fig.*?\n    return fig\n', re.DOTALL
)
NEW_LAY = '''\
def _lay(fig: go.Figure, h: int = 300, margin: dict | None = None, **kw) -> go.Figure:
    """Two-step layout: globals first, then per-chart kw. Zero duplicate-key risk."""
    # Step 1 — global theme (paper/plot bg, font, hover)
    fig.update_layout(**_BASE, height=h, margin=margin or _M)
    # Step 2 — per-chart props (title, barmode, xaxis, yaxis, legend, yaxis2, …)
    if kw:
        fig.update_layout(**kw)
    # Step 3 — grid & tick defaults on every axis (subplots included)
    fig.update_xaxes(gridcolor=_GRID, zeroline=False, tickfont_size=10)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False, tickfont_size=10)
    return fig

'''

m = _lay_pattern.search(src)
assert m, "Could not find _lay() definition"
src = src[:m.start()] + NEW_LAY + src[m.end():]

# ── 3. Convert all remaining direct update_layout(**_BASE, …) calls ─────────
# These appear as:  fig_X.update_layout(\n    **_BASE, ...
# After step 2 above _lay() is defined; any leftover direct spreads must go.
# Safest: route them through _lay() by stripping **_BASE from the call.
# Pattern: find    fig_X.update_layout(\n        **_BASE, height=NNN, ...
# and rewrite to   _lay(fig_X, h=NNN, ...

def convert_direct_base_calls(s: str) -> str:
    """Replace fig.update_layout(**_BASE, height=H, ...) with _lay(fig, h=H, ...)."""
    result = []
    i = 0
    while i < len(s):
        # Look for pattern:  <fig_var>.update_layout(
        m = re.search(r'(\w+)\.update_layout\(', s[i:])
        if not m:
            result.append(s[i:]); break
        pre_end = i + m.start()
        result.append(s[i:pre_end])
        fig_var = m.group(1)
        call_start = i + m.end() - 1  # position of opening '('
        # Check if **_BASE is the first arg
        inner_start = call_start + 1
        stripped = s[inner_start:inner_start+30].lstrip()
        if not stripped.startswith('**_BASE'):
            # Not a direct-base call, leave as-is up to and including this match
            result.append(s[pre_end: call_start + 1])
            i = call_start + 1
            continue
        # Find matching close paren
        depth = 1; j = call_start + 1
        while j < len(s) and depth:
            if s[j] == '(': depth += 1
            elif s[j] == ')': depth -= 1
            j += 1
        call_body = s[call_start+1:j-1]  # everything inside the outer parens
        # Strip leading **_BASE,
        call_body = re.sub(r'^\s*\*\*_BASE,\s*', '', call_body, count=1)
        # Extract height= if present
        hm = re.search(r'\bheight=(\d+)', call_body)
        h_val = hm.group(1) if hm else '300'
        if hm:
            call_body = call_body[:hm.start()] + call_body[hm.end():]
            call_body = re.sub(r'^,\s*', '', call_body)
        # Extract margin=dict(...) if present at top level
        mm = re.search(r'\bmargin=(dict\([^)]*\))', call_body)
        margin_val = mm.group(1) if mm else None
        if mm:
            call_body = call_body[:mm.start()] + call_body[mm.end():]
            call_body = re.sub(r'^,\s*', '', call_body)
        # Strip leading/trailing comma-whitespace
        call_body = call_body.strip().strip(',').strip()
        # Build _lay() call
        lay_args = f'h={h_val}'
        if margin_val:
            lay_args += f', margin={margin_val}'
        if call_body:
            lay_args += f',\n                 {call_body}'
        result.append(f'_lay({fig_var}, {lay_args})')
        i = j
    return ''.join(result)

before = src.count('update_layout(**_BASE')
src = convert_direct_base_calls(src)
after = src.count('update_layout(**_BASE')
print(f"Direct **_BASE calls: {before} -> {after}")

# ── 4. Syntax check ──────────────────────────────────────────────────────────
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    # Show context
    lines = src.splitlines()
    lo = max(0, e.lineno - 4)
    hi = min(len(lines), e.lineno + 3)
    print(f"SYNTAX ERROR line {e.lineno}: {e.msg}")
    for n, l in enumerate(lines[lo:hi], lo+1):
        print(f"  {n:4d}  {l}")
    raise

Path("dashboard/app.py").write_text(src, encoding="utf-8")
print("Saved.")

#!/usr/bin/env python3
"""Build EvoClaw leaderboard page.

Reads analysis CSVs and generates visualization/leaderboard/index.html.
Usage: python visualization/leaderboard/build.py
"""

import json
import shutil
import sys
import pathlib

import pandas as pd

VIZ_DIR = pathlib.Path(__file__).resolve().parent
AGENTBENCH_ROOT = pathlib.Path("/home/gangda/workspace/AgentBench")
ANALYSIS_ROOT = AGENTBENCH_ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from common import load_e2e, AGENT_MODEL_ORDER

DATA_DIR = ANALYSIS_ROOT / "data"
E2E_TRIAL_CSV = DATA_DIR / "e2e_trial.csv"
OPENHANDS_TRIAL_CSV = DATA_DIR / "openhands_e2e_trial.csv"

# ── Model name normalization ────────────────────────────────────────────────
MODEL_ALIASES = {
    "gemini-3-pro-preview": "gemini-3-pro",
    "gemini-3-flash-preview": "gemini-3-flash",
}

# ── Agent groups (same order as paper table) ────────────────────────────────
AGENT_GROUPS = [
    ("claude-code", [
        "claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101",
        "claude-sonnet-4-6", "claude-opus-4-6",
    ]),
    ("codex", ["gpt-5.2-codex", "gpt-5.2", "gpt-5.3-codex"]),
    ("gemini-cli", ["gemini-3-pro", "gemini-3.1-pro", "gemini-3-flash"]),
]
OPENHANDS_MODELS = [
    "minimax-m2.5", "kimi-k2.5", "gemini-3-flash",
    "gpt-5.3-codex", "claude-opus-4-6",
]

# ── Display names ───────────────────────────────────────────────────────────
MODEL_DISPLAY = {
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-6": "Claude Opus 4.6",
    "gpt-5.2-codex": "GPT 5.2 Codex",
    "gpt-5.2": "GPT 5.2",
    "gpt-5.3-codex": "GPT 5.3 Codex",
    "gemini-3-pro": "Gemini 3 Pro",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "gemini-3-flash": "Gemini 3 Flash",
    "kimi-k2.5": "Kimi K2.5",
    "minimax-m2.5": "MiniMax M2.5",
}
AGENT_DISPLAY = {
    "claude-code": "Claude Code",
    "codex": "Codex CLI",
    "gemini-cli": "Gemini CLI",
    "openhands": "OpenHands",
}
MODEL_ORG = {
    "claude-sonnet-4-5-20250929": "Anthropic",
    "claude-opus-4-5-20251101": "Anthropic",
    "claude-sonnet-4-6": "Anthropic",
    "claude-opus-4-6": "Anthropic",
    "gpt-5.2-codex": "OpenAI",
    "gpt-5.2": "OpenAI",
    "gpt-5.3-codex": "OpenAI",
    "gemini-3-pro": "Google",
    "gemini-3.1-pro": "Google",
    "gemini-3-flash": "Google",
    "kimi-k2.5": "Moonshot AI",
    "minimax-m2.5": "MiniMax",
}

# ── Chart labels ────────────────────────────────────────────────────────────
CHART_LABELS = {
    ("claude-code", "claude-sonnet-4-5-20250929"): "Claude Sonnet 4.5",
    ("claude-code", "claude-opus-4-5-20251101"): "Claude Opus 4.5",
    ("claude-code", "claude-sonnet-4-6"): "Claude Sonnet 4.6",
    ("claude-code", "claude-opus-4-6"): "Claude Opus 4.6",
    ("codex", "gpt-5.2-codex"): "GPT-5.2-Codex",
    ("codex", "gpt-5.2"): "GPT-5.2",
    ("codex", "gpt-5.3-codex"): "GPT-5.3-Codex",
    ("gemini-cli", "gemini-3-pro"): "Gemini 3 Pro",
    ("gemini-cli", "gemini-3.1-pro"): "Gemini 3.1 Pro",
    ("gemini-cli", "gemini-3-flash"): "Gemini 3 Flash",
    ("openhands", "minimax-m2.5"): "MiniMax M2.5",
    ("openhands", "kimi-k2.5"): "Kimi K2.5",
    ("openhands", "gemini-3-flash"): "Gemini 3 Flash",
    ("openhands", "gpt-5.3-codex"): "GPT-5.3-Codex",
    ("openhands", "claude-opus-4-6"): "Claude Opus 4.6",
}

# ── Per-entry text position overrides (to avoid overlap) ───────────────────
# Default is 'middle right'.
CHART_TEXT_POS = {
    ("claude-code", "claude-sonnet-4-5-20250929"): "middle left",
    ("codex", "gpt-5.2"): "middle left",
    ("codex", "gpt-5.3-codex"): "middle left",
    ("gemini-cli", "gemini-3-flash"): "middle left",
    ("openhands", "minimax-m2.5"): "middle left",
    ("openhands", "kimi-k2.5"): "middle left",
    ("openhands", "claude-opus-4-6"): "middle left",
}

# ── Per-entry colors (brightened for dark bg) ───────────────────────────────
ENTRY_COLORS = {
    ("claude-code", "claude-sonnet-4-5-20250929"): "#D07A5E",
    ("claude-code", "claude-opus-4-5-20251101"): "#D07A5E",
    ("claude-code", "claude-sonnet-4-6"): "#D07A5E",
    ("claude-code", "claude-opus-4-6"): "#D07A5E",
    ("codex", "gpt-5.2-codex"): "#90C890",
    ("codex", "gpt-5.2"): "#90C890",
    ("codex", "gpt-5.3-codex"): "#90C890",
    ("gemini-cli", "gemini-3-pro"): "#7AAED8",
    ("gemini-cli", "gemini-3.1-pro"): "#7AAED8",
    ("gemini-cli", "gemini-3-flash"): "#7AAED8",
    ("openhands", "minimax-m2.5"): "#E06070",
    ("openhands", "kimi-k2.5"): "#D4A050",
    ("openhands", "gemini-3-flash"): "#7AAED8",
    ("openhands", "gpt-5.3-codex"): "#90C890",
    ("openhands", "claude-opus-4-6"): "#D07A5E",
}

ORG_COLORS = {
    "Anthropic": "#D97757",
    "OpenAI": "#10A37F",
    "Google": "#4285F4",
    "Moonshot AI": "#FFFFFF",
    "MiniMax": "#F03A5D",
}
AGENT_COLORS = {
    "claude-code": {"bg": "rgba(217,119,87,0.15)", "fg": "#D97757"},
    "codex": {"bg": "rgba(16,163,127,0.15)", "fg": "#10A37F"},
    "gemini-cli": {"bg": "rgba(66,133,244,0.15)", "fg": "#4285F4"},
    "openhands": {"bg": "rgba(232,186,58,0.18)", "fg": "#e0b040"},
}


def compute_records():
    """Compute leaderboard data from CSVs."""
    e2e = load_e2e()
    e2e["is_resolved"] = (e2e["eval_status"] == "passed").astype(float)

    # Trial-level output tokens
    trial_df = pd.read_csv(E2E_TRIAL_CSV)
    trial_df["model"] = trial_df["model"].replace(MODEL_ALIASES)
    trial_tokens = {}
    trial_duration = {}
    for model in trial_df["model"].unique():
        sub = trial_df[trial_df["model"] == model]
        trial_tokens[model] = sub["total_output_tokens"].mean() / 1000
        trial_duration[model] = sub["total_duration_ms"].mean() / 3_600_000

    records = []

    # Non-OpenHands agents
    for am in AGENT_MODEL_ORDER:
        sub = e2e[e2e["agent_model"] == am]
        if len(sub) == 0:
            continue
        agent = sub["agent_name"].iloc[0]
        model = sub["model"].iloc[0]

        ws_score = sub.groupby("workspace")["score_reliable"].mean()
        ws_prec = sub.groupby("workspace")["score_precision"].mean()
        ws_rec = sub.groupby("workspace")["score_recall"].mean()
        ws_res = sub.groupby("workspace")["is_resolved"].mean()
        ws_cost = sub.groupby("workspace")["m_cost_usd"].sum()
        ws_dur = trial_df[trial_df["model"] == model].set_index("workspace")["total_duration_ms"] / 3_600_000
        ws_turns = sub.groupby("workspace")["m_turns"].sum()

        records.append({
            "agent": agent,
            "agent_display": AGENT_DISPLAY.get(agent, agent),
            "model": model,
            "model_display": MODEL_DISPLAY.get(model, model),
            "org": MODEL_ORG.get(model, ""),
            "org_color": ORG_COLORS.get(MODEL_ORG.get(model, ""), "#888"),
            "agent_bg": AGENT_COLORS.get(agent, {}).get("bg", ""),
            "agent_fg": AGENT_COLORS.get(agent, {}).get("fg", ""),
            "color": ENTRY_COLORS.get((agent, model), "#888"),
            "chart_label": CHART_LABELS.get((agent, model), model),
            "chart_textpos": CHART_TEXT_POS.get((agent, model), "middle right"),
            "score": round(ws_score.mean() * 100, 2),
            "precision": round(ws_prec.mean() * 100, 2),
            "recall": round(ws_rec.mean() * 100, 2),
            "resolve": round(ws_res.mean() * 100, 2),
            "cost": round(ws_cost.mean(), 2),
            "out_tok_k": round(trial_tokens.get(model, 0)),
            "time_h": round(ws_dur.mean(), 2),
            "turns": round(ws_turns.mean()),
        })

    # OpenHands agents
    oh_df = pd.read_csv(OPENHANDS_TRIAL_CSV)
    for model in OPENHANDS_MODELS:
        sub = oh_df[oh_df["model"] == model]
        if len(sub) == 0:
            continue
        reliable = sub[sub["total_output_tokens"] > 0]
        tok_sub = reliable if len(reliable) < len(sub) else sub

        records.append({
            "agent": "openhands",
            "agent_display": "OpenHands",
            "model": model,
            "model_display": MODEL_DISPLAY.get(model, model),
            "org": MODEL_ORG.get(model, ""),
            "org_color": ORG_COLORS.get(MODEL_ORG.get(model, ""), "#888"),
            "agent_bg": AGENT_COLORS["openhands"]["bg"],
            "agent_fg": AGENT_COLORS["openhands"]["fg"],
            "color": ENTRY_COLORS.get(("openhands", model), "#888"),
            "chart_label": CHART_LABELS.get(("openhands", model), model),
            "chart_textpos": CHART_TEXT_POS.get(("openhands", model), "middle right"),
            "score": round(sub["mean_score_reliable"].mean() * 100, 2),
            "precision": round(sub["mean_score_precision"].mean() * 100, 2),
            "recall": round(sub["mean_score_recall"].mean() * 100, 2),
            "resolve": round(sub["resolve_rate"].mean() * 100, 2),
            "cost": round(tok_sub["total_cost_usd"].mean(), 2),
            "out_tok_k": round(tok_sub["total_output_tokens"].mean() / 1000),
            "time_h": round(sub["total_duration_ms"].mean() / 3_600_000, 2),
            "turns": round(tok_sub["total_turns"].mean()),
        })

    # Mark official entries: non-openhands are always official;
    # openhands is official only if that model has no native agent entry.
    native_models = {r["model"] for r in records if r["agent"] != "openhands"}
    for r in records:
        if r["agent"] == "openhands" and r["model"] in native_models:
            r["is_official"] = False
        else:
            r["is_official"] = True

    records.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(records):
        r["rank"] = i + 1

    return records


def main():
    print("Computing leaderboard data...")
    records = compute_records()
    data_json = json.dumps(records)

    # Load logos
    logos_path = VIZ_DIR / "logos.json"
    if logos_path.exists():
        logos = json.load(open(logos_path))
        print(f"  loaded {len(logos)} logos")
    else:
        logos = {}
        print("  [warn] logos.json not found")
    logos_json = json.dumps(logos)

    # Copy CSVs
    out_data = VIZ_DIR / "data"
    out_data.mkdir(exist_ok=True)
    for csv_name in ["e2e.csv", "e2e_trial.csv", "openhands_e2e_trial.csv"]:
        src = DATA_DIR / csv_name
        if src.exists():
            shutil.copy2(src, out_data / csv_name)

    # Generate HTML
    html = HTML_TEMPLATE.replace('"__LEADERBOARD_DATA__"', data_json)
    html = html.replace('"__LOGOS__"', logos_json)
    out_path = VIZ_DIR / "index.html"
    out_path.write_text(html)
    print(f"  wrote {out_path} ({len(records)} entries)")


# ═══════════════════════════════════════════════════════════════════════════
# HTML Template
# ═══════════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EvoClaw Leaderboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script>(function(){var t=localStorage.getItem('evoclaw-theme');if(t==='light'){document.documentElement.setAttribute('data-theme','light');document.addEventListener('DOMContentLoaded',function(){document.querySelectorAll('.theme-btn').forEach(function(b){b.classList.toggle('active',b.dataset.themeVal==='light')})})}})()</script>
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg-0:#08080d;--bg-1:#0e0e16;--bg-2:#14141f;--bg-3:#1c1c2b;
  --border:#262640;--text-1:#e2e8f0;--text-2:#94a3b8;--text-3:#64748b;
  --accent:#6366f1;--accent-light:#818cf8;
}
[data-theme="light"]{
  --bg-0:#f7f8fc;--bg-1:#ffffff;--bg-2:#eef0f8;--bg-3:#e2e5f0;
  --border:#cfd4e2;--text-1:#1a1d2e;--text-2:#4a5068;--text-3:#7c839a;
  --accent:#5b5cf6;--accent-light:#4f46e5;
}
body,.panel,.nav-btn,.table-wrap{transition:background-color .3s ease,border-color .3s ease}
html{scroll-behavior:smooth}
body{background:var(--bg-0);color:var(--text-1);font-family:'Inter',system-ui,-apple-system,sans-serif;line-height:1.6;min-height:100vh}
.container{max-width:1280px;margin:0 auto;padding:2.5rem 1.5rem}

/* ── Header ────────────────────────────────────────────────── */
.hero{text-align:left;padding:1.5rem 0 2.5rem}
.hero-title{font-size:2rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:1.25rem}
.hero-title .brand{
  background:linear-gradient(135deg,#6366f1 0%,#a78bfa 50%,#c4b5fd 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero-title .sep{color:var(--text-3);font-weight:400;margin:0 .15em}
.hero-title .desc{color:var(--text-2);font-weight:400;font-size:.85em}
.hero-desc{color:var(--text-3);font-size:.9rem;margin:0 0 1.25rem;line-height:1.7}
.hero-badges{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.25rem}
.hero-badges a img{height:22px}
.nav-btns{display:flex;gap:.6rem;justify-content:flex-start;flex-wrap:wrap}
.nav-btn{display:inline-flex;align-items:center;gap:.45rem;
  padding:.55rem 1.2rem;border-radius:8px;font-size:.875rem;font-weight:500;
  text-decoration:none;color:var(--text-2);background:var(--bg-2);
  border:1px solid var(--border);transition:all .2s ease;cursor:pointer}
.nav-btn:hover{color:var(--text-1);border-color:var(--accent);background:var(--bg-3)}
.nav-btn.active{color:#fff;background:var(--accent);border-color:var(--accent)}
.nav-btn svg{width:16px;height:16px;flex-shrink:0}

/* ── Panels ────────────────────────────────────────────────── */
.panel{background:var(--bg-1);border:1px solid var(--border);border-radius:14px;padding:1.5rem;margin-bottom:1.75rem}
.panel-title{font-size:1.15rem;font-weight:600;margin-bottom:1rem;color:var(--text-1)}
#chart{width:100%;height:500px;border-radius:8px;background:var(--bg-2)}

/* ── Table ─────────────────────────────────────────────────── */
.table-wrap{overflow-x:auto;border-radius:8px;border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:.875rem}
thead{position:sticky;top:0;z-index:2}
th{background:var(--bg-3);color:var(--text-2);font-weight:600;padding:.7rem .9rem;
  text-align:left;white-space:nowrap;cursor:pointer;user-select:none;
  border-bottom:2px solid var(--border);transition:color .15s}
th:hover{color:var(--text-1)}
th.sort-asc::after{content:' \2191';color:var(--accent)}
th.sort-desc::after{content:' \2193';color:var(--accent)}
th.num-col{text-align:right}
td{padding:.55rem .9rem;border-bottom:1px solid rgba(38,38,64,.5);white-space:nowrap;transition:background .15s}
tr:hover td{background:rgba(99,102,241,.04)}
.num{text-align:right;font-variant-numeric:tabular-nums}

/* rank */
.rank-cell{text-align:center;font-weight:700;width:48px;font-size:.95rem}
.rank-medal{font-size:1.1rem}

/* org logo */
.org-logo{width:22px;min-width:22px;height:22px;margin-right:8px;vertical-align:middle;flex-shrink:0;border-radius:4px;object-fit:contain;object-position:center}
.org-badge{display:inline-flex;align-items:center;justify-content:center;
  width:22px;height:22px;border-radius:5px;font-size:10px;font-weight:700;
  color:#fff;margin-right:8px;vertical-align:middle;flex-shrink:0}
.model-cell{align-items:center;font-weight:500}
/* agent icon in table */
.agent-cell{display:inline-flex;align-items:center;gap:8px;vertical-align:middle}
.agent-icon{width:22px;height:22px;object-fit:contain;flex-shrink:0;border-radius:4px}

/* agent badge */
.agent-badge{display:inline-block;padding:2px 10px;border-radius:6px;font-size:.78rem;font-weight:500}

/* score bar */
.score-cell{position:relative;text-align:right;font-variant-numeric:tabular-nums}
.score-bar{position:absolute;right:0;top:4px;bottom:4px;border-radius:4px;opacity:.18;z-index:0}
.score-val{position:relative;z-index:1;font-weight:600}
.best-val{color:var(--accent-light);font-weight:700}

/* top-3 row accents */
tr.top-1 td{background:rgba(251,191,36,.04)}
tr.top-2 td{background:rgba(192,192,210,.025)}
tr.top-3 td{background:rgba(205,127,50,.025)}

/* ── Footer ────────────────────────────────────────────────── */
footer{text-align:center;padding:2rem 0 1rem;color:var(--text-3);font-size:.8rem}
footer a{color:var(--text-2);text-decoration:none}
footer a:hover{color:var(--accent)}

/* ── Tooltip ──────────────────────────────────────────────── */
.tip-popup{position:fixed;background:var(--bg-3);color:var(--text-1);border:1px solid var(--border);border-radius:8px;
  padding:6px 14px;font-size:.82rem;font-weight:500;white-space:nowrap;
  pointer-events:none;z-index:9999;opacity:0;transition:opacity .15s;
  box-shadow:0 4px 16px rgba(0,0,0,.15)}
.tip-popup.show{opacity:1}

/* ── Toggle ───────────────────────────────────────────────── */
.toggle-label{display:inline-flex;align-items:center;gap:.5rem;cursor:pointer;font-size:.85rem;color:var(--text-2)}
.toggle-label input{width:16px;height:16px;accent-color:var(--accent);cursor:pointer}
.toggle-text{user-select:none}

/* ── Plotly overrides ──────────────────────────────────────── */
.js-plotly-plot .plotly .modebar{background:transparent !important}
.js-plotly-plot .plotly .modebar-btn path{fill:var(--text-3) !important}
.js-plotly-plot .plotly .modebar-btn:hover path{fill:var(--accent) !important}

/* ── Theme toggle ─────────────────────────────────────────── */
.theme-toggle{position:fixed;top:1.25rem;right:1.5rem;z-index:100;
  display:flex;border-radius:10px;padding:2px;
  background:var(--bg-3);border:1px solid var(--border);
  box-shadow:0 2px 8px rgba(0,0,0,.1);
  transition:background-color .3s,border-color .3s}
.theme-btn{width:30px;height:26px;border:none;background:transparent;
  cursor:pointer;display:flex;align-items:center;justify-content:center;
  color:var(--text-3);transition:background-color .2s,color .2s,box-shadow .2s;
  border-radius:7px;padding:0}
.theme-btn svg{width:14px;height:14px}
.theme-btn:hover{color:var(--text-1)}
.theme-btn.active{background:var(--bg-1);color:var(--accent);box-shadow:0 1px 3px rgba(0,0,0,.12)}
[data-theme="light"] tr.top-1 td{background:rgba(251,191,36,.08)}
[data-theme="light"] tr.top-2 td{background:rgba(160,160,180,.06)}
[data-theme="light"] tr.top-3 td{background:rgba(205,127,50,.06)}
[data-theme="light"] .score-bar{opacity:.25}
[data-theme="light"] tr:hover td{background:rgba(79,70,229,.06)}

@media(max-width:768px){
  .hero-title{font-size:1.4rem}
  .container{padding:1.5rem 1rem}
  #chart{height:360px}
  .theme-toggle{top:1rem;right:1rem}
}
</style>
</head>
<body>
<div class="theme-toggle">
  <button class="theme-btn active" data-theme-val="dark" onclick="setTheme('dark')" aria-label="Dark mode">
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M13.9 8.6a6 6 0 01-6.5-6.5A6 6 0 108.6 13.9a6 6 0 005.3-5.3z"/></svg>
  </button>
  <button class="theme-btn" data-theme-val="light" onclick="setTheme('light')" aria-label="Light mode">
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="8" r="3"/><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41"/></svg>
  </button>
</div>
<div class="container">

<!-- ═══ Header ═══ -->
<header class="hero">
  <h1 class="hero-title">
    <span class="brand">EvoClaw</span><span class="sep">:</span><span class="desc">Evaluating AI Agents on Continuous Software Evolution</span>
  </h1>
  <p class="hero-desc">Long-running agents build customized software (a &ldquo;Claw&rdquo;) to interact with their environments. For practical use in complex, real-world tasks, these agents must fully and autonomously evolve this software in response to a continuous stream of end-user requirements. EvoClaw evaluates how well frontier LLM agents handle this continuous development, benchmarking them against real-world evolution itineraries from open-source repositories.</p>
  <div class="hero-badges">
    <a href="https://arxiv.org/pdf/2603.13428" target="_blank"><img src="https://img.shields.io/badge/arXiv-2603.13428-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
    <a href="https://huggingface.co/datasets/hyd2apse/EvoClaw-data" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow" alt="HuggingFace"></a>
    <a href="https://evo-claw.com/" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Website-evo--claw.com-blue" alt="Website"></a>
  </div>
  <nav class="nav-btns">
    <a class="nav-btn active" href="#leaderboard">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 2h8v5a4 4 0 01-8 0V2z"/><path d="M4 4H2.5a1 1 0 00-1 1v.5A2.5 2.5 0 004 8"/><path d="M12 4h1.5a1 1 0 011 1v.5A2.5 2.5 0 0112 8"/><path d="M6 11v2h4v-2"/><path d="M5 13h6"/></svg>
      Leaderboard</a>
    <a class="nav-btn" href="https://arxiv.org/pdf/2603.13428" target="_blank">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 1H4a1 1 0 00-1 1v12a1 1 0 001 1h8a1 1 0 001-1V5L9 1z"/><path d="M9 1v4h4"/><path d="M5 8h6M5 11h4"/></svg>
      Paper</a>
    <a class="nav-btn" href="https://huggingface.co/datasets/hyd2apse/EvoClaw-data" target="_blank">
      <svg viewBox="0 0 24 24" fill="currentColor" fill-rule="evenodd"><path d="M16.781 3.277c2.997 1.704 4.844 4.851 4.844 8.258 0 .995-.155 1.955-.443 2.857a1.332 1.332 0 011.125.4 1.41 1.41 0 01.2 1.723c.204.165.352.385.428.632l.017.062c.06.222.12.69-.2 1.166.244.37.279.836.093 1.236-.255.57-.893 1.018-2.128 1.5l-.202.078-.131.048c-.478.173-.89.295-1.061.345l-.086.024c-.89.243-1.808.375-2.732.394-1.32 0-2.3-.36-2.923-1.067a9.852 9.852 0 01-3.18.018C9.778 21.647 8.802 22 7.494 22a11.249 11.249 0 01-2.541-.343l-.221-.06-.273-.08a16.574 16.574 0 01-1.175-.405c-1.237-.483-1.875-.93-2.13-1.501-.186-.4-.151-.867.093-1.236a1.42 1.42 0 01-.2-1.166c.069-.273.226-.516.447-.694a1.41 1.41 0 01.2-1.722c.233-.248.557-.391.917-.407l.078-.001a9.385 9.385 0 01-.44-2.85c0-3.407 1.847-6.554 4.844-8.258a9.822 9.822 0 019.687 0zM4.188 14.758c.125.687 2.357 2.35 2.14 2.707-.19.315-.796-.239-.948-.386l-.041-.04-.168-.147c-.561-.479-2.304-1.9-2.74-1.432-.43.46.119.859 1.055 1.42l.784.467.136.083c1.045.643 1.12.84.95 1.113-.188.295-3.07-2.1-3.34-1.083-.27 1.011 2.942 1.304 2.744 2.006-.2.7-2.265-1.324-2.685-.537-.425.79 2.913 1.718 2.94 1.725l.16.04.175.042c1.227.284 3.565.65 4.435-.604.673-.973.64-1.709-.248-2.61l-.057-.057c-.945-.928-1.495-2.288-1.495-2.288l-.017-.058-.025-.072c-.082-.22-.284-.639-.63-.584-.46.073-.798 1.21.12 1.933l.05.038c.977.721-.195 1.21-.573.534l-.058-.104-.143-.25c-.463-.799-1.282-2.111-1.739-2.397-.532-.332-.907-.148-.782.541zm14.842-.541c-.533.335-1.563 2.074-1.94 2.751a.613.613 0 01-.687.302.436.436 0 01-.176-.098.303.303 0 01-.049-.06l-.014-.028-.008-.02-.007-.019-.003-.013-.003-.017a.289.289 0 01-.004-.048c0-.12.071-.266.25-.427.026-.024.054-.047.084-.07l.047-.036c.022-.016.043-.032.063-.049.883-.71.573-1.81.131-1.917l-.031-.006-.056-.004a.368.368 0 00-.062.006l-.028.005-.042.014-.039.017-.028.015-.028.019-.036.027-.023.02c-.173.158-.273.428-.31.542l-.016.054s-.53 1.309-1.439 2.234l-.054.054c-.365.358-.596.69-.702 1.018-.143.437-.066.868.21 1.353.055.097.117.195.187.296.882 1.275 3.282.876 4.494.59l.286-.07.25-.074c.276-.084.736-.233 1.2-.42l.188-.077.065-.028.064-.028.124-.056.081-.038c.529-.252.964-.543.994-.827l.001-.036a.299.299 0 00-.037-.139c-.094-.176-.271-.212-.491-.168l-.045.01c-.044.01-.09.024-.136.04l-.097.035-.054.022c-.559.23-1.238.705-1.607.745h.006a.452.452 0 01-.05.003h-.024l-.024-.003-.023-.005c-.068-.016-.116-.06-.14-.142a.22.22 0 01-.005-.1c.062-.345.958-.595 1.713-.91l.066-.028c.528-.224.97-.483.985-.832v-.04a.47.47 0 00-.016-.098c-.048-.18-.175-.251-.36-.251-.785 0-2.55 1.36-2.92 1.36-.025 0-.048-.007-.058-.024a.6.6 0 01-.046-.088c-.1-.238.068-.462 1.06-1.066l.209-.126c.538-.32 1.01-.588 1.341-.831.29-.212.475-.406.503-.6l.003-.028c.008-.113-.038-.227-.147-.344a.266.266 0 00-.07-.054l-.034-.015-.013-.005a.403.403 0 00-.13-.02c-.162 0-.369.07-.595.18-.637.313-1.431.952-1.826 1.285l-.249.215-.033.033c-.08.078-.288.27-.493.386l-.071.037-.041.019a.535.535 0 01-.122.036h.005a.346.346 0 01-.031.003l.01-.001-.013.001c-.079.005-.145-.021-.19-.095a.113.113 0 01-.014-.065c.027-.465 2.034-1.991 2.152-2.642l.009-.048c.1-.65-.271-.817-.791-.493zM11.938 2.984c-4.798 0-8.688 3.829-8.688 8.55 0 .692.083 1.364.24 2.008l.008-.009c.252-.298.612-.46 1.017-.46.355.008.699.117.993.312.22.14.465.384.715.694.261-.372.69-.598 1.15-.605.852 0 1.367.728 1.562 1.383l.047.105.06.127c.192.396.595 1.139 1.143 1.68 1.06 1.04 1.324 2.115.8 3.266a8.865 8.865 0 002.024-.014c-.505-1.12-.26-2.17.74-3.186l.066-.066c.695-.684 1.157-1.69 1.252-1.912.195-.655.708-1.383 1.56-1.383.46.007.889.233 1.15.605.25-.31.495-.553.718-.694a1.87 1.87 0 01.99-.312c.357 0 .682.126.925.36.14-.61.215-1.245.215-1.898 0-4.722-3.89-8.55-8.687-8.55zm1.857 8.926l.439-.212c.553-.264.89-.383.89.152 0 1.093-.771 3.208-3.155 3.262h-.184c-2.325-.052-3.116-2.06-3.156-3.175l-.001-.087c0-1.107 1.452.586 3.25.586.716 0 1.379-.272 1.917-.526zm4.017-3.143c.45 0 .813.358.813.8 0 .441-.364.8-.813.8a.806.806 0 01-.812-.8c0-.442.364-.8.812-.8zm-11.624 0c.448 0 .812.358.812.8 0 .441-.364.8-.812.8a.806.806 0 01-.813-.8c0-.442.364-.8.813-.8zm7.79-.841c.32-.384.846-.54 1.33-.394.483.146.83.564.878 1.06.048.495-.212.97-.659 1.203-.322.168-.447-.477-.767-.585l.002-.003c-.287-.098-.772.362-.925.079a1.215 1.215 0 01.14-1.36zm-4.323 0c.322.384.377.92.14 1.36-.152.283-.64-.177-.925-.079l.003.003c-.108.036-.194.134-.273.24l-.118.165c-.11.15-.22.262-.377.18a1.226 1.226 0 01-.658-1.204c.048-.495.395-.913.878-1.059a1.262 1.262 0 011.33.394z"/></svg>
      Data</a>
    <a class="nav-btn" href="https://github.com/Hydrapse/EvoClaw" target="_blank">
      <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
      Code</a>
    <a class="nav-btn" href="https://openhands.dev/blog/evoclaw-benchmark" target="_blank">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M1 3h10a2 2 0 012 2v8a1 1 0 01-1 1H4a3 3 0 01-3-3V3z"/><path d="M13 7h1a1 1 0 011 1v4a3 3 0 01-3 3h0"/><path d="M4 7h5M4 10h3"/></svg>
      Blog</a>
  </nav>
</header>

<!-- ═══ Chart ═══ -->
<section class="panel" id="overview">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem">
    <h2 class="panel-title" style="margin-bottom:0">Overall Cost / Performance on EvoClaw</h2>
    <label class="toggle-label">
      <input type="checkbox" id="officialToggle">
      <span class="toggle-text">Official agent only</span>
    </label>
  </div>
  <div id="chart"></div>
</section>

<!-- ═══ Leaderboard ═══ -->
<section class="panel" id="leaderboard">
  <h2 class="panel-title">Leaderboard</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th data-key="rank">#</th>
          <th data-key="model_display" data-type="str">Model</th>
          <th data-key="agent_display" data-type="str">Agent</th>
          <th data-key="score" class="num-col has-tip sort-desc" data-tip="F1 of Precision and Recall. Balances fixing required tests and avoiding regressions.">Score (%)</th>
          <th data-key="precision" class="num-col has-tip" data-tip="Ratio of bugs fixed without breaking existing tests. Higher means cleaner changes.">Precision (%)</th>
          <th data-key="recall" class="num-col has-tip" data-tip="Ratio of required test cases successfully fixed. Higher means the agent addresses more of the required changes.">Recall (%)</th>
          <th data-key="resolve" class="num-col has-tip" data-tip="% of milestones fully resolved, where all required tests pass with no regressions.">Resolve (%)</th>
          <th data-key="cost" class="num-col has-tip" data-tip="Average API cost per evolution range ($)">Cost ($)</th>
          <th data-key="out_tok_k" class="num-col has-tip" data-tip="Average output tokens per evolution range, incl. reasoning/thinking (K)">Out Tok. (K)</th>
          <th data-key="time_h" class="num-col has-tip" data-tip="Average wall-clock time per evolution range (hours)">Time (h)</th>
          <th data-key="turns" class="num-col has-tip" data-tip="Average agent turns per evolution range">Turns</th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</section>

<footer><p>EvoClaw Benchmark &middot; 2026</p></footer>
</div>

<script>
const DATA = "__LEADERBOARD_DATA__";
const LOGOS = "__LOGOS__";

// org key mapping
const ORG_KEY = { 'Anthropic':'anthropic','OpenAI':'openai','Google':'google','Moonshot AI':'moonshot','MiniMax':'minimax' };
const AGENT_KEY = { 'claude-code':'claude-code','codex':'codex','gemini-cli':'gemini-cli','openhands':'openhands' };

function isLight() { return document.documentElement.getAttribute('data-theme') === 'light'; }
function themeColors() {
  return isLight() ? {
    paper:'#ffffff', plot:'#ffffff', grid:'#e2e5f0', text:'#1a1d2e',
    text2:'#4a5068', hover_bg:'#e2e5f0', hover_border:'#cfd4e2',
  } : {
    paper:'#0e0e16', plot:'#14141f', grid:'#1c1c2b', text:'#e2e8f0',
    text2:'#c8d0dc', hover_bg:'#1c1c2b', hover_border:'#262640',
  };
}
function setTheme(mode) {
  if (mode === 'light') document.documentElement.setAttribute('data-theme', 'light');
  else document.documentElement.removeAttribute('data-theme');
  localStorage.setItem('evoclaw-theme', mode);
  document.querySelectorAll('.theme-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.themeVal === mode);
  });
  renderChart(); renderTable();
}

function logoKey(key) {
  if (isLight() && key === 'moonshot' && LOGOS['moonshot_light']) return 'moonshot_light';
  return key;
}

// ═══════════════════════════════════════════════════════════
// Shared state
// ═══════════════════════════════════════════════════════════
const orgs = [
  { key: 'Anthropic',   color: '#D97757' },
  { key: 'OpenAI',      color: '#10A37F' },
  { key: 'Google',      color: '#4285F4' },
  { key: 'Moonshot AI', color: '#FFFFFF' },
  { key: 'MiniMax',     color: '#F03A5D' },
];

function getFiltered() {
  const official = document.getElementById('officialToggle').checked;
  return official ? DATA.filter(d => d.is_official) : DATA;
}

// ═══════════════════════════════════════════════════════════
// Chart
// ═══════════════════════════════════════════════════════════
function renderChart() {
  const tc = themeColors();
  const fdata = getFiltered();

  const allX = fdata.map(d => d.cost), allY = fdata.map(d => d.score);
  const xMin = Math.min(...allX), xMax = Math.max(...allX);
  const yMin = Math.min(...allY), yMax = Math.max(...allY);
  const xPad = (xMax - xMin) * 0.15, xPadR = (xMax - xMin) * 0.25, yPad = (yMax - yMin) * 0.15;
  const iconX = (xMax - xMin + 2*xPad) * 0.025 * 0.9;
  const iconY = (yMax - yMin + 2*yPad) * 0.055 * 0.9;

  const traces = orgs.map(org => {
    const pts = fdata.filter(d => d.org === org.key);
    if (!pts.length) return null;
    return {
      x: pts.map(p => p.cost),
      y: pts.map(p => p.score),
      text: pts.map(() => ''),
      customdata: pts.map(p => [
        p.model_display, p.agent_display, p.precision, p.recall,
        p.resolve, p.out_tok_k, p.time_h, p.turns
      ]),
      mode: 'markers+text',
      type: 'scatter',
      name: org.key,
      marker: { size: 25, color: 'rgba(0,0,0,0)' },
      textposition: pts.map(p => p.chart_textpos || 'middle right'),
      textfont: { size: 13, color: tc.text2, family: 'Inter, sans-serif' },
      hovertemplate:
        '<b>%{customdata[0]}</b> (<b>%{customdata[1]}</b>)<br>' +
        'Score: %{y:.1f}%<br>' +
        'Cost: $%{x:.2f}<br>' +
        'Precision: %{customdata[2]:.1f}%<br>' +
        'Recall: %{customdata[3]:.1f}%<br>' +
        'Resolve: %{customdata[4]:.1f}%<br>' +
        'Output Tok: %{customdata[5]}K<br>' +
        'Time: %{customdata[6]:.1f}h<br>' +
        'Turns: %{customdata[7]}<extra></extra>',
      showlegend: false,
    };
  }).filter(Boolean);

  // Pareto frontier line
  const frontierModels = ['kimi-k2.5', 'gemini-3-flash', 'gpt-5.3-codex', 'claude-opus-4-6'];
  const frontierPts = frontierModels.map(m => {
    const entries = fdata.filter(d => d.model === m);
    if (!entries.length) return null;
    return entries.reduce((best, d) => d.score > best.score ? d : best);
  }).filter(Boolean).sort((a, b) => a.cost - b.cost);
  if (frontierPts.length > 1) {
    traces.push({
      x: frontierPts.map(p => p.cost),
      y: frontierPts.map(p => p.score),
      mode: 'lines',
      type: 'scatter',
      line: { color: 'rgba(129,82,236,0.6)', width: 2.5, dash: 'dot' },
      hoverinfo: 'skip',
      showlegend: false,
    });
  }

  const frontierSet = new Set(frontierPts.map(p => p.agent + '|' + p.model));

  // Center dots: purple for frontier, entry color for others
  traces.push({
    x: fdata.map(d => d.cost),
    y: fdata.map(d => d.score),
    mode: 'markers',
    type: 'scatter',
    marker: { size: 5, color: fdata.map(d => frontierSet.has(d.agent + '|' + d.model) ? '#a78bfa' : d.color) },
    hoverinfo: 'skip',
    showlegend: false,
  });

  const vGap = iconY * 1.1;
  const agentIconY = iconY;
  // Agent colors for pill backgrounds
  const AGENT_PILL = {
    'claude-code': 'rgba(217,119,87,0.18)',
    'codex': 'rgba(16,163,127,0.18)',
    'gemini-cli': 'rgba(66,133,244,0.18)',
    'openhands': 'rgba(232,186,58,0.18)',
  };

  function hexToRgba(hex, a) {
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
  }

  // Build pill SVG as data URI
  function pillSvg(color) {
    const strokeColor = color.replace(/[\d.]+\)$/, function(m) { return Math.min(parseFloat(m)*1.8, 0.5).toFixed(2) + ')'; });
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 80"><rect x="2" y="2" width="36" height="76" rx="18" fill="' + color + '" stroke="' + strokeColor + '" stroke-width="1.5"/></svg>';
    return 'data:image/svg+xml;base64,' + btoa(svg);
  }

  const pillW = iconX * 1.6;
  const pillH = iconY * 2.4 + vGap * 0.8;
  const images = [];
  fdata.forEach(d => {
    const orgK = ORG_KEY[d.org];
    const agentK = AGENT_KEY[d.agent];
    let pillColor;
    if (d.agent === 'openhands') {
      const oc = (isLight() && d.org_color === '#FFFFFF') ? '#333344' : d.org_color;
      pillColor = hexToRgba(oc, 0.25);
    } else {
      pillColor = AGENT_PILL[d.agent] || 'rgba(128,128,128,0.15)';
    }

    // Pill background (below icons)
    images.push({
      source: pillSvg(pillColor),
      xref: 'x', yref: 'y',
      x: d.cost, y: d.score,
      sizex: pillW, sizey: pillH,
      xanchor: 'center', yanchor: 'middle',
      layer: 'below',
    });

    // Model org icon (top)
    const orgLK = logoKey(orgK);
    if (orgLK && LOGOS[orgLK]) {
      images.push({
        source: LOGOS[orgLK],
        xref: 'x', yref: 'y',
        x: d.cost, y: d.score + vGap * 0.7,
        sizex: iconX, sizey: iconY,
        xanchor: 'center', yanchor: 'middle',
        layer: 'above',
      });
    }
    // Agent icon (bottom) — scale down CLI agents
    if (agentK && LOGOS[agentK]) {
      const agentScale = (d.agent === 'openhands') ? 1.0 : 0.9;
      images.push({
        source: LOGOS[agentK],
        xref: 'x', yref: 'y',
        x: d.cost, y: d.score - vGap * 0.7,
        sizex: iconX * agentScale, sizey: agentIconY * agentScale,
        xanchor: 'center', yanchor: 'middle',
        layer: 'above',
      });
    }
  });

  const annotations = fdata.map(d => {
    const isLeft = (d.chart_textpos || 'middle right') === 'middle left';
    const onFrontier = frontierSet.has(d.agent + '|' + d.model);
    const label = onFrontier
      ? '<b>' + d.chart_label + '</b><br>(' + d.agent_display + ')'
      : d.chart_label + '<br>(' + d.agent_display + ')';
    return {
      x: d.cost, y: d.score, xref: 'x', yref: 'y',
      text: label,
      showarrow: false,
      xanchor: isLeft ? 'right' : 'left',
      yanchor: 'middle',
      align: isLeft ? 'right' : 'left',
      xshift: isLeft ? -18 : 18,
      font: { size: onFrontier ? 14 : 13, color: onFrontier ? (isLight() ? '#6d28d9' : '#c4b5fd') : tc.text2, family: 'Inter, sans-serif' },
    };
  });

  const layout = {
    paper_bgcolor: tc.paper,
    plot_bgcolor: tc.plot,
    font: { color: tc.text, family: 'Inter, system-ui, sans-serif', size: 12 },
    xaxis: {
      title: { text: 'Average Cost Per Evolution Range (USD)', font: { size: 13 } },
      gridcolor: tc.grid, zerolinecolor: tc.grid,
      tickprefix: '$', tickfont: { size: 11 },
      range: [xMin - xPad, xMax + xPadR],
    },
    yaxis: {
      title: { text: 'Average Score', font: { size: 13 } },
      gridcolor: tc.grid, zerolinecolor: tc.grid,
      ticksuffix: '%', tickfont: { size: 11 },
      range: [yMin - yPad, yMax + yPad],
    },
    images: images,
    annotations: annotations.concat([{
      text: '<a href="https://evo-claw.com/" style="color:#999">https://evo-claw.com</a>',
      xref: 'paper', yref: 'paper', x: 1, y: 0,
      xanchor: 'right', yanchor: 'bottom',
      xshift: 0, yshift: -54,
      showarrow: false,
      font: { size: 15, color: '#999' },
    }]),
    showlegend: false,
    margin: { t: 16, r: 32, b: 56, l: 56 },
    hovermode: 'closest',
    hoverlabel: {
      bgcolor: tc.hover_bg, bordercolor: tc.hover_border,
      font: { family: 'Inter, sans-serif', size: 12, color: tc.text },
    },
  };

  Plotly.newPlot('chart', traces, layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  });

  const chartEl = document.getElementById('chart');
  const initXRange = xMax + xPadR - (xMin - xPad);
  const initYRange = yMax + yPad - (yMin - yPad);
  const BASE_XSHIFT = 18;
  let scaling = false;

  chartEl.on('plotly_relayout', function(ev) {
    if (scaling) return;
    const xl = chartEl.layout.xaxis.range;
    const yl = chartEl.layout.yaxis.range;
    if (!xl || !yl) return;
    const curXRange = xl[1] - xl[0];
    const curYRange = yl[1] - yl[0];
    const scale = Math.sqrt((initXRange / curXRange) * (initYRange / curYRange));
    const newShift = Math.round(BASE_XSHIFT * scale);

    const update = {};
    annotations.forEach((a, i) => {
      update['annotations[' + i + '].xshift'] = a.xanchor === 'right' ? -newShift : newShift;
    });
    scaling = true;
    Plotly.relayout(chartEl, update).then(() => { scaling = false; });
  });
}

// ═══════════════════════════════════════════════════════════
// Leaderboard Table
// ═══════════════════════════════════════════════════════════
const HIGHER = new Set(['score','precision','recall','resolve']);
const LOWER  = new Set(['cost','out_tok_k','time_h','turns']);

function isBest(val, key, fdata) {
  if (HIGHER.has(key)) return Math.abs(val - Math.max(...fdata.map(d => d[key]))) < 0.005;
  if (LOWER.has(key))  return Math.abs(val - Math.min(...fdata.map(d => d[key]))) < 0.005;
  return false;
}
function fmtNum(v, d) { return v.toFixed(d); }

function orgIcon(d) {
  const lk = logoKey(ORG_KEY[d.org]);
  if (lk && LOGOS[lk]) {
    return '<img class="org-logo" src="' + LOGOS[lk] + '" alt="' + d.org + '">';
  }
  return '<span class="org-badge" style="background:' + d.org_color + '">' + d.org.charAt(0) + '</span>';
}
function agentIcon(d) {
  const ak = AGENT_KEY[d.agent];
  if (ak && LOGOS[ak]) {
    const extraStyle = (ak === 'openhands') ? ' style="width:25px;height:25px;margin-left:-1.5px"' : '';
    return '<img class="agent-icon"' + extraStyle + ' src="' + LOGOS[ak] + '" alt="' + d.agent + '">';
  }
  return '';
}

let curKey = 'score', curAsc = false, curClicks = 1;
function defaultAsc(key) {
  if (HIGHER.has(key)) return false;
  if (key === 'rank' || key === 'model_display' || key === 'agent_display') return true;
  return true;
}

function renderTable() {
  const fdata = getFiltered();
  const light = isLight();
  // Re-rank filtered data
  const ranked = [...fdata].sort((a, b) => b.score - a.score);
  ranked.forEach((d, i) => d._rank = i + 1);

  const maxScore = Math.max(...fdata.map(d => d.score));
  function numCell(v, k, dec) {
    const cls = isBest(v, k, fdata) ? 'num best-val' : 'num';
    return '<td class="' + cls + '">' + fmtNum(v, dec) + '</td>';
  }

  const sorted = [...ranked].sort((a, b) => {
    const sk = curKey === 'rank' ? '_rank' : curKey;
    const va = a[sk], vb = b[sk];
    if (typeof va === 'string') return curAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return curAsc ? va - vb : vb - va;
  });

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = sorted.map(d => {
    const r = d._rank;
    let medal = r;
    if (r === 1) medal = '<span class="rank-medal">\uD83E\uDD47</span>';
    else if (r === 2) medal = '<span class="rank-medal">\uD83E\uDD48</span>';
    else if (r === 3) medal = '<span class="rank-medal">\uD83E\uDD49</span>';
    const topCls = r <= 3 ? ' top-' + r : '';
    const barW = (d.score / maxScore * 100).toFixed(1);
    const _abg = (light && d.agent === 'openhands') ? 'rgba(139,117,0,0.15)' : d.agent_bg;
    const _afg = (light && d.agent === 'openhands') ? '#8B7500' : d.agent_fg;
    return '<tr class="' + topCls + '">' +
      '<td class="rank-cell">' + medal + '</td>' +
      '<td class="model-cell">' + orgIcon(d) + d.model_display + '</td>' +
      '<td style="vertical-align:middle"><span class="agent-cell" style="' + (d.agent === 'openhands' ? 'gap:6.5px' : '') + '">' + agentIcon(d) + '<span class="agent-badge" style="background:' + _abg + ';color:' + _afg + '">' + d.agent_display + '</span></span></td>' +
      '<td class="score-cell"><div class="score-bar" style="width:' + barW + '%;background:' + d.agent_fg + '"></div><span class="score-val' + (isBest(d.score,'score',fdata) ? ' best-val' : '') + '">' + fmtNum(d.score, 2) + '</span></td>' +
      numCell(d.precision, 'precision', 2) +
      numCell(d.recall, 'recall', 2) +
      numCell(d.resolve, 'resolve', 2) +
      numCell(d.cost, 'cost', 2) +
      '<td class="num' + (isBest(d.out_tok_k,'out_tok_k',fdata) ? ' best-val' : '') + '">' + d.out_tok_k.toLocaleString() + '</td>' +
      numCell(d.time_h, 'time_h', 2) +
      '<td class="num' + (isBest(d.turns,'turns',fdata) ? ' best-val' : '') + '">' + d.turns.toLocaleString() + '</td>' +
    '</tr>';
  }).join('');
}

// Initial render
renderChart();
renderTable();

// Toggle handler
document.getElementById('officialToggle').addEventListener('change', () => {
  renderChart();
  renderTable();
});

// Sort handler: click 1 = default order, click 2 = reverse, click 3 = reset to rank
document.querySelectorAll('th[data-key]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.key;
    if (key === curKey) {
      curClicks++;
      if (curClicks >= 3) {
        // Reset to default
        curKey = 'score'; curAsc = false; curClicks = 1;
        document.querySelectorAll('th[data-key]').forEach(h => h.classList.remove('sort-asc','sort-desc'));
        document.querySelector('th[data-key="score"]').classList.add('sort-desc');
        renderTable();
        return;
      }
      curAsc = !curAsc;
    } else {
      curKey = key; curAsc = defaultAsc(key); curClicks = 1;
    }
    document.querySelectorAll('th[data-key]').forEach(h => h.classList.remove('sort-asc','sort-desc'));
    th.classList.add(curAsc ? 'sort-asc' : 'sort-desc');
    renderTable();
  });
});

// Tooltip for [data-tip] elements
(function() {
  const tip = document.createElement('div');
  tip.className = 'tip-popup';
  document.body.appendChild(tip);
  document.querySelectorAll('[data-tip]').forEach(el => {
    el.addEventListener('mouseenter', e => {
      const rect = el.getBoundingClientRect();
      tip.textContent = el.dataset.tip;
      tip.style.left = (rect.left + rect.width/2) + 'px';
      tip.style.top = (rect.top - 8) + 'px';
      tip.style.transform = 'translateX(-50%) translateY(-100%)';
      tip.classList.add('show');
    });
    el.addEventListener('mouseleave', () => tip.classList.remove('show'));
  });
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

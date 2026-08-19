"""
Minimal local web server for browsing QA agent run history.
Uses only Python stdlib — no new dependencies.
"""
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

from qa_agent.storage import load_runs

logger = logging.getLogger(__name__)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QA Agent Runs</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }
  a { color: inherit; text-decoration: none; }

  /* ── layout ── */
  .layout { display: flex; flex: 1; overflow: hidden; }
  .sidebar { width: 280px; min-width: 220px; border-right: 1px solid #2d3748;
             overflow-y: auto; display: flex; flex-direction: column; }
  .main { flex: 1; overflow-y: auto; padding: 32px 40px; }

  /* ── sidebar ── */
  .sidebar-header { padding: 20px 16px 12px; font-size: 11px; font-weight: 600;
                    letter-spacing: .08em; color: #718096; text-transform: uppercase; }
  /* scenario folder */
  .scenario-folder { border-bottom: 1px solid #1a202c; }
  .scenario-folder-header { padding: 12px 16px; cursor: pointer; display: flex; align-items: flex-start;
                             gap: 8px; transition: background .12s; }
  .scenario-folder-header:hover { background: #1a202c; }
  .folder-icon { font-size: 12px; color: #4a5568; margin-top: 2px; flex-shrink: 0; transition: transform .15s; }
  .scenario-folder.open .folder-icon { transform: rotate(90deg); }
  .folder-meta { flex: 1; min-width: 0; }
  .folder-id { font-size: 11px; font-family: monospace; color: #718096; margin-bottom: 2px; }
  .folder-purpose { font-size: 13px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .folder-stats { font-size: 11px; color: #718096; margin-top: 4px; }
  .rate-bar { height: 3px; border-radius: 2px; background: #2d3748; margin-top: 5px; }
  .rate-fill { height: 100%; border-radius: 2px; }
  /* run entries inside a folder */
  .run-entries { display: none; background: #0d1117; }
  .scenario-folder.open .run-entries { display: block; }
  .run-item { padding: 9px 16px 9px 36px; cursor: pointer; border-top: 1px solid #1a202c;
              display: flex; align-items: center; gap: 8px; transition: background .12s; }
  .run-item:hover { background: #1a202c; }
  .run-item.active { background: #1e2a3a; border-left: 3px solid #4299e1; padding-left: 33px; }
  .run-iter { font-size: 11px; font-family: monospace; color: #4a5568; flex-shrink: 0; width: 28px; }
  .run-ts   { font-size: 12px; color: #a0aec0; flex: 1; }
  .run-rate-text { font-size: 12px; flex-shrink: 0; }

  /* ── run header ── */
  .run-header { margin-bottom: 28px; }
  .run-header h1 { font-size: 20px; font-weight: 600; margin-bottom: 6px; }
  .run-header .meta { font-size: 13px; color: #a0aec0; }
  .run-header .meta span { margin-right: 16px; }
  .pass-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .badge-pass { background: #1a3a2a; color: #68d391; }
  .badge-fail { background: #3a1a1a; color: #fc8181; }

  /* ── section title ── */
  .section-title { font-size: 11px; font-weight: 600; letter-spacing: .08em; color: #718096;
                   text-transform: uppercase; margin-bottom: 12px; padding-bottom: 8px;
                   border-bottom: 1px solid #2d3748; margin-top: 32px; }
  .section-title:first-child { margin-top: 0; }

  /* ── summary ── */
  .summary-box { background: #1a202c; border-radius: 8px; padding: 16px 20px;
                 font-size: 14px; line-height: 1.7; color: #cbd5e0; }

  /* ── prompt comparison ── */
  .cmp-header { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
  .cmp-stat { font-size: 12px; padding: 3px 10px; border-radius: 10px; font-family: monospace; }
  .cmp-added   { background: #1a3a2a; color: #68d391; }
  .cmp-removed { background: #3a1a1a; color: #fc8181; }
  .cmp-same    { background: #2d3748; color: #a0aec0; }
  .cmp-tabs { display: flex; gap: 2px; margin-bottom: 12px; }
  .cmp-tab { padding: 6px 14px; font-size: 12px; border-radius: 6px 6px 0 0; cursor: pointer;
             background: #1a202c; color: #718096; border: 1px solid #2d3748; border-bottom: none; }
  .cmp-tab.active { background: #141922; color: #e2e8f0; border-color: #4a5568; }
  .cmp-pane { display: none; background: #141922; border: 1px solid #2d3748; border-radius: 0 8px 8px 8px; overflow: hidden; }
  .cmp-pane.active { display: block; }
  /* diff view */
  .diff-view { font-family: monospace; font-size: 12px; line-height: 1.6; overflow-x: auto; max-height: 500px; overflow-y: auto; }
  .diff-line { padding: 1px 12px; white-space: pre; display: block; }
  .diff-add  { background: #0d2a1a; color: #68d391; }
  .diff-del  { background: #2a0d0d; color: #fc8181; }
  .diff-hunk { background: #1a2535; color: #63b3ed; }
  .diff-ctx  { color: #4a5568; }
  /* side-by-side prompt view */
  .prompt-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: #2d3748; max-height: 500px; }
  .prompt-col { background: #141922; overflow-y: auto; padding: 16px; }
  .prompt-col h4 { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
                   color: #718096; margin-bottom: 10px; }
  .prompt-col pre { font-size: 12px; line-height: 1.7; color: #cbd5e0; white-space: pre-wrap; word-break: break-word; }

  /* ── test cases ── */
  .tc-list { display: flex; flex-direction: column; gap: 10px; }
  .tc-card { background: #1a202c; border-radius: 8px; overflow: hidden; }
  .tc-header { display: flex; align-items: center; gap: 10px; padding: 12px 16px;
               cursor: pointer; user-select: none; }
  .tc-header:hover { background: #202836; }
  .tc-status { width: 20px; height: 20px; border-radius: 50%; flex-shrink: 0;
               display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; }
  .status-pass { background: #276749; color: #9ae6b4; }
  .status-fail { background: #742a2a; color: #feb2b2; }
  .tc-id    { font-size: 12px; color: #718096; flex-shrink: 0; font-family: monospace; }
  .tc-desc  { flex: 1; font-size: 14px; font-weight: 500; }
  .tc-cat   { font-size: 11px; color: #a0aec0; background: #2d3748; padding: 2px 8px; border-radius: 10px; flex-shrink: 0; }
  .tc-score { font-size: 12px; font-family: monospace; flex-shrink: 0; width: 36px; text-align: right; }
  .score-pass { color: #68d391; }
  .score-fail { color: #fc8181; }
  .chevron { color: #4a5568; font-size: 12px; transition: transform .2s; }
  .tc-card.open .chevron { transform: rotate(90deg); }
  .tc-detail { padding: 0 16px 16px; border-top: 1px solid #2d3748; display: none; }
  .tc-card.open .tc-detail { display: block; }
  .tc-goal { font-size: 12px; color: #a0aec0; margin: 12px 0 16px; }
  .tc-goal strong { color: #cbd5e0; }

  /* ── turns ── */
  .turn { margin-bottom: 14px; }
  .turn-label { font-size: 11px; font-weight: 600; letter-spacing: .06em;
                text-transform: uppercase; color: #718096; margin-bottom: 6px; }
  .bubble { padding: 10px 14px; border-radius: 8px; font-size: 13px; line-height: 1.6;
            white-space: pre-wrap; word-break: break-word; }
  .bubble-user  { background: #1e3a5f; color: #bee3f8; }
  .bubble-agent { background: #2d3748; color: #e2e8f0; }
  .trace-link { display: inline-flex; align-items: center; gap: 6px; margin-top: 8px;
                font-size: 12px; color: #63b3ed; padding: 4px 10px; border-radius: 6px;
                background: #1a2a3a; border: 1px solid #2c4a6a; transition: background .12s; }
  .trace-link:hover { background: #1e3350; }

  /* ── eval box ── */
  .eval-box { margin-top: 14px; padding: 12px 14px; border-radius: 8px; font-size: 13px; line-height: 1.6; }
  .eval-pass { background: #1a3a2a; border-left: 3px solid #48bb78; }
  .eval-fail { background: #2d1515; border-left: 3px solid #fc8181; }
  .eval-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; }
  .eval-pass .eval-label { color: #68d391; }
  .eval-fail .eval-label { color: #fc8181; }
  .eval-detail { margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,.08);
                 color: #a0aec0; font-size: 12px; }

  /* ── advisor ── */
  .advisor-sections { display: flex; flex-direction: column; gap: 10px; }
  .advisor-section { background: #1a2030; border: 1px solid #2d4a6a; border-radius: 8px; overflow: hidden; }
  .advisor-section-header { padding: 10px 16px; font-size: 11px; font-weight: 600; letter-spacing: .08em;
                            text-transform: uppercase; color: #63b3ed; cursor: pointer;
                            display: flex; justify-content: space-between; align-items: center; }
  .advisor-section-header:hover { background: #1e2d45; }
  .advisor-section-body { padding: 12px 16px; font-size: 13px; line-height: 1.8; color: #cbd5e0;
                          white-space: pre-wrap; border-top: 1px solid #2d4a6a;
                          max-height: 400px; overflow-y: auto; }
  .advisor-section-body.proposed { font-family: monospace; font-size: 12px; background: #141922; }
  .advisor-chevron { color: #4a5568; font-size: 11px; transition: transform .2s; }
  .advisor-section.open .advisor-chevron { transform: rotate(90deg); }
  .advisor-section:not(.open) .advisor-section-body { display: none; }

  /* ── empty ── */
  .empty { padding: 60px 40px; text-align: center; color: #4a5568; }
  .empty h2 { font-size: 18px; margin-bottom: 8px; color: #718096; }
  .empty p { font-size: 14px; }
</style>
</head>
<body>
<div class="layout">
  <div class="sidebar">
    <div class="sidebar-header">Run History</div>
    <div id="run-list"></div>
  </div>
  <div class="main" id="main-panel">
    <div class="empty"><h2>No run selected</h2><p>Select a run from the sidebar to view results.</p></div>
  </div>
</div>

<script>
const SCENARIOS = __RUNS_JSON__;

function passColor(rate) {
  if (rate >= 0.8) return '#48bb78';
  if (rate >= 0.5) return '#ecc94b';
  return '#fc8181';
}
function fmtDate(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return d.toLocaleDateString('en-US', {month:'short', day:'numeric'}) + ' ' +
         d.toLocaleTimeString('en-US', {hour:'2-digit', minute:'2-digit', hour12:false});
}
function fmtDateShort(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return d.toLocaleDateString('en-US', {month:'short', day:'numeric'}) + ' ' +
         d.toLocaleTimeString('en-US', {hour:'2-digit', minute:'2-digit', hour12:false});
}
function escHtml(s) {
  if (!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── sidebar ───────────────────────────────────────────────────────────────────
function renderSidebar() {
  const el = document.getElementById('run-list');
  if (!SCENARIOS.length) {
    el.innerHTML = '<div style="padding:16px;color:#4a5568;font-size:13px">No runs yet.</div>';
    return;
  }
  el.innerHTML = SCENARIOS.map((sc, si) => {
    const latestRun = sc.runs[sc.runs.length - 1];
    const rate = latestRun ? (latestRun.pass_rate ?? 0) : 0;
    const pct  = Math.round(rate * 100);
    const color = passColor(rate);
    const purpose = escHtml((sc.purpose || sc.scenario_id || '').substring(0, 40));
    const scId = escHtml((sc.scenario_id || '').substring(0, 24));

    const runItems = sc.runs.map((r, ri) => {
      const rRate = r.pass_rate ?? 0;
      const rPct  = Math.round(rRate * 100);
      const rColor = passColor(rRate);
      const iter = r.iteration != null ? `#${r.iteration}` : `r${ri+1}`;
      return `<div class="run-item" id="run-${si}-${ri}" onclick="selectRun(${si},${ri})">
        <span class="run-iter">${escHtml(iter)}</span>
        <span class="run-ts">${fmtDateShort(r.timestamp)}</span>
        <span class="run-rate-text" style="color:${rColor}">${rPct}%</span>
      </div>`;
    }).join('');

    return `<div class="scenario-folder" id="sc-${si}">
      <div class="scenario-folder-header" onclick="toggleScenario(${si})">
        <span class="folder-icon">▶</span>
        <div class="folder-meta">
          <div class="folder-id">${scId}</div>
          <div class="folder-purpose">${purpose}</div>
          <div class="folder-stats">${sc.runs.length} run${sc.runs.length !== 1 ? 's' : ''} &middot; latest ${pct}%</div>
          <div class="rate-bar"><div class="rate-fill" style="width:${pct}%;background:${color}"></div></div>
        </div>
      </div>
      <div class="run-entries">${runItems}</div>
    </div>`;
  }).join('');
}

function toggleScenario(si) {
  document.getElementById('sc-' + si).classList.toggle('open');
}

// ── prompt comparison ─────────────────────────────────────────────────────────
function renderPromptComparison(cmp) {
  if (!cmp) return '';
  const added   = cmp.diff_added   ?? 0;
  const removed = cmp.diff_removed ?? 0;
  const same    = added === 0 && removed === 0;

  const statsHtml = same
    ? `<span class="cmp-stat cmp-same">identical prompts</span>`
    : `<span class="cmp-stat cmp-added">+${added} lines</span>
       <span class="cmp-stat cmp-removed">-${removed} lines</span>`;

  const baselineName = escHtml(cmp.baseline_name || cmp.baseline_id || 'Baseline');
  const testedName   = escHtml(cmp.tested_name   || cmp.tested_id   || 'Tested');

  // diff tab
  const diffLines = (cmp.diff_lines || []).map(line => {
    const cls = line.startsWith('+++') || line.startsWith('---') ? 'diff-ctx'
              : line.startsWith('@@')  ? 'diff-hunk'
              : line.startsWith('+')   ? 'diff-add'
              : line.startsWith('-')   ? 'diff-del'
              : 'diff-ctx';
    return `<span class="diff-line ${cls}">${escHtml(line)}</span>`;
  }).join('');
  const diffHtml = diffLines
    ? `<div class="diff-view">${diffLines}</div>`
    : `<div style="padding:20px;color:#4a5568;font-size:13px">No differences found — prompts are identical.</div>`;

  // side-by-side tab
  const sideBySide = `
    <div class="prompt-cols">
      <div class="prompt-col">
        <h4>Baseline &mdash; ${baselineName}</h4>
        <pre>${escHtml(cmp.baseline_prompt || '(no prompt)')}</pre>
      </div>
      <div class="prompt-col">
        <h4>Tested &mdash; ${testedName}</h4>
        <pre>${escHtml(cmp.tested_prompt || '(no prompt)')}</pre>
      </div>
    </div>`;

  return `
    <div class="section-title">Prompt Comparison</div>
    <div class="cmp-header">
      <span style="font-size:13px;color:#a0aec0">${baselineName} &rarr; ${testedName}</span>
      ${statsHtml}
    </div>
    <div class="cmp-tabs">
      <div class="cmp-tab active" onclick="switchCmpTab(this,'diff')">Diff</div>
      <div class="cmp-tab" onclick="switchCmpTab(this,'side')">Side by side</div>
    </div>
    <div class="cmp-pane active" data-pane="diff">${diffHtml}</div>
    <div class="cmp-pane" data-pane="side">${sideBySide}</div>`;
}

function switchCmpTab(tab, pane) {
  const section = tab.closest('.main') || document.getElementById('main-panel');
  section.querySelectorAll('.cmp-tab').forEach(t => t.classList.remove('active'));
  section.querySelectorAll('.cmp-pane').forEach(p => p.classList.remove('active'));
  tab.classList.add('active');
  const target = section.querySelector(`.cmp-pane[data-pane="${pane}"]`);
  if (target) target.classList.add('active');
}

// ── main panel ────────────────────────────────────────────────────────────────
function selectRun(si, ri) {
  document.querySelectorAll('.run-item').forEach(el => el.classList.remove('active'));
  const item = document.getElementById('run-' + si + '-' + ri);
  if (item) item.classList.add('active');
  const r = SCENARIOS[si].runs[ri];
  const panel = document.getElementById('main-panel');
  const total  = (r.results || []).length;
  const passed = (r.results || []).filter(x => x.passed).length;
  const rate = r.pass_rate ?? 0;
  const pct  = Math.round(rate * 100);
  const color = passColor(rate);

  panel.innerHTML = `
    <div class="run-header">
      <h1>${escHtml(r.purpose || r.scenario_id || 'Run')}</h1>
      <div class="meta">
        <span>${fmtDate(r.timestamp)}</span>
        <span style="font-family:monospace;color:#718096">${escHtml(r.scenario_id || '')}</span>
        <span class="pass-badge ${pct === 100 ? 'badge-pass' : 'badge-fail'}" style="color:${color}">${passed}/${total} &middot; ${pct}%</span>
      </div>
    </div>

    ${r.summary ? `<div class="section-title">Summary</div><div class="summary-box">${escHtml(r.summary)}</div>` : ''}

    ${renderPromptComparison(r.prompt_comparison)}

    <div class="section-title">Test Cases</div>
    <div class="tc-list">${renderTestCases(r.results || [])}</div>

    ${r.advice ? `<div class="section-title">Prompt Advisor</div>${renderAdvisor(r.advice)}` : ''}
  `;
}

// ── test cases ────────────────────────────────────────────────────────────────
function renderTestCases(results) {
  return results.map((r, i) => {
    const pass  = r.passed;
    const score = (r.score ?? 0).toFixed(2);
    const turns = (r.turns || []).map((t, ti) => `
      <div class="turn">
        <div class="turn-label">Turn ${ti + 1}</div>
        <div class="bubble bubble-user">${escHtml(t.sent)}</div>
        <div class="bubble bubble-agent" style="margin-top:6px">${escHtml(t.received)}</div>
        ${t.trace_url
          ? `<a class="trace-link" href="${escHtml(t.trace_url)}" target="_blank" rel="noopener">
               <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                 <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/>
                 <polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>
               </svg>Open trace</a>`
          : t.trace_id ? `<span style="font-size:11px;color:#4a5568;margin-top:6px;display:block">Trace: ${escHtml(t.trace_id)}</span>` : ''}
      </div>`).join('');

    return `<div class="tc-card" id="tc-${i}">
      <div class="tc-header" onclick="toggleCard(${i})">
        <div class="tc-status ${pass ? 'status-pass' : 'status-fail'}">${pass ? '✓' : '✗'}</div>
        <span class="tc-id">${escHtml(r.id)}</span>
        <span class="tc-desc">${escHtml(r.description)}</span>
        <span class="tc-cat">${escHtml(r.category)}</span>
        <span class="tc-score ${pass ? 'score-pass' : 'score-fail'}">${score}</span>
        <span class="chevron">▶</span>
      </div>
      <div class="tc-detail">
        <div class="tc-goal"><strong>Goal:</strong> ${escHtml(r.goal || '')}</div>
        ${turns}
        <div class="eval-box ${pass ? 'eval-pass' : 'eval-fail'}">
          <div class="eval-label">${pass ? '✓ Passed' : '✗ Failed'} &middot; score ${score}</div>
          <div>${escHtml(r.rationale)}</div>
          ${r.failure_detail ? `<div class="eval-detail">${escHtml(r.failure_detail)}</div>` : ''}
        </div>
      </div>
    </div>`;
  }).join('');
}

function toggleCard(i) {
  document.getElementById('tc-' + i).classList.toggle('open');
}

// ── advisor ───────────────────────────────────────────────────────────────────
function renderAdvisor(advice) {
  const SECTIONS = [
    { key: 'ASSESSMENT',     label: 'Assessment',     cls: '',         open: true  },
    { key: 'CHANGE ANALYSIS',label: 'Change Analysis',cls: '',         open: true  },
    { key: 'PROPOSED PROMPT',label: 'Proposed Prompt',cls: 'proposed', open: false },
  ];
  // Split advice into sections
  const parts = {};
  let remaining = advice.replace(/^---[ \\t]*\\n?/, '').replace(/\\n?---[ \\t]*$/, '');
  for (let i = 0; i < SECTIONS.length; i++) {
    const key = SECTIONS[i].key;
    const nextKey = i + 1 < SECTIONS.length ? SECTIONS[i + 1].key : null;
    const start = remaining.indexOf(key);
    if (start === -1) continue;
    const after = remaining.slice(start + key.length).replace(/^[ \\t]*\\n/, '');
    const end = nextKey ? after.indexOf(nextKey) : -1;
    parts[key] = (end === -1 ? after : after.slice(0, end)).trim();
  }
  const sections = SECTIONS.filter(s => parts[s.key]);
  if (!sections.length) {
    return `<div class="advisor-section open">
      <div class="advisor-section-header" onclick="this.closest('.advisor-section').classList.toggle('open')">
        Advisor Output <span class="advisor-chevron">▶</span>
      </div>
      <div class="advisor-section-body">${escHtml(advice)}</div>
    </div>`;
  }
  return `<div class="advisor-sections">${sections.map(s => `
    <div class="advisor-section${s.open ? ' open' : ''}">
      <div class="advisor-section-header" onclick="this.closest('.advisor-section').classList.toggle('open')">
        ${s.label} <span class="advisor-chevron">▶</span>
      </div>
      <div class="advisor-section-body ${s.cls}">${escHtml(parts[s.key])}</div>
    </div>`).join('')}
  </div>`;
}

// ── init ──────────────────────────────────────────────────────────────────────
renderSidebar();
// Auto-open the first scenario and select its latest run
if (SCENARIOS.length) {
  document.getElementById('sc-0').classList.add('open');
  const firstRuns = SCENARIOS[0].runs;
  if (firstRuns.length) selectRun(0, firstRuns.length - 1);
}
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    runs_dir: str

    def log_message(self, fmt, *args):
        logger.debug("HTTP %s", fmt % args)

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_response(404)
            self.end_headers()
            return
        runs = load_runs(self.runs_dir)
        # Group runs by scenario_id, oldest-first within each group
        scenarios_map: dict = {}
        for r in reversed(runs):  # load_runs is newest-first; reverse for asc order per scenario
            sid = r.get("scenario_id") or "unknown"
            if sid not in scenarios_map:
                scenarios_map[sid] = {
                    "scenario_id": sid,
                    "purpose": r.get("purpose") or "",
                    "runs": [],
                }
            scenarios_map[sid]["runs"].append(r)
        # Order scenarios by their latest run timestamp (most recent scenario first)
        scenarios = sorted(
            scenarios_map.values(),
            key=lambda s: (s["runs"][-1].get("timestamp") or "") if s["runs"] else "",
            reverse=True,
        )
        scenarios_json = json.dumps(scenarios, ensure_ascii=False)
        # Encode HTML metacharacters so they can never be interpreted as tags
        # when embedded inside a <script> block, regardless of content.
        scenarios_json = (scenarios_json
                          .replace("&", "\\u0026")
                          .replace("<", "\\u003c")
                          .replace(">", "\\u003e"))
        html = _HTML_TEMPLATE.replace("__RUNS_JSON__", scenarios_json)
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(port: int = 8080, runs_dir: str = "runs", quiet: bool = False) -> None:
    handler = _Handler
    handler.runs_dir = runs_dir
    server = HTTPServer(("", port), handler)
    if not quiet:
        print(f"QA Agent dashboard → http://localhost:{port}")
        print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

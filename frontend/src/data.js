/** Contract-aware, read-only parsing of existing Python publication artifacts. */
export const FILES = Object.freeze({
  analysis: 'analysis.json',
  history: 'historical_decisions.json',
  shadow: 'v3_challenger.json',
  shadowHistory: 'v3_challenger_history.csv',
  version: 'version.json',
});

export function finite(value) {
  if (value === '' || value === null || value === undefined || value === 'NA') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
export function isoDate(value) {
  if (!value) return null;
  const str = String(value).slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(str) && !Number.isNaN(Date.parse(`${str}T00:00:00Z`)) ? str : null;
}
export function pct(value, digits = 1) {
  const n = finite(value);
  return n === null ? '—' : `${(n * 100).toFixed(digits)}%`;
}
export function valueFmt(value, digits = 1) {
  const n = finite(value);
  return n === null ? '—' : n.toFixed(digits);
}
export function signedPct(value, digits = 1) {
  const n = finite(value);
  return n === null ? '—' : `${n > 0 ? '+' : ''}${(n * 100).toFixed(digits)}%`;
}
export function groupSentiment(value) {
  const n = finite(value);
  if (n === null || n < 0 || n > 100) return { label: 'Unavailable', tone: 'neutral' };
  if (n < 25) return { label: 'Extreme fear', tone: 'fear' };
  if (n < 45) return { label: 'Fear', tone: 'fear' };
  if (n <= 55) return { label: 'Neutral', tone: 'neutral' };
  if (n <= 75) return { label: 'Greed', tone: 'greed' };
  return { label: 'Extreme greed', tone: 'greed' };
}
export function daysSince(date, now = new Date()) {
  const datePart = isoDate(date);
  if (!datePart) return null;
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  const observed = Date.parse(`${datePart}T00:00:00Z`);
  return Math.max(0, Math.round((today - observed) / 86400000));
}
export function freshness(analysis, now = new Date()) {
  const source = isoDate(analysis?.latest?.signal_date);
  const age = daysSince(source, now);
  return { source, age, stale: age === null || age > 4, label: age === null ? 'Data unavailable' : age > 4 ? `Stale · ${age} days old` : 'Latest published observation' };
}
export function assertAnalysis(value) {
  if (!value || typeof value !== 'object') throw new Error('Missing production analysis');
  if (!isoDate(value.latest?.signal_date) || finite(value.latest?.fear_greed) === null) {
    throw new Error('Production analysis has an invalid latest observation');
  }
  if (!value.verdict || typeof value.verdict !== 'object') throw new Error('Production verdict is missing');
  if (!Array.isArray(value.warnings)) throw new Error('Production warnings must be an array');
  return value;
}
export function assertHistory(value) {
  if (!value || !Array.isArray(value.decisions)) throw new Error('Historical decisions payload is missing');
  if (!value.decisions.every(row => isoDate(row.decision_date))) throw new Error('Historical dates are invalid');
  return value;
}
export function inspectShadow(value) {
  if (!value) return { status: 'UNAVAILABLE', safe: false, reason: 'Research snapshot unavailable' };
  const guards = value.guardrails || {};
  const safe = value.mode === 'RESEARCH_ONLY' && value.production_effect === 'NONE' &&
    guards.production_action_changed === false && guards.champion_selected === false &&
    guards.v3_019_eligible === false && guards.evid001_outcomes_opened === false &&
    finite(guards.sizing_multiplier) === 1;
  return { status: safe ? 'RESEARCH_ONLY' : 'INVALID', safe, reason: safe ? null : 'Research isolation contract could not be confirmed' };
}
export function normalizeHistoryRows(rows) {
  return rows.filter(row => isoDate(row.decision_date)).map(row => ({
    ...row,
    date: isoDate(row.decision_date),
    sentiment: finite(row.fear_greed),
    forward5: finite(row.forward_5d),
    forward20: finite(row.forward_20d),
    return20: finite(row.market_return_20d),
    drawdown20: finite(row.max_drawdown_20d),
    actionName: String(row.action || 'Unknown'),
    timingName: String(row.timing_action || 'Unknown'),
    regimeName: String(row.market_regime || 'Unknown'),
  })).sort((a, b) => a.date.localeCompare(b.date));
}
export function filterHistory(rows, opts = {}) {
  const q = String(opts.search || '').toLowerCase().trim();
  const result = rows.filter(row => {
    if (opts.from && row.date < opts.from) return false;
    if (opts.to && row.date > opts.to) return false;
    if (opts.action && opts.action !== 'all' && row.actionName !== opts.action) return false;
    if (opts.regime && opts.regime !== 'all' && row.regimeName !== opts.regime) return false;
    if (q && ![row.date, row.actionName, row.timingName, row.regimeName].some(x => String(x).toLowerCase().includes(q))) return false;
    return true;
  });
  return opts.direction === 'asc' ? result : [...result].reverse();
}
export function summaryOf(rows) {
  const observed = rows.map(r => r.forward5).filter(v => v !== null);
  const n = observed.length;
  const ordered = [...observed].sort((a, b) => a - b);
  return {
    count: rows.length,
    mature: n,
    positiveRate: n ? observed.filter(v => v > 0).length / n : null,
    avg: n ? observed.reduce((a, b) => a + b, 0) / n : null,
    median: n ? (ordered[Math.floor((n - 1) / 2)] + ordered[Math.floor(n / 2)]) / 2 : null,
  };
}
export function parseCsv(source) {
  const records = []; let row = []; let cell = ''; let quoted = false;
  for (let i = 0; i < source.length; i++) {
    const ch = source[i];
    if (ch === '"') { if (quoted && source[i+1] === '"') { cell += '"'; i++; } else quoted = !quoted; }
    else if (ch === ',' && !quoted) { row.push(cell); cell = ''; }
    else if ((ch === '\n' || ch === '\r') && !quoted) {
      if (ch === '\r' && source[i+1] === '\n') i++;
      row.push(cell); if (row.some(c => c.length)) records.push(row); row = []; cell = '';
    } else cell += ch;
  }
  if (cell || row.length) { row.push(cell); records.push(row); }
  if (quoted) throw new Error('Malformed CSV: unterminated quotation');
  const [header = [], ...body] = records;
  if (new Set(header).size !== header.length) throw new Error('Duplicate CSV columns');
  return body.map(r => Object.fromEntries(header.map((key, i) => [key, r[i] ?? ''])));
}
export function csvEscape(v) {
  const s = String(v ?? '');
  // CSV formula injection protection when exporting user-selectable records.
  const numeric = /^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$/.test(s);
  const escaped = !numeric && /^[\s]*[=+\-@]/.test(s) ? `'${s}` : s;
  return /[",\r\n]/.test(escaped) ? `"${escaped.replaceAll('"', '""')}"` : escaped;
}
export function exportCsv(rows) {
  const cols = ['decision_date','fear_greed','market_regime','action','timing_action','forward_5d','forward_20d','max_drawdown_20d'];
  return [cols.join(','), ...rows.map(r => cols.map(k => csvEscape(r[k])).join(','))].join('\r\n') + '\r\n';
}
export async function loadDashboard(signal, fetcher = fetch) {
  const get = async (name, required = true, text = false) => {
    try {
      const response = await fetcher(`./${name}?v=1`, { signal, cache: 'no-store' });
      if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
      return text ? response.text() : response.json();
    } catch (error) { if (required || error?.name === 'AbortError') throw error; return null; }
  };
  const [analysis, history, version, shadow, csv] = await Promise.all([
    get(FILES.analysis), get(FILES.history), get(FILES.version),
    get(FILES.shadow, false), get(FILES.shadowHistory, false, true),
  ]);
  return {
    analysis: assertAnalysis(analysis),
    history: assertHistory(history),
    version,
    shadow,
    shadowHistory: csv ? parseCsv(csv) : [],
  };
}

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { assertAnalysis, assertHistory, inspectShadow, filterHistory, normalizeHistoryRows, summaryOf, freshness, parseCsv, exportCsv, groupSentiment, finite } from '../src/data.js';

test('fails closed on malformed production analysis', () => {
  assert.throws(() => assertAnalysis({}), /invalid latest/);
  assert.equal(assertAnalysis({ latest: { signal_date: '2026-10-09', fear_greed: 30 }, verdict: {}, warnings: [] }).latest.fear_greed, 30);
  assert.throws(() => assertHistory({decisions: [{decision_date: 'garbage'}]}), /invalid/);
});
test('shadow must prove all research-only guards', () => {
  const original = { mode: 'RESEARCH_ONLY', production_effect: 'NONE', guardrails: { production_action_changed: false, champion_selected: false, v3_019_eligible: false, evid001_outcomes_opened: false, sizing_multiplier: 1 } };
  assert.equal(inspectShadow(original).safe, true);
  assert.equal(inspectShadow({...original, guardrails: {...original.guardrails, champion_selected: true}}).safe, false);
  assert.equal(inspectShadow(null).safe, false);
});
test('historical filters and mature-sample denominators', () => {
  const rows = normalizeHistoryRows([{ decision_date: '2026-01-01', action: 'A', market_regime:'x', forward_5d: 0.05, fear_greed: 20 },{ decision_date:'2026-01-02',action:'B',market_regime:'y',forward_5d:null,fear_greed:45 },{decision_date:'2026-01-03',action:'A',market_regime:'x',forward_5d:-0.02, fear_greed:80}]);
  const filtered = filterHistory(rows,{action:'A', from:'2026-01-02'});
  assert.equal(filtered.length,1); assert.equal(filtered[0].date,'2026-01-03');
  const stats = summaryOf(rows); assert.equal(stats.count,3); assert.equal(stats.mature,2); assert.equal(stats.positiveRate,.5);
});
test('freshness is never described as live', () => {
  const x={latest:{signal_date:'2026-09-01'}};
  assert.equal(freshness(x, new Date('2026-10-09T12:00:00Z')).stale,true);
  assert.equal(groupSentiment(18).label,'Extreme fear'); assert.equal(finite('NA'),null);
});
test('CSV parser preserves escaped text; exports protect spreadsheet formula injection', () => {
  assert.deepEqual(parseCsv('date,value\n"2026-01-01","a,b"\n'),[{date:'2026-01-01',value:'a,b'}]);
  assert.match(exportCsv([{decision_date:'2026-01-01',action:'=IMPORTXML("evil")'}]), /'=IMPORTXML/);
});

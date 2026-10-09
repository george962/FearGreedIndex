import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, ArrowDownRight, ArrowRight, ArrowUpRight, BarChart3, BookOpen,
  CalendarDays, CheckCircle2, ChevronDown, CircleAlert, Clock3, Database,
  Download, ExternalLink, FileCheck2, Github, Globe2, Info, Layers3, LineChart,
  Menu, Moon, RefreshCw, Search, ShieldCheck, SlidersHorizontal, Sparkles,
  Sun, Table2, TrendingUp, X, Zap,
} from 'lucide-react';
import {
  Area, AreaChart, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import {
  loadDashboard, finite, isoDate, pct, signedPct, valueFmt, groupSentiment,
  freshness, normalizeHistoryRows, filterHistory, summaryOf, inspectShadow, exportCsv,
} from './data.js';
import './styles.css';

const REPO = 'https://github.com/george962/FearGreedIndex';
const SECTIONS = [
  { key: 'overview', name: 'Overview', icon: BarChart3, hint: 'Market conditions & context' },
  { key: 'explorer', name: 'Historical Explorer', icon: LineChart, hint: 'Point-in-time decision history' },
  { key: 'strategy', name: 'Strategy Evidence', icon: Layers3, hint: 'Rule-based diagnostic results' },
  { key: 'research', name: 'Research Lab', icon: BookOpen, hint: 'Immutable V3 shadow evidence' },
];
const periodOptions = [{key:'1M',n:30},{key:'3M',n:90},{key:'1Y',n:365},{key:'ALL',n:0}];
const monthDay = d => d ? new Date(`${d}T12:00:00Z`).toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'}) : '—';
const dateLong = d => d ? new Date(`${d}T12:00:00Z`).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}) : '—';
const classOf = v => finite(v) === null ? '' : finite(v) >= 0 ? 'positive' : 'negative';
const proper = s => String(s || 'Unavailable').replace(/_/g,' ').replace(/\b\w/g, x=>x.toUpperCase());
function useHashRoute() {
  const parse = () => {
    const page = window.location.hash.replace(/^#\/?/, '').split('/')[0];
    return SECTIONS.some(s=>s.key === page) ? page : 'overview';
  };
  const [page, setPage] = useState(parse);
  useEffect(() => { const fn = () => setPage(parse()); window.addEventListener('hashchange',fn); return () => window.removeEventListener('hashchange',fn); }, []);
  return [page, next => { window.location.hash = `/${next}`; setPage(next); window.scrollTo({top:0,behavior:'smooth'}); }];
}
function useDashboard() {
  const [state, setState] = useState({ status: 'loading', data:null, error:null });
  const [refreshKey, setRefreshKey] = useState(0);
  useEffect(() => {
    const abort = new AbortController();
    setState(x=>({...x,status:x.data ? 'refreshing':'loading',error:null}));
    loadDashboard(abort.signal).then(data => {
      setState({status:'ready',data,error:null});
    }).catch(error => {
      if (error?.name !== 'AbortError') setState(x=>({...x,status:'error',error:String(error.message || error)}));
    });
    return () => abort.abort();
  },[refreshKey]);
  return [state, () => setRefreshKey(k=>k+1)];
}
function useVersionPoll(onUpdate) {
  const [available,setAvailable] = useState(false);
  useEffect(() => {
    if (!onUpdate) return;
    let active=true;
    const fn=async()=>{try{
      const r=await fetch(`./version.json?check=${Date.now()}`,{cache:'no-store'});
      if(!r.ok) return;
      const next=await r.json();
      if(active && next.build_id && next.build_id!==onUpdate) setAvailable(true);
    }catch{/* Use existing published data if update checks fail. */}};
    const timer=setInterval(fn,300000);
    return()=>{active=false;clearInterval(timer);};
  },[onUpdate]);
  return available;
}
function Panel({children,className='',...rest}){return <section className={`panel ${className}`} {...rest}>{children}</section>;}
function Pill({children,tone='neutral',icon:Icon}){return <span className={`pill pill-${tone}`}>{Icon && <Icon size={12} aria-hidden="true"/>}{children}</span>;}
function Metric({label,value,note,icon:Icon,delta,tone}){return <div className="metric-card">
  <div className="metric-top"><span>{label}</span>{Icon&&<Icon size={18} aria-hidden="true"/>}</div>
  <div className="metric-value">{value}</div>
  <div className="metric-foot">{delta!==undefined ? <span className={`metric-delta ${tone||classOf(delta)}`}>{signedPct(delta)}</span>:null}{note}</div>
</div>}
function SectionHead({eyebrow,title,description,aside}){return <div className="section-head">
  <div><div className="overline">{eyebrow}</div><h2>{title}</h2>{description&&<p>{description}</p>}</div>{aside&&<div className="section-aside">{aside}</div>}
</div>}
function ChartTooltip({active,payload,label}){if(!active||!payload?.length)return null;return <div className="chart-tooltip"><strong>{dateLong(label)}</strong>{payload.filter(x=>x.value!==null && x.value!==undefined).map((entry,i)=><div key={i}><span className="legend-dot" style={{background:entry.color}}/>{entry.name}: <b>{entry.name.toLowerCase().includes('return')||entry.name.toLowerCase().includes('percentile') ? (entry.name.toLowerCase().includes('percentile')?valueFmt(entry.value,2):signedPct(entry.value)) : valueFmt(entry.value,1)}</b></div>)}</div>}
function SentimentBadge({value}){const item=groupSentiment(value);return <Pill tone={item.tone}>{item.label}</Pill>}
function ProgressGauge({score}){const n=finite(score);return <div className="gauge-layout">
  <div className="gauge-ring" style={{'--pct': `${n===null ? 0 : Math.min(100,Math.max(0,n))}%`}} role="img" aria-label={n===null?'Sentiment unavailable':`Fear and Greed ${n.toFixed(1)} out of 100`}>
    <div className="gauge-inside"><div className="gauge-score">{n===null?'—':n.toFixed(1)}</div><span>out of 100</span></div>
  </div><div className="gauge-scale"><span>0 · Extreme fear</span><span>100 · Extreme greed</span></div>
</div>}
function EmptyChart({message='No compatible observations in this range.'}){return <div className="empty-chart"><BarChart3 size={27}/><span>{message}</span></div>}
function SentimentChart({rows}){const [period,setPeriod]=useState('3M');
  const items=useMemo(()=>{const p=periodOptions.find(p=>p.key===period); if(!rows.length)return [];
    const last=Date.parse(`${rows[rows.length-1].date}T00:00:00Z`);
    const cutoff=p.n>0?last-p.n*86400000:-Infinity;
    const data=rows.filter(x=>Date.parse(`${x.date}T00:00:00Z`)>=cutoff && x.sentiment!==null).map(x=>({date:x.date,sentiment:x.sentiment,return20:x.return20}));
    // Bound SVG points on long histories; chart remains descriptive, not model recomputation.
    const step=Math.max(1,Math.ceil(data.length/450));return data.filter((x,i)=>i%step===0||i===data.length-1);
  },[rows,period]);
  return <Panel className="chart-panel"><div className="panel-title-row"><div><div className="overline">SENTIMENT TREND</div><h3>Fear & Greed over time</h3><p>Published historical observations · 0–100 scale</p></div><div className="segments" role="group" aria-label="Chart period">{periodOptions.map(x=><button key={x.key} type="button" aria-pressed={period===x.key} className={period===x.key?'selected':''} onClick={()=>setPeriod(x.key)}>{x.key}</button>)}</div></div>
    {items.length===0?<EmptyChart/>:<div className="chart-area" role="img" aria-label={`Fear and Greed sentiment time series for ${period}`}><ResponsiveContainer width="100%" height="100%"><AreaChart data={items} margin={{top:16,right:18,bottom:5,left:-19}}><defs><linearGradient id="fgi-trend-gradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#1D8B78" stopOpacity=".22"/><stop offset="95%" stopColor="#1D8B78" stopOpacity="0"/></linearGradient></defs><CartesianGrid vertical={false} stroke="#e8edf0" strokeDasharray="4 5"/><XAxis dataKey="date" tickFormatter={monthDay} minTickGap={34} axisLine={false} tickLine={false} tick={{fill:'#83909B',fontSize:11}}/><YAxis domain={[0,100]} ticks={[0,25,50,75,100]} axisLine={false} tickLine={false} tick={{fill:'#83909B',fontSize:11}}/><Tooltip content={<ChartTooltip/>}/><ReferenceLine y={50} stroke="#c5d3d9" strokeDasharray="4 5"/><Area name="Sentiment" type="monotone" dataKey="sentiment" stroke="#197e71" strokeWidth={2.7} fill="url(#fgi-trend-gradient)" dot={false} activeDot={{r:5,strokeWidth:2,stroke:'#fff'}} connectNulls={false}/></AreaChart></ResponsiveContainer></div>}
    <div className="chart-footer"><div><i className="legend-dot" style={{background:'#197e71'}}/>Fear & Greed index</div><span>{items.length} plotted observations · chart is descriptive only</span></div></Panel>
}
function RiskBar({warnings,stale}){if(!warnings?.length&&!stale)return null;return <div className="notice" role="status"><CircleAlert size={19} aria-hidden="true"/><div><strong>Important context</strong><span>{stale?'This dashboard contains historical/published data; the latest sentiment observation may be stale. ':''}{warnings?.[0]||''}</span>{warnings?.length>1&&<details><summary>{warnings.length-1} more data notes</summary><ul>{warnings.slice(1).map((w,i)=><li key={i}>{w}</li>)}</ul></details>}</div></div>}
function Overview({data,rows,onNavigate}){
 const {analysis,shadow}=data;const latest=analysis.latest;const verdict=analysis.verdict||{};const timing=analysis.fast_timing||{};const fg=freshness(analysis);const inspector=inspectShadow(shadow);const note=(analysis.warnings||[]).length;
 const metric=finite(latest.market_return_20d);
 return <div className="page-body">
  <div className="page-heading"><div><div className="heading-kicker"><Sparkles size={15}/> THE BIG PICTURE</div><h1>Market sentiment, in context.</h1><p>An evidence-first view of market emotion, price behavior, and historical research. No hype, just the data.</p></div><div className="heading-date"><CalendarDays size={15}/><span>Observation {dateLong(latest.signal_date)}</span></div></div>
  <RiskBar warnings={analysis.warnings} stale={fg.stale}/>
  <div className="hero-grid"><Panel className="sentiment-panel"><div className="panel-kicker"><span>MARKET SENTIMENT INDEX</span><SentimentBadge value={latest.fear_greed}/></div><ProgressGauge score={latest.fear_greed}/><div className="sentiment-meta"><div><span>Latest observation</span><b>{dateLong(latest.signal_date)}</b></div><div><span>5-observation change</span><b className={classOf(latest.fg_change_5)}>{finite(latest.fg_change_5)===null?'—':`${finite(latest.fg_change_5)>0?'+':''}${valueFmt(latest.fg_change_5)}`}</b></div></div></Panel>
   <Panel className="stance-panel"><div className="panel-kicker"><span>FROZEN v2.1 RESEARCH SIGNAL</span><Pill tone="neutral" icon={ShieldCheck}>Rule-based</Pill></div><h2 className="stance-title">{verdict.action||'Unavailable'}</h2><p className="stance-description">{verdict.rationale||'The model did not publish an explanation for this decision.'}</p><div className="stance-stats"><div><span>Confidence</span><b>{String(verdict.confidence??'—')}</b></div><div><span>Historical analogs</span><b>{verdict.sample_size??'—'}</b></div><div><span>Market regime</span><b>{proper(latest.market_regime)}</b></div></div><div className="stance-disclaimer"><ShieldCheck size={16}/>Historical diagnostics, not personalized trading advice. Tactical sizing remains disabled.</div></Panel>
  </div>
  <div className="metric-grid"><Metric label="20-day market return" value={signedPct(metric)} note="Trailing market performance" icon={TrendingUp} tone={classOf(metric)}/><Metric label="From 252-day high" value={signedPct(latest.distance_from_252d_high)} note="Market drawdown context" icon={ArrowDownRight}/><Metric label="Distance from 200D average" value={signedPct(latest.distance_from_sma_200)} note="Long-term price extension" icon={Activity}/><Metric label="Historical decisions" value={(analysis.historical_decisions?.total??rows.length).toLocaleString()} note="Point-in-time replay dates" icon={Database}/></div>
  <div className="overview-lower"><SentimentChart rows={rows}/><div className="stack"><Panel className="inside-panel"><div className="overline">FAST TIMING LAYER</div><div className="inline-heading"><Zap size={20}/><h3>{timing.action || 'No signal available'}</h3></div><p>{timing.rationale || 'No timing explanation provided.'}</p><div className="facts"><div><span>Recommendation</span><strong>{timing.recommendation||'—'}</strong></div><div><span>Confirmations</span><strong>{timing.confirmation_count??'—'} / {timing.confirmation_total??'—'}</strong></div></div></Panel><Panel className="inside-panel"><div className="overline">V3 EXPERIMENTAL RESEARCH</div><div className="inline-heading"><Layers3 size={20}/><h3>Shadow model</h3></div><Pill tone={inspector.safe?'info':'warning'} icon={ShieldCheck}>{inspector.safe?'Research only':'Verification unavailable'}</Pill><p>{inspector.safe?`Method ${shadow?.method_id||'STAB-004'} remains unpromoted and cannot change production actions.`:'Shadow evidence cannot be safely displayed without the isolation contract.'}</p><button type="button" className="inline-link" onClick={()=>onNavigate('research')}>Explore research <ArrowRight size={16}/></button></Panel></div></div>
  <Panel className="method-strip"><div><ShieldCheck size={20}/><b>How to read this dashboard</b><p>Historical similarity is not predictive proof. Returns are shown only where the observation window has matured.</p></div><button type="button" onClick={()=>onNavigate('explorer')}>Explore the evidence <ArrowRight size={16}/></button></Panel>
 </div>;
}
function FilterSelect({label,value,onChange,values}){return <label className="filter-label">{label}<select value={value} onChange={e=>onChange(e.target.value)}><option value="all">All {label.toLowerCase()}s</option>{values.map(v=><option key={v} value={v}>{proper(v)}</option>)}</select></label>}
function downloadCsv(content,name){const blob=new Blob([content],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function Explorer({data,rows}){
  const [search,setSearch]=useState(''),[from,setFrom]=useState(''),[to,setTo]=useState(''),[action,setAction]=useState('all'),[regime,setRegime]=useState('all'),[direction,setDirection]=useState('desc'),[page,setPage]=useState(1);
  const actions=useMemo(()=>[...new Set(rows.map(x=>x.actionName))].sort(),[rows]);const regimes=useMemo(()=>[...new Set(rows.map(x=>x.regimeName))].sort(),[rows]);
  const filtered=useMemo(()=>filterHistory(rows,{search,from,to,action,regime,direction}),[rows,search,from,to,action,regime,direction]);
  const summary=useMemo(()=>summaryOf(filtered),[filtered]);const maxPage=Math.max(1,Math.ceil(filtered.length/20));const safePage=Math.min(page,maxPage);const slice=filtered.slice((safePage-1)*20,safePage*20);
  const reset=()=>{setSearch('');setFrom('');setTo('');setAction('all');setRegime('all');setDirection('desc');setPage(1)};
  const chart=useMemo(()=>{const step=Math.max(1,Math.ceil(filtered.length/130));return [...filtered].reverse().filter((x,i)=>i%step===0||i===filtered.length-1).map(x=>({date:x.date,forward5:x.forward5,return20:x.return20}));},[filtered]);
  return <div className="page-body"><div className="page-heading"><div><div className="heading-kicker"><SlidersHorizontal size={15}/> HISTORICAL EXPLORER</div><h1>Explore the evidence.</h1><p>Filter recorded point-in-time decisions. Explore only; filters do not retrain or validate the underlying strategy.</p></div><button type="button" className="button primary" onClick={()=>downloadCsv(exportCsv(filtered),`fgi-historical-${new Date().toISOString().slice(0,10)}.csv`)} disabled={!filtered.length}><Download size={16}/> Export selection</button></div>
  <Panel className="filter-panel"><div className="filter-panel-title"><SlidersHorizontal size={17}/><strong>Refine observations</strong><span>{filtered.length.toLocaleString()} of {rows.length.toLocaleString()} dates</span></div><div className="filters"><label className="filter-label search-label">Search<div className="search-input"><Search size={16}/><input value={search} onChange={e=>{setSearch(e.target.value);setPage(1)}} placeholder="Date, action, regime..."/></div></label><label className="filter-label">From<input aria-label="From date" type="date" value={from} max={to||undefined} onChange={e=>{setFrom(e.target.value);setPage(1)}}/></label><label className="filter-label">To<input aria-label="To date" type="date" value={to} min={from||undefined} onChange={e=>{setTo(e.target.value);setPage(1)}}/></label><FilterSelect label="Action" value={action} onChange={x=>{setAction(x);setPage(1)}} values={actions}/><FilterSelect label="Regime" value={regime} onChange={x=>{setRegime(x);setPage(1)}} values={regimes}/></div><div className="filter-bottom"><span>Observation filters are descriptive; future outcomes appear only when known.</span><button type="button" onClick={reset}>Reset filters</button></div></Panel>
  <div className="metric-grid"><Metric label="Matched dates" value={summary.count.toLocaleString()} note="After filters" icon={CalendarDays}/><Metric label="Mature 5D samples" value={summary.mature.toLocaleString()} note="Excluded missing outcomes" icon={FileCheck2}/><Metric label="5-day positive frequency" value={pct(summary.positiveRate)} note="Historical, not predictive" icon={TrendingUp}/><Metric label="Average realized 5D" value={signedPct(summary.avg)} note="Not an investment forecast" icon={Activity}/></div>
  <Panel className="chart-panel"><div className="panel-title-row"><div><div className="overline">OUTCOME DISTRIBUTION OVER TIME</div><h3>Historical 5-day outcomes</h3><p>Realized outcomes on selected decision dates; immature rows are omitted from the series.</p></div><Pill icon={Info}>Selection only</Pill></div>{chart.some(x=>x.forward5!==null)?<div className="chart-area"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={chart} margin={{top:18,right:12,left:0,bottom:0}}><CartesianGrid stroke="#e8edf0" vertical={false} strokeDasharray="3 5"/><XAxis dataKey="date" tickFormatter={monthDay} minTickGap={35} tick={{fontSize:11,fill:'#85909B'}} axisLine={false} tickLine={false}/><YAxis tickFormatter={v=>`${(v*100).toFixed(0)}%`} width={45} tick={{fontSize:11,fill:'#85909B'}} axisLine={false} tickLine={false}/><ReferenceLine y={0} stroke="#a9bac3"/><Tooltip content={<ChartTooltip/>}/><Line type="monotone" name="5D return" dataKey="forward5" stroke="#197e71" strokeWidth={2} dot={false} connectNulls={false}/></ComposedChart></ResponsiveContainer></div>:<EmptyChart message="No matured outcomes match your current filters."/>}</Panel>
  <Panel className="table-panel"><div className="panel-title-row"><div><div className="overline">REPLAY RECORDS</div><h3>Historical decisions</h3><p>Each row represents the original decision-date record.</p></div><label className="compact-select">Sort <select value={direction} onChange={e=>{setDirection(e.target.value);setPage(1)}}><option value="desc">Newest first</option><option value="asc">Oldest first</option></select></label></div><div className="table-scroll"><table><thead><tr><th scope="col">Decision date</th><th scope="col">Sentiment</th><th scope="col">Action</th><th scope="col">Market regime</th><th scope="col">5D outcome</th><th scope="col">20D outcome</th></tr></thead><tbody>{slice.map(x=><tr key={x.date}><td className="td-strong">{dateLong(x.date)}</td><td><span className="sentiment-cell"><span className="small-dot"/>{valueFmt(x.sentiment)}</span></td><td><span className="action-cell">{x.actionName}</span></td><td>{proper(x.regimeName)}</td><td className={classOf(x.forward5)}>{signedPct(x.forward5)}</td><td className={classOf(x.forward20)}>{signedPct(x.forward20)}</td></tr>)}{!slice.length&&<tr><td colSpan="6"><div className="table-empty">No observations match these filters. Try another range.</div></td></tr>}</tbody></table></div><div className="pagination"><span>Page {safePage} of {maxPage} · {filtered.length} matching dates</span><div><button type="button" disabled={safePage===1} onClick={()=>setPage(n=>Math.max(1,n-1))}>Previous</button><button type="button" disabled={safePage>=maxPage} onClick={()=>setPage(n=>Math.min(maxPage,n+1))}>Next <ArrowRight size={15}/></button></div></div></Panel>
  <p className="footnote">The 5D and 20D outcome columns are decimal returns from the source replay, formatted as percentages; unavailable and not-yet-matured values remain blank (—). These summaries are not independent out-of-sample performance claims.</p>
 </div>;
}
function Strategy({data,rows}){
  const {analysis}=data;const verdict=analysis.verdict||{},guidance=analysis.position_guidance||{},timing=analysis.fast_timing||{};
  const stats=summaryOf(rows);const summary=analysis.historical_decisions||{};
  const checks=Array.isArray(verdict.decision_checks)?verdict.decision_checks:[];
  return <div className="page-body"><div className="page-heading"><div><div className="heading-kicker"><Layers3 size={15}/> STRATEGY EVIDENCE</div><h1>Signals with their caveats.</h1><p>A transparent view of the frozen operational strategy, sample support, and historical diagnostics.</p></div><Pill tone="info" icon={ShieldCheck}>v2.1 frozen baseline</Pill></div>
   <RiskBar warnings={analysis.warnings} stale={freshness(analysis).stale}/>
   <div className="strategy-grid"><Panel><div className="overline">CONFIRMATION MODEL</div><h2 className="strategy-action">{verdict.action||'Unavailable'}</h2><p>{verdict.rationale||'No rationale published.'}</p><div className="facts"><div><span>Method</span><strong>{verdict.analog_method||'—'}</strong></div><div><span>Regime</span><strong>{proper(analysis.latest?.market_regime)}</strong></div><div><span>Confidence</span><strong>{verdict.confidence||'—'}</strong></div><div><span>Analog count</span><strong>{verdict.sample_size??'—'}</strong></div><div><span>Required sample</span><strong>{verdict.required_sample??'—'}</strong></div></div></Panel>
   <Panel><div className="overline">FASTER TIMING MODEL</div><h2 className="strategy-action">{timing.action||'Unavailable'}</h2><p>{timing.rationale||'No rationale published.'}</p><div className="facts"><div><span>Recommendation</span><strong>{timing.recommendation||'—'}</strong></div><div><span>Side</span><strong>{timing.side||'—'}</strong></div><div><span>Confirmations</span><strong>{timing.confirmation_count??'—'} / {timing.confirmation_total??'—'}</strong></div><div><span>Position guidance</span><strong>{guidance.sizing_label||'1.00× baseline'}</strong></div></div></Panel></div>
  <div className="metric-grid"><Metric label="All replay decisions" value={summary.total??rows.length} note="Point-in-time records" icon={CalendarDays}/><Metric label="Mature outcomes" value={stats.mature.toLocaleString()} note="From historical replay" icon={FileCheck2}/><Metric label="5D positive frequency" value={pct(stats.positiveRate)} note="All mature replay outcomes" icon={TrendingUp}/><Metric label="Mean historical 5D return" value={signedPct(stats.avg)} note="Descriptive; not return guidance" icon={BarChart3}/></div>
  <Panel><SectionHead eyebrow="EVIDENCE AND DECISION CRITERIA" title="What the model checked" description="These are the original logged decision checks, not newly tuned rules."/>{checks.length?<div className="checks-grid">{checks.map((c,i)=><div className="check-item" key={`${c.label}-${i}`}><span className={c.passed?'check-success':'check-fail'}>{c.passed?<CheckCircle2 size={20}/>:<CircleAlert size={20}/>}</span><div><strong>{c.label}</strong><p>{String(c.value??'—')} · Required: {String(c.requirement??'—')}</p></div></div>)}</div>:<p className="muted">No decision checks published for this build.</p>}</Panel>
  <Panel className="method-panel"><SectionHead eyebrow="METHODOLOGY" title="How to interpret the evidence"/><div className="method-cards"><div><span className="method-number">01</span><h3>Point-in-time evaluation</h3><p>Historical decisions are replayed using information available on each decision date. Outcomes are not used for the original decision.</p></div><div><span className="method-number">02</span><h3>Independent analog matching</h3><p>The research method looks for comparable observations in similar market conditions. Sparse analog support reduces confidence.</p></div><div><span className="method-number">03</span><h3>Negative results remain visible</h3><p>Unfavorable, neutral, and insufficient-evidence outcomes are not concealed. Shadow research cannot modify the operational strategy.</p></div></div></Panel>
  <div className="links-card"><FileCheck2 size={19}/><span>Audit the underlying published files</span><a href="./analysis.json" target="_blank" rel="noopener noreferrer">Production analysis <ExternalLink size={14}/></a><a href="./historical_decisions.csv" download>Historical CSV <Download size={14}/></a><a href="./event_study.csv" download>Event study <Download size={14}/></a><a href="./legacy-dashboard.html">Legacy view <ArrowRight size={14}/></a></div>
 </div>;
}
function Research({data}){
  const snapshot=data.shadow;const check=inspectShadow(snapshot);
  const forwardRows=data.shadowHistory.map(r=>({date:isoDate(r.decision_date),score:finite(r.v3_opportunity_percentile),call:r.v3_call_state,production:r.production_action})).filter(r=>r.date);
  const last=check.safe?snapshot:null;
  return <div className="page-body"><div className="page-heading"><div><div className="heading-kicker"><BookOpen size={15}/> RESEARCH LAB</div><h1>Experimental, by design.</h1><p>Immutable forward predictions and model governance. V3 remains shadow-only unless explicit evidence and promotion gates are met.</p></div><Pill tone="info" icon={ShieldCheck}>Not deployed for decisions</Pill></div>
   {!check.safe?<div className="notice error" role="alert"><CircleAlert size={20}/><div><strong>Research view unavailable</strong><span>{check.reason}. The system will not present unverified research as approved.</span></div></div>:<>
   <Panel className="research-hero"><div><div className="overline">V3 SHADOW STATUS</div><h2>{last?.method_id||'STAB-004'} <span>· {proper(last?.method_status||'Unpromoted')}</span></h2><p>This candidate is a historical research artifact, not a live investment signal. Recorded predictions are immutable and separate from v2.1.</p></div><div className="research-mark"><ShieldCheck size={27}/><strong>Isolation enforced</strong><span>No production effect</span></div></Panel>
   <div className="metric-grid"><Metric label="Recorded predictions" value={last?.prediction_ledger_rows??'—'} note="Append-only shadow history" icon={Database}/><Metric label="Latest shadow decision" value={last?.latest_decision_date?monthDay(last.latest_decision_date):'—'} note={last?.latest_decision_date||'No records'} icon={CalendarDays}/><Metric label="Rolling percentile" value={last?.rolling_percentile===null?'—':valueFmt(last?.rolling_percentile,3)} note="Research score" icon={Activity}/><Metric label="Current state" value={proper(last?.call_state)} note="Shadow only" icon={BarChart3}/></div>
   <Panel className="chart-panel"><SectionHead eyebrow="IMMUTABLE FORWARD HISTORY" title="Published shadow ranks" description="Scores recorded before outcomes were opened; not an investment-performance chart."/>{forwardRows.some(x=>x.score!==null)?<div className="chart-area"><ResponsiveContainer height="100%" width="100%"><AreaChart data={forwardRows} margin={{top:15,right:10,bottom:0,left:-15}}><CartesianGrid vertical={false} stroke="#e8edf0" strokeDasharray="4 5"/><XAxis dataKey="date" tickFormatter={monthDay} tick={{fontSize:11,fill:'#83909B'}} axisLine={false} tickLine={false}/><YAxis domain={[0,1]} tick={{fontSize:11,fill:'#83909B'}} axisLine={false} tickLine={false}/><ReferenceLine y={0.5} stroke="#bac8d0" strokeDasharray="3 4"/><Tooltip content={<ChartTooltip/>}/><Area type="monotone" dataKey="score" name="Shadow percentile" stroke="#4576b1" fill="#4576b1" fillOpacity={.10} strokeWidth={2.5} dot={{r:3}} connectNulls={false}/></AreaChart></ResponsiveContainer></div>:<EmptyChart message="No scored shadow observations are published."/>}</Panel>
   <Panel><SectionHead eyebrow="GOVERNANCE GATES" title="Promotion is explicitly blocked"/><div className="guard-grid"><div><ShieldCheck size={20}/><strong>Research-only mode</strong><span>Confirmed</span></div><div><ShieldCheck size={20}/><strong>Production action changed</strong><span>No</span></div><div><ShieldCheck size={20}/><strong>Champion selected</strong><span>No</span></div><div><ShieldCheck size={20}/><strong>Evidence outcomes opened</strong><span>No</span></div><div><ShieldCheck size={20}/><strong>V3-019 promotion eligible</strong><span>No</span></div><div><ShieldCheck size={20}/><strong>Sizing multiplier</strong><span>1.00×</span></div></div></Panel>
   <div className="links-card"><Info size={19}/><span>Research references</span><a href="./v3_challenger.json" target="_blank" rel="noopener noreferrer">Research snapshot <ExternalLink size={14}/></a><a href="./v3_challenger_history.csv" download>Shadow history <Download size={14}/></a><a href={`${REPO}/tree/main/v3`} target="_blank" rel="noopener noreferrer">Methodology <ExternalLink size={14}/></a></div>
   </>}
  </div>
}
function Skeleton(){return <div className="page-body"><div className="skeleton title"/><div className="skeleton subtitle"/><div className="hero-grid"><div className="skeleton hero"/><div className="skeleton hero"/></div><div className="metric-grid">{[1,2,3,4].map(x=><div key={x} className="skeleton metric"/>)}</div><div className="skeleton chart"/></div>}
function App(){
 const [page,navigate]=useHashRoute();const [result,reload]=useDashboard();const [menu,setMenu]=useState(false);const [theme,setTheme]=useState('light');const updated=useVersionPoll(result.data?.version?.build_id);
 const rows=useMemo(()=>result.data?normalizeHistoryRows(result.data.history.decisions):[],[result.data]);
 const current=SECTIONS.find(s=>s.key===page)||SECTIONS[0];const freshnessInfo=result.data?freshness(result.data.analysis):null;
 return <div className={`app-shell theme-${theme}`}><aside className={`sidebar ${menu?'open':''}`} aria-label="Main navigation">
  <div className="brand"><div className="brand-mark"><Activity size={22}/></div><div><strong>FEAR & GREED<span className="brand-sub">ATLAS</span></strong><small>Market research intelligence</small></div><button className="mobile-close" type="button" onClick={()=>setMenu(false)} aria-label="Close menu"><X size={20}/></button></div>
  <div className="nav-caption">WORKSPACE</div><nav className="nav-items">{SECTIONS.map(s=>{const Icon=s.icon;return <button key={s.key} className={`nav-item ${page===s.key?'active':''}`} type="button" aria-current={page===s.key?'page':undefined} onClick={()=>{navigate(s.key);setMenu(false)}}><Icon size={19}/><span>{s.name}</span>{page===s.key&&<span className="nav-active-dot"/>}</button>})}</nav>
  <div className="nav-caption nav-secondary-caption">RESOURCES</div><a className="nav-item nav-link" href={`${REPO}/blob/main/README.md`} target="_blank" rel="noopener noreferrer"><BookOpen size={19}/><span>Documentation</span><ExternalLink size={13}/></a><a className="nav-item nav-link" href={REPO} target="_blank" rel="noopener noreferrer"><Github size={19}/><span>View repository</span><ExternalLink size={13}/></a>
  <div className="sidebar-bottom"><div className="sidebar-status"><div className="status-light"/><div><strong>Published research</strong><span>Read-only data · No trading</span></div></div><p>Independent market research. Not financial advice or a real-time feed.</p><span className="sidebar-version">ATLAS 1.0</span></div>
 </aside>
 <div className="main-col"><header className="topbar"><div className="top-left"><button className="icon-button mobile-menu" type="button" aria-label="Open menu" onClick={()=>setMenu(true)}><Menu size={22}/></button><div className="breadcrumb">Workspace <span>/</span> <strong>{current.name}</strong></div></div><div className="top-actions"><div className="build-indicator"><span className="status-light"/><span>{freshnessInfo?.stale?'Data may be stale':result.status==='error'?'Data unavailable':'Research data published'}</span></div><button className="icon-button" title="Refresh published data" aria-label="Refresh published data" onClick={reload}><RefreshCw size={17} className={result.status==='refreshing'?'spin':''}/></button><button className="icon-button" aria-label={theme==='light'?'Switch to dark mode':'Switch to light mode'} title="Toggle appearance" onClick={()=>setTheme(x=>x==='light'?'dark':'light')}>{theme==='light'?<Moon size={17}/>:<Sun size={17}/>}</button><a className="top-repo-link" href={REPO} target="_blank" rel="noopener noreferrer"><Github size={16}/> GitHub <ExternalLink size={13}/></a></div></header>
 {updated&&<div className="update-banner" role="status"><RefreshCw size={16}/><span>A newer published dashboard is available.</span><button onClick={()=>window.location.reload()} type="button">Load update <ArrowRight size={15}/></button></div>}
 {result.status==='loading'&&!result.data?<Skeleton/>:result.status==='error'&&!result.data?<div className="fatal-panel" role="alert"><CircleAlert size={30}/><h1>Published data is unavailable</h1><p>The dashboard could not load its required analysis or history. No substitute data will be displayed.</p><pre>{result.error}</pre><button className="button primary" onClick={reload}><RefreshCw size={17}/> Try again</button></div>:result.data?<>
   {result.status==='error'&&<div className="update-banner warning" role="alert">An update failed. You are viewing previously loaded published data. <button onClick={reload}>Retry</button></div>}
   {page==='overview'&&<Overview data={result.data} rows={rows} onNavigate={navigate}/>}
   {page==='explorer'&&<Explorer data={result.data} rows={rows}/>}
   {page==='strategy'&&<Strategy data={result.data} rows={rows}/>}
   {page==='research'&&<Research data={result.data}/>}
 </>:null}
 <footer className="footer"><span>© {new Date().getFullYear()} Fear & Greed Atlas · Independent research</span><span>Source timestamps reflect published data, not a real-time market feed.</span><a href={`${REPO}/blob/main/README-dashboard.md`} target="_blank" rel="noopener noreferrer">Methodology & limitations <ArrowRight size={14}/></a></footer>
 </div>{menu&&<button className="menu-backdrop" aria-label="Close navigation" onClick={()=>setMenu(false)} type="button"/>}
 </div>;
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);

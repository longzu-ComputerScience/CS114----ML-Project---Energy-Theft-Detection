"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  Upload,
  Zap,
  Shield,
  BarChart3,
  Users,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Search,
  ChevronLeft,
  ChevronRight,
  Database,
  Activity,
  FileText,
  X,
  Play,
  Info,
} from "lucide-react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
} from "recharts";

const API = "http://127.0.0.1:8000";

/* ─── Helpers ─── */
function riskColor(level) {
  const m = {
    Low: "var(--risk-low)",
    Medium: "var(--risk-medium)",
    High: "var(--risk-high)",
    Critical: "var(--risk-critical)",
  };
  return m[level] || "var(--text-secondary)";
}

function riskClass(level) {
  return (level || "").toLowerCase();
}

/* ─── Model Card ─── */
function ModelCard({ info }) {
  if (!info) return null;
  const metrics = info.test_metrics || {};
  return (
    <div className="card">
      <div className="card-header">
        <div className="icon">
          <Shield size={18} />
        </div>
        <h2>Model Card</h2>
        <span className="app-badge">LightGBM</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div className="metric-row">
          <span className="metric-name">Threshold</span>
          <span className="metric-value font-mono">{info.threshold?.toFixed(4)}</span>
        </div>
        <div className="metric-row">
          <span className="metric-name">Features</span>
          <span className="metric-value">{info.feature_count}</span>
        </div>
        <div className="metric-row">
          <span className="metric-name">F2 Score</span>
          <span className="metric-value">{metrics.F2}</span>
        </div>
        <div className="metric-row">
          <span className="metric-name">Recall</span>
          <span className="metric-value">{metrics.Recall}</span>
        </div>
        <div className="metric-row">
          <span className="metric-name">PR-AUC</span>
          <span className="metric-value">{metrics.PR_AUC}</span>
        </div>
        <div className="metric-row">
          <span className="metric-name">ROC-AUC</span>
          <span className="metric-value">{metrics.ROC_AUC}</span>
        </div>
      </div>
    </div>
  );
}

/* ─── Upload Panel ─── */
function UploadPanel({ onResult, loading, setLoading }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState("");

  async function handleFile(file) {
    if (!file) return;
    setFileName(file.name);
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/predict/upload`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }
      const data = await res.json();
      onResult(data);
    } catch (e) {
      alert("Prediction failed: " + e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleSample() {
    setFileName("test_raw_15_percent.csv (sample)");
    setLoading(true);
    try {
      const res = await fetch(`${API}/predict/sample`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }
      const data = await res.json();
      onResult(data);
    } catch (e) {
      alert("Sample prediction failed: " + e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <div className="card-header">
        <div className="icon">
          <Upload size={18} />
        </div>
        <h2>Upload Data</h2>
      </div>

      <div
        className={`upload-zone${dragOver ? " drag-over" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files?.[0]); }}
      >
        <Upload size={36} className="icon-large" />
        <p><strong>Drop CSV here</strong> or click to browse</p>
        <p className="hint">Raw format: CONS_NO + 1034 date columns + optional FLAG</p>
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      <div style={{ marginTop: 16, display: "flex", gap: 12, alignItems: "center" }}>
        <button className="btn btn-secondary btn-sm" onClick={handleSample} disabled={loading}>
          <Database size={14} /> Use test sample (6,356 rows)
        </button>
        {fileName && (
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            <FileText size={13} style={{ verticalAlign: -2, marginRight: 4 }} />
            {fileName}
          </span>
        )}
      </div>
    </div>
  );
}

/* ─── Summary Stats ─── */
function SummaryStats({ result }) {
  if (!result) return null;
  const { summary, confusion, rows, elapsed_seconds, threshold } = result;
  const theftRatio = ((summary.predicted_theft / rows) * 100).toFixed(1);

  return (
    <>
      <div className="grid-4 mb-24">
        <div className="stat-card">
          <span className="stat-label">Total Customers</span>
          <span className="stat-value">{rows.toLocaleString()}</span>
          <span className="stat-sub">{elapsed_seconds}s inference</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Suspected Theft</span>
          <span className="stat-value" style={{ color: "var(--risk-critical)" }}>
            {summary.predicted_theft.toLocaleString()}
          </span>
          <span className="stat-sub">{theftRatio}% of total</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Normal</span>
          <span className="stat-value" style={{ color: "var(--risk-low)" }}>
            {summary.predicted_normal.toLocaleString()}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Average Score</span>
          <span className="stat-value font-mono">{summary.average_score.toFixed(4)}</span>
          <span className="stat-sub">Threshold: {threshold.toFixed(4)}</span>
        </div>
      </div>

      {confusion && <ConfusionPanel confusion={confusion} />}
    </>
  );
}

/* ─── Confusion Panel ─── */
function ConfusionPanel({ confusion }) {
  const total = confusion.TP + confusion.TN + confusion.FP + confusion.FN;
  const accuracy = ((confusion.TP + confusion.TN) / total * 100).toFixed(1);

  return (
    <div className="card mb-24">
      <div className="card-header">
        <div className="icon">
          <BarChart3 size={18} />
        </div>
        <h2>Ground Truth Comparison</h2>
        <span className="app-badge" style={{ marginLeft: "auto" }}>
          Accuracy: {accuracy}%
        </span>
      </div>
      <div style={{ display: "flex", gap: 40, flexWrap: "wrap", alignItems: "center" }}>
        <div className="confusion-grid">
          <div className="confusion-cell" style={{ background: "rgba(34,197,94,0.12)" }}>
            <div className="label" style={{ color: "var(--risk-low)" }}>TP</div>
            <div className="value" style={{ color: "#4ade80" }}>{confusion.TP}</div>
          </div>
          <div className="confusion-cell" style={{ background: "rgba(249,115,22,0.12)" }}>
            <div className="label" style={{ color: "var(--risk-high)" }}>FP</div>
            <div className="value" style={{ color: "#fb923c" }}>{confusion.FP}</div>
          </div>
          <div className="confusion-cell" style={{ background: "rgba(239,68,68,0.12)" }}>
            <div className="label" style={{ color: "var(--risk-critical)" }}>FN</div>
            <div className="value" style={{ color: "#f87171" }}>{confusion.FN}</div>
          </div>
          <div className="confusion-cell" style={{ background: "rgba(59,130,246,0.12)" }}>
            <div className="label" style={{ color: "var(--accent-blue)" }}>TN</div>
            <div className="value" style={{ color: "#60a5fa" }}>{confusion.TN}</div>
          </div>
        </div>

        <div style={{ flex: 1, minWidth: 200 }}>
          <ConfusionPie confusion={confusion} />
        </div>
      </div>
    </div>
  );
}

function ConfusionPie({ confusion }) {
  const data = [
    { name: "TP", value: confusion.TP, fill: "#4ade80" },
    { name: "FP", value: confusion.FP, fill: "#fb923c" },
    { name: "FN", value: confusion.FN, fill: "#f87171" },
    { name: "TN", value: confusion.TN, fill: "#60a5fa" },
  ];
  return (
    <ResponsiveContainer width="100%" height={160}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={40}
          outerRadius={65}
          dataKey="value"
          paddingAngle={3}
          stroke="none"
        >
          {data.map((d, i) => (
            <Cell key={i} fill={d.fill} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            background: "#1a2235",
            border: "1px solid rgba(148,163,184,0.15)",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}

/* ─── Score Distribution Chart ─── */
function ScoreDistribution({ records, threshold }) {
  const bins = useMemo(() => {
    const n = 25;
    const arr = Array.from({ length: n }, (_, i) => ({
      range: `${(i / n).toFixed(2)}`,
      count: 0,
    }));
    for (const r of records) {
      const idx = Math.min(Math.floor(r.score * n), n - 1);
      arr[idx].count++;
    }
    return arr;
  }, [records]);

  return (
    <div className="card mb-24">
      <div className="card-header">
        <div className="icon">
          <Activity size={18} />
        </div>
        <h2>Score Distribution</h2>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={bins} barCategoryGap="8%">
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
          <XAxis dataKey="range" tick={{ fontSize: 10, fill: "#64748b" }} interval={4} />
          <YAxis tick={{ fontSize: 11, fill: "#64748b" }} />
          <Tooltip
            contentStyle={{
              background: "#1a2235",
              border: "1px solid rgba(148,163,184,0.15)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <Bar dataKey="count" radius={[3, 3, 0, 0]}>
            {bins.map((b, i) => {
              const mid = (i + 0.5) / bins.length;
              let fill = "#3b82f6";
              if (mid >= 0.7) fill = "#ef4444";
              else if (mid >= threshold) fill = "#f97316";
              else if (mid >= 0.3) fill = "#f59e0b";
              else fill = "#22c55e";
              return <Cell key={i} fill={fill} opacity={0.8} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{ display: "flex", gap: 16, justifyContent: "center", marginTop: 8 }}>
        {[
          ["Low", "#22c55e"],
          ["Medium", "#f59e0b"],
          ["High", "#f97316"],
          ["Critical", "#ef4444"],
        ].map(([label, color]) => (
          <span key={label} style={{ fontSize: "0.75rem", color: "#94a3b8", display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: color, display: "inline-block" }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ─── Results Table ─── */
const PAGE_SIZE = 20;

function ResultsTable({ records, selectedIdx, onSelect }) {
  const [page, setPage] = useState(0);
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("All");

  const filtered = useMemo(() => {
    let arr = records;
    if (search) {
      const q = search.toLowerCase();
      arr = arr.filter((r) => r.cons_no.toLowerCase().includes(q));
    }
    if (riskFilter !== "All") {
      arr = arr.filter((r) => r.risk_level === riskFilter);
    }
    return arr;
  }, [records, search, riskFilter]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageRecords = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const hasOutcome = records.some((r) => r.outcome);

  useEffect(() => { setPage(0); }, [search, riskFilter]);

  return (
    <div className="card">
      <div className="card-header" style={{ marginBottom: 12 }}>
        <div className="icon">
          <Users size={18} />
        </div>
        <h2>Customer Results ({filtered.length.toLocaleString()})</h2>
        <div style={{ marginLeft: "auto", display: "flex", gap: 10, alignItems: "center" }}>
          <div className="search-input">
            <Search size={14} color="var(--text-muted)" />
            <input
              placeholder="Search CONS_NO…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <select
            className="filter-select"
            value={riskFilter}
            onChange={(e) => setRiskFilter(e.target.value)}
          >
            <option>All</option>
            <option>Low</option>
            <option>Medium</option>
            <option>High</option>
            <option>Critical</option>
          </select>
        </div>
      </div>

      <div className="table-container" style={{ maxHeight: 520, overflowY: "auto" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>CONS_NO</th>
              <th>Score</th>
              <th>Risk</th>
              <th>Prediction</th>
              {hasOutcome && <th>Actual</th>}
              {hasOutcome && <th>Outcome</th>}
            </tr>
          </thead>
          <tbody>
            {pageRecords.map((r) => {
              const globalIdx = records.indexOf(r);
              return (
                <tr
                  key={globalIdx}
                  className={selectedIdx === globalIdx ? "selected" : ""}
                  onClick={() => onSelect(globalIdx)}
                  style={{ cursor: "pointer" }}
                >
                  <td className="font-mono" style={{ fontSize: "0.82rem" }}>{r.cons_no}</td>
                  <td className="font-mono" style={{ color: riskColor(r.risk_level) }}>
                    {r.score.toFixed(4)}
                  </td>
                  <td>
                    <span className={`risk-badge ${riskClass(r.risk_level)}`}>
                      {r.risk_level}
                    </span>
                  </td>
                  <td>
                    {r.prediction === "Suspected theft" ? (
                      <span style={{ color: "var(--risk-critical)", display: "flex", alignItems: "center", gap: 4 }}>
                        <AlertTriangle size={13} /> Theft
                      </span>
                    ) : (
                      <span style={{ color: "var(--risk-low)", display: "flex", alignItems: "center", gap: 4 }}>
                        <CheckCircle2 size={13} /> Normal
                      </span>
                    )}
                  </td>
                  {hasOutcome && (
                    <td>
                      {r.actual_label === 1 ? (
                        <span style={{ color: "#f87171" }}>Theft</span>
                      ) : (
                        <span style={{ color: "#60a5fa" }}>Normal</span>
                      )}
                    </td>
                  )}
                  {hasOutcome && (
                    <td>
                      <span className={`outcome-badge ${(r.outcome || "").toLowerCase()}`}>
                        {r.outcome}
                      </span>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0}>
            <ChevronLeft size={14} />
          </button>
          <span className="page-info">
            Page {page + 1} of {totalPages}
          </span>
          <button onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1}>
            <ChevronRight size={14} />
          </button>
        </div>
      )}
    </div>
  );
}

/* ─── Detail Panel ─── */
function DetailPanel({ record, onClose }) {
  if (!record) return null;

  // Downsample time-series for performance (show ~200 points)
  const ts = record.consumption_timeseries || [];
  const step = Math.max(1, Math.floor(ts.length / 200));
  const chartData = ts
    .filter((_, i) => i % step === 0)
    .map((d) => ({
      date: d.date,
      value: d.value,
    }));

  const fs = record.feature_summary || {};

  return (
    <div className="detail-panel">
      <div className="detail-header">
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <h3 className="font-mono">{record.cons_no}</h3>
          <span className={`risk-badge ${riskClass(record.risk_level)}`}>
            {record.risk_level}
          </span>
          {record.outcome && (
            <span className={`outcome-badge ${record.outcome.toLowerCase()}`}>
              {record.outcome}
            </span>
          )}
        </div>
        <button className="btn btn-ghost btn-sm" onClick={onClose}>
          <X size={16} />
        </button>
      </div>

      {/* Score card */}
      <div className="grid-3 mb-24">
        <div className="stat-card">
          <span className="stat-label">Risk Score</span>
          <span className="stat-value font-mono" style={{ color: riskColor(record.risk_level) }}>
            {record.score.toFixed(4)}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Prediction</span>
          <span className="stat-value" style={{ fontSize: "1.1rem" }}>
            {record.prediction}
          </span>
        </div>
        {record.actual_label !== undefined && (
          <div className="stat-card">
            <span className="stat-label">Actual Label</span>
            <span className="stat-value" style={{ fontSize: "1.1rem" }}>
              {record.actual_label === 1 ? "Theft" : "Normal"}
            </span>
          </div>
        )}
      </div>

      {/* Feature summary */}
      <div className="section-title mb-12">
        <Info size={16} color="var(--accent-blue)" /> Key Features
      </div>
      <div className="grid-3 mb-24" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))" }}>
        {Object.entries(fs).map(([k, v]) => (
          <div className="stat-card" key={k} style={{ padding: "12px 16px" }}>
            <span className="stat-label" style={{ fontSize: "0.72rem" }}>
              {k.replace(/_/g, " ")}
            </span>
            <span className="stat-value font-mono" style={{ fontSize: "1.1rem" }}>
              {typeof v === "number" ? v.toFixed(4) : v}
            </span>
          </div>
        ))}
      </div>

      {/* Consumption time-series */}
      {chartData.length > 0 && (
        <>
          <div className="section-title mb-12">
            <Activity size={16} color="var(--accent-cyan)" /> Consumption Time-Series
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 10, fill: "#64748b" }}
                interval={Math.floor(chartData.length / 6)}
              />
              <YAxis tick={{ fontSize: 11, fill: "#64748b" }} />
              <Tooltip
                contentStyle={{
                  background: "#1a2235",
                  border: "1px solid rgba(148,163,184,0.15)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke="var(--accent-cyan)"
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 3, fill: "var(--accent-cyan)" }}
              />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   Main Page
   ═══════════════════════════════════════════════════════════════════════ */
export default function HomePage() {
  const [modelInfo, setModelInfo] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [healthy, setHealthy] = useState(null);

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then(() => setHealthy(true))
      .catch(() => setHealthy(false));

    fetch(`${API}/model-info`)
      .then((r) => r.json())
      .then(setModelInfo)
      .catch(() => {});
  }, []);

  const selectedRecord =
    result && selectedIdx !== null ? result.records[selectedIdx] : null;

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="app-title">
          <Zap size={28} color="var(--accent-cyan)" />
          <h1>Energy Theft Detection</h1>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {healthy === true && (
            <span className="app-badge" style={{ background: "rgba(34,197,94,0.12)", color: "#22c55e", borderColor: "rgba(34,197,94,0.2)" }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e", display: "inline-block" }} />
              API Connected
            </span>
          )}
          {healthy === false && (
            <span className="app-badge" style={{ background: "rgba(239,68,68,0.12)", color: "#ef4444", borderColor: "rgba(239,68,68,0.2)" }}>
              <XCircle size={12} /> API Offline
            </span>
          )}
        </div>
      </header>

      {/* Model card + Upload side by side */}
      <div className="grid-2 mb-32">
        <ModelCard info={modelInfo} />
        <UploadPanel onResult={setResult} loading={loading} setLoading={setLoading} />
      </div>

      {/* Loading */}
      {loading && (
        <div className="loading-overlay">
          <div className="spinner" style={{ width: 36, height: 36 }} />
          <p>Running inference pipeline…</p>
          <p style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
            Preprocessing → Feature Engineering → LightGBM Predict
          </p>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <>
          <SummaryStats result={result} />
          <ScoreDistribution records={result.records} threshold={result.threshold} />
          <ResultsTable
            records={result.records}
            selectedIdx={selectedIdx}
            onSelect={(i) => setSelectedIdx(i === selectedIdx ? null : i)}
          />
          <DetailPanel
            record={selectedRecord}
            onClose={() => setSelectedIdx(null)}
          />
        </>
      )}
    </div>
  );
}

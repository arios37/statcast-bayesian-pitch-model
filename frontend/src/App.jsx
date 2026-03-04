import { useState, useEffect, useMemo } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
  AreaChart, Area, ComposedChart, Bar,
  BarChart, Line,
} from "recharts";

// ─── PITCH NAME LOOKUP ─────────────────────────────────────────────────────
const PITCH_NAMES = {
  FF: "4-Seam Fastball", SI: "Sinker", FC: "Cutter", SL: "Slider",
  CU: "Curveball", CH: "Changeup", ST: "Sweeper", FS: "Splitter",
  KC: "Knuckle Curve", SV: "Screwball",
};

// ─── GAUSSIAN PDF ──────────────────────────────────────────────────────────
function gaussianPDF(x, mu, sigma) {
  const coeff = 1 / (sigma * Math.sqrt(2 * Math.PI));
  const exp = -0.5 * ((x - mu) / sigma) ** 2;
  return coeff * Math.exp(exp);
}

function generatePosterior(mu, sigma, nPoints = 200) {
  const lo = mu - 4 * sigma;
  const hi = mu + 4 * sigma;
  const step = (hi - lo) / nPoints;
  const pts = [];
  for (let x = lo; x <= hi; x += step) {
    pts.push({
      x: parseFloat((x * 1000).toFixed(2)),
      density: gaussianPDF(x, mu, sigma),
    });
  }
  return pts;
}

// ─── COUNT LEVERAGE DATA ───────────────────────────────────────────────────
function generateCountData(pitches) {
  const counts = [
    "0-0", "0-1", "0-2", "1-0", "1-1", "1-2",
    "2-0", "2-1", "2-2", "3-0", "3-1", "3-2",
  ];
  const leverageMultiplier = {
    "0-0": 1.0, "0-1": 0.7, "0-2": 0.4, "1-0": 1.2, "1-1": 0.9, "1-2": 0.5,
    "2-0": 1.5, "2-1": 1.1, "2-2": 0.7, "3-0": 1.8, "3-1": 1.4, "3-2": 1.0,
  };
  return counts.map((c) => {
    const bestPitch = Object.entries(pitches).reduce(
      (best, [type, p]) => (p.rawDRE < best.rawDRE ? { type, ...p } : best),
      { type: "", rawDRE: 1 }
    );
    const base = bestPitch.rawDRE * leverageMultiplier[c];
    return {
      count: c,
      expectedDRE: parseFloat((base * 1000).toFixed(2)),
      pitcherAdvantage: leverageMultiplier[c] < 1,
    };
  });
}

// ─── CUSTOM TOOLTIP COMPONENTS ─────────────────────────────────────────────
const tooltipStyle = {
  background: "rgba(15, 18, 25, 0.95)",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 6,
  padding: "10px 14px",
  fontSize: 12,
  fontFamily: "'JetBrains Mono', monospace",
  color: "#c8ccd4",
};

const MovementTooltip = ({ active, payload }) => {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div style={tooltipStyle}>
      <div style={{ fontWeight: 700, color: "#fff", marginBottom: 4 }}>
        {PITCH_NAMES[d.type] || d.type}
      </div>
      <div>
        Horz: <span style={{ color: "#60a5fa" }}>{d.pfx_x}"</span> &nbsp;
        Vert: <span style={{ color: "#60a5fa" }}>{d.pfx_z}"</span>
      </div>
      <div>
        Velo: <span style={{ color: "#f59e0b" }}>{d.velo} mph</span> &nbsp;
        Spin: <span style={{ color: "#f59e0b" }}>{d.spin} rpm</span>
      </div>
      <div>
        Usage: <span style={{ color: "#a78bfa" }}>{(d.usage * 100).toFixed(0)}%</span>
        &nbsp; DRE: <span style={{ color: d.rawDRE < 0 ? "#34d399" : "#f87171" }}>
          {(d.rawDRE * 1000).toFixed(1)}
        </span>
      </div>
    </div>
  );
};

const PosteriorTooltip = ({ active, payload }) => {
  if (!active || !payload?.[0]) return null;
  return (
    <div style={tooltipStyle}>
      <div>
        alpha x 10^3: <span style={{ color: "#60a5fa" }}>{payload[0].payload.x.toFixed(1)}</span>
      </div>
      <div>
        density: <span style={{ color: "#a78bfa" }}>{payload[0].value.toFixed(1)}</span>
      </div>
    </div>
  );
};

// ─── LOADING STATE ─────────────────────────────────────────────────────────
function LoadingScreen() {
  return (
    <div style={{
      minHeight: "100vh", background: "#0a0c10", display: "flex",
      alignItems: "center", justifyContent: "center", flexDirection: "column",
      fontFamily: "'JetBrains Mono', monospace", color: "#c8ccd4",
    }}>
      <div style={{ fontSize: 10, letterSpacing: 4, color: "#60a5fa", marginBottom: 12 }}>
        STATCAST x PYMC
      </div>
      <div style={{ fontSize: 18, fontWeight: 700, color: "#fff" }}>Loading posterior data...</div>
      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginTop: 8 }}>
        Fetching hierarchical model results
      </div>
    </div>
  );
}

function ErrorScreen({ error }) {
  return (
    <div style={{
      minHeight: "100vh", background: "#0a0c10", display: "flex",
      alignItems: "center", justifyContent: "center", flexDirection: "column",
      fontFamily: "'JetBrains Mono', monospace", color: "#c8ccd4", padding: 40,
    }}>
      <div style={{ fontSize: 18, fontWeight: 700, color: "#f87171", marginBottom: 12 }}>
        Failed to load model data
      </div>
      <div style={{ fontSize: 12, color: "rgba(255,255,255,0.5)", maxWidth: 500, textAlign: "center" }}>
        {error}
      </div>
      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", marginTop: 16 }}>
        Run: python -m src.export_posteriors to generate the data file.
      </div>
    </div>
  );
}

// ─── MAIN APP ──────────────────────────────────────────────────────────────
export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPitcher, setSelectedPitcher] = useState(null);
  const [activeTab, setActiveTab] = useState("movement");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);

  // Load JSON data
  useEffect(() => {
    fetch(import.meta.env.BASE_URL + "data/posteriors.json")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        return res.json();
      })
      .then((json) => {
        setData(json);
        // Default to first pitcher
        const names = Object.keys(json.pitchers);
        if (names.length > 0) setSelectedPitcher(names[0]);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Derived data
  const pitcher = data?.pitchers?.[selectedPitcher];
  const pitcherNames = data ? Object.keys(data.pitchers) : [];

  const movementData = useMemo(() => {
    if (!pitcher) return [];
    return Object.entries(pitcher.pitches).map(([type, p]) => ({
      type, ...p, name: PITCH_NAMES[type] || type,
    }));
  }, [pitcher]);

  const posteriorData = useMemo(() => {
    if (!pitcher) return [];
    return generatePosterior(pitcher.posteriorMean, pitcher.posteriorSD);
  }, [pitcher]);

  const shrinkageData = useMemo(() => {
    if (!data || !pitcher) return [];
    const team = pitcher.team;
    return Object.entries(data.pitchers)
      .filter(([, p]) => p.team === team)
      .map(([name, p]) => ({
        name: name.split(" ").pop(),
        fullName: name,
        raw: parseFloat((p.rawMeanDRE * 1000).toFixed(2)),
        posterior: parseFloat((p.posteriorMean * 1000).toFixed(2)),
        ci90lo: parseFloat((p.ci90[0] * 1000).toFixed(2)),
        ci90hi: parseFloat((p.ci90[1] * 1000).toFixed(2)),
        nPitches: p.nPitches,
        selected: name === selectedPitcher,
        inModel: p.inModel,
      }))
      .sort((a, b) => a.posterior - b.posterior);
  }, [data, pitcher, selectedPitcher]);

  const countData = useMemo(() => {
    if (!pitcher) return [];
    return generateCountData(pitcher.pitches);
  }, [pitcher]);

  const filteredPitchers = useMemo(() => {
    if (!data) return [];
    const q = searchQuery.toLowerCase().trim();
    if (!q) return [];
    return pitcherNames
      .filter((name) => name.toLowerCase().includes(q))
      .slice(0, 25);
  }, [data, pitcherNames, searchQuery]);

  if (loading) return <LoadingScreen />;
  if (error) return <ErrorScreen error={error} />;
  if (!pitcher) return <ErrorScreen error="No pitcher data available" />;

  const ci90 = pitcher.ci90.map((v) => (v * 1000).toFixed(1));
  const leagueMean = data.leagueMeanDRE;

  const tabs = [
    { key: "movement", label: "ARSENAL" },
    { key: "posterior", label: "POSTERIOR" },
    { key: "shrinkage", label: "SHRINKAGE" },
    { key: "counts", label: "COUNT LEVERAGE" },
  ];

  return (
    <div style={{
      minHeight: "100vh", background: "#0a0c10",
      fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      color: "#c8ccd4", padding: 0,
    }}>
      {/* Header */}
      <div style={{
        background: "linear-gradient(135deg, #005a9c 0%, #002d62 50%, #0a0c10 100%)",
        padding: "28px 32px 20px",
        borderBottom: "2px solid #ef4444",
      }}>
        <div style={{
          display: "flex", justifyContent: "space-between",
          alignItems: "flex-start", flexWrap: "wrap", gap: 16,
        }}>
          <div>
            <div style={{
              fontSize: 10, letterSpacing: 4, color: "#60a5fa",
              marginBottom: 4, fontWeight: 600,
            }}>
              STATCAST x PYMC
            </div>
            <h1 style={{
              margin: 0, fontSize: 26, fontWeight: 800,
              letterSpacing: -0.5, color: "#fff", lineHeight: 1.1,
            }}>
              Bayesian Pitch Model Explorer
            </h1>
            <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", marginTop: 6 }}>
              Hierarchical model | {data._meta.year} season |{" "}
              {data._meta.totalPitches.toLocaleString()} pitches |{" "}
              Pitcher-level partial pooling on delta run expectancy
            </div>
          </div>
          <div style={{
            background: "rgba(0,0,0,0.3)", borderRadius: 8, padding: "10px 14px",
            border: "1px solid rgba(255,255,255,0.08)", minWidth: 240, position: "relative",
          }}>
            <div style={{ fontSize: 9, letterSpacing: 2, color: "#60a5fa", marginBottom: 6 }}>
              SEARCH PITCHER
            </div>
            <input
              type="text"
              value={searchOpen ? searchQuery : selectedPitcher || ""}
              placeholder="Type a name..."
              onFocus={() => { setSearchOpen(true); setSearchQuery(""); }}
              onBlur={() => setTimeout(() => setSearchOpen(false), 200)}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && filteredPitchers.length > 0) {
                  setSelectedPitcher(filteredPitchers[0]);
                  setSearchQuery("");
                  setSearchOpen(false);
                  e.target.blur();
                }
                if (e.key === "Escape") {
                  setSearchQuery("");
                  setSearchOpen(false);
                  e.target.blur();
                }
              }}
              style={{
                width: "100%", background: "rgba(255,255,255,0.06)", color: "#fff",
                border: `1px solid ${searchOpen ? "#60a5fa" : "rgba(255,255,255,0.15)"}`,
                borderRadius: searchOpen && searchQuery.trim().length > 0 && filteredPitchers.length > 0 ? "4px 4px 0 0" : 4,
                padding: "6px 8px", fontSize: 13, fontFamily: "inherit",
                outline: "none", transition: "border-color 0.15s",
              }}
            />
            {searchOpen && searchQuery.trim().length > 0 && (
              <div style={{
                position: "absolute", top: "100%", left: 0, right: 0, zIndex: 50,
                background: "#1a1d27", border: "1px solid rgba(255,255,255,0.15)",
                borderTop: "none", borderRadius: "0 0 6px 6px",
                maxHeight: 280, overflowY: "auto",
              }}>
                {filteredPitchers.length === 0 ? (
                  <div style={{
                    padding: "10px 12px", fontSize: 11, color: "rgba(255,255,255,0.3)",
                    textAlign: "center",
                  }}>
                    No pitchers found
                  </div>
                ) : (
                  filteredPitchers.map((name) => (
                    <div
                      key={name}
                      onMouseDown={(e) => {
                        e.preventDefault();
                        setSelectedPitcher(name);
                        setSearchQuery("");
                        setSearchOpen(false);
                      }}
                      style={{
                        padding: "7px 12px", fontSize: 12, cursor: "pointer",
                        color: name === selectedPitcher ? "#60a5fa" : "#c8ccd4",
                        background: name === selectedPitcher ? "rgba(96,165,250,0.08)" : "transparent",
                        borderLeft: name === selectedPitcher ? "2px solid #60a5fa" : "2px solid transparent",
                        transition: "background 0.1s",
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = name === selectedPitcher
                          ? "rgba(96,165,250,0.08)" : "transparent";
                      }}
                    >
                      <span>{name}</span>
                      <span style={{
                        float: "right", fontSize: 10, color: "rgba(255,255,255,0.25)",
                      }}>
                        <span style={{ color: "rgba(255,255,255,0.35)", marginRight: 8 }}>
                          {data.pitchers[name].team}
                        </span>
                        {data.pitchers[name].nPitches.toLocaleString()}
                      </span>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>

        {/* Stat chips */}
        <div style={{ display: "flex", gap: 12, marginTop: 16, flexWrap: "wrap" }}>
          {[
            { label: "TEAM", value: pitcher.team || "—" },
            { label: "PITCHES", value: pitcher.nPitches.toLocaleString() },
            {
              label: "alpha x 10^3",
              value: (pitcher.posteriorMean * 1000).toFixed(1),
              color: pitcher.posteriorMean < 0 ? "#34d399" : "#f87171",
            },
            { label: "90% CI", value: `[${ci90[0]}, ${ci90[1]}]` },
            {
              label: "SOURCE",
              value: pitcher.inModel ? "MCMC" : "CONJUGATE",
              color: pitcher.inModel ? "#60a5fa" : "#f59e0b",
            },
          ].map((s) => (
            <div key={s.label} style={{
              background: "rgba(0,0,0,0.25)", borderRadius: 6, padding: "6px 12px",
              border: "1px solid rgba(255,255,255,0.06)",
            }}>
              <div style={{ fontSize: 8, letterSpacing: 2, color: "rgba(255,255,255,0.4)" }}>
                {s.label}
              </div>
              <div style={{
                fontSize: 15, fontWeight: 700,
                color: s.color || "#fff", marginTop: 1,
              }}>
                {s.value}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Tab bar */}
      <div style={{
        display: "flex", gap: 0, background: "#0d0f14",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}>
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setActiveTab(t.key)}
            style={{
              padding: "12px 20px", fontSize: 10, letterSpacing: 2.5, fontWeight: 700,
              background: activeTab === t.key ? "rgba(96, 165, 250, 0.08)" : "transparent",
              color: activeTab === t.key ? "#60a5fa" : "rgba(255,255,255,0.35)",
              border: "none",
              borderBottom: activeTab === t.key
                ? "2px solid #60a5fa"
                : "2px solid transparent",
              cursor: "pointer", fontFamily: "inherit", transition: "all 0.15s",
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: "24px 28px" }}>

        {/* ── MOVEMENT TAB ──────────────────────────────────── */}
        {activeTab === "movement" && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "#fff" }}>
                Pitch Movement Profile
              </h2>
              <p style={{ margin: "4px 0 0", fontSize: 11, color: "rgba(255,255,255,0.4)" }}>
                Horizontal vs. vertical induced movement (inches). Pitcher POV.
                Negative pfx_x = arm-side run.
              </p>
            </div>
            <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
              <div style={{ flex: "1 1 420px", minWidth: 320 }}>
                <ResponsiveContainer width="100%" height={360}>
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis
                      dataKey="pfx_x" type="number" name="Horz Movement"
                      domain={[-18, 12]} tick={{ fill: "#666", fontSize: 10 }}
                      label={{ value: "Horizontal Movement (in)", position: "bottom", offset: 5, fill: "#555", fontSize: 10 }}
                    />
                    <YAxis
                      dataKey="pfx_z" type="number" name="Vert Movement"
                      domain={[-14, 22]} tick={{ fill: "#666", fontSize: 10 }}
                      label={{ value: "Induced Vert. Break (in)", angle: -90, position: "insideLeft", offset: 5, fill: "#555", fontSize: 10 }}
                    />
                    <ReferenceLine x={0} stroke="rgba(255,255,255,0.1)" />
                    <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
                    <Tooltip content={<MovementTooltip />} />
                    <Scatter data={movementData} shape="circle">
                      {movementData.map((d, i) => (
                        <Cell key={i} fill={d.color} r={Math.max(8, d.usage * 40)} opacity={0.85} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <div style={{ flex: "0 0 220px" }}>
                <div style={{
                  fontSize: 9, letterSpacing: 2, color: "#60a5fa",
                  marginBottom: 10, fontWeight: 600,
                }}>
                  ARSENAL BREAKDOWN
                </div>
                {movementData.map((d) => (
                  <div key={d.type} style={{
                    display: "flex", alignItems: "center", gap: 10, padding: "8px 0",
                    borderBottom: "1px solid rgba(255,255,255,0.04)",
                  }}>
                    <div style={{
                      width: 10, height: 10, borderRadius: "50%",
                      background: d.color, flexShrink: 0,
                    }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: "#fff" }}>{d.type}</div>
                      <div style={{ fontSize: 9, color: "rgba(255,255,255,0.35)" }}>{d.name}</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: 11, color: "#fff" }}>{d.velo} mph</div>
                      <div style={{
                        fontSize: 9,
                        color: d.rawDRE < 0 ? "#34d399" : "#f87171",
                      }}>
                        {(d.rawDRE * 1000).toFixed(1)} DRE
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── POSTERIOR TAB ─────────────────────────────────── */}
        {activeTab === "posterior" && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "#fff" }}>
                Posterior Distribution: alpha_{selectedPitcher.split(" ").pop()}
              </h2>
              <p style={{ margin: "4px 0 0", fontSize: 11, color: "rgba(255,255,255,0.4)" }}>
                {pitcher.inModel
                  ? "MCMC posterior from hierarchical model. Pitcher-level random intercept from Normal(mu_alpha, sigma_alpha)."
                  : "Conjugate normal-normal update using the model's estimated hyperprior. Analytically equivalent to MCMC for this model structure."}
                {" "}Units: delta run expectancy x 10^3. Negative = prevents runs.
              </p>
            </div>
            <ResponsiveContainer width="100%" height={380}>
              <AreaChart data={posteriorData} margin={{ top: 10, right: 30, bottom: 30, left: 20 }}>
                <defs>
                  <linearGradient id="posteriorGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#60a5fa" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="x" type="number" tick={{ fill: "#666", fontSize: 10 }}
                  label={{
                    value: "alpha x 10^3 (delta run expectancy per pitch)",
                    position: "bottom", offset: 10, fill: "#555", fontSize: 10,
                  }}
                />
                <YAxis
                  tick={{ fill: "#666", fontSize: 10 }}
                  label={{ value: "density", angle: -90, position: "insideLeft", fill: "#555", fontSize: 10 }}
                />
                <Tooltip content={<PosteriorTooltip />} />
                <ReferenceLine
                  x={leagueMean * 1000}
                  stroke="#ef4444" strokeDasharray="6 3" strokeWidth={1.5}
                  label={{ value: "mu_alpha (league)", position: "top", fill: "#ef4444", fontSize: 9 }}
                />
                <ReferenceLine
                  x={pitcher.posteriorMean * 1000}
                  stroke="#60a5fa" strokeDasharray="3 3" strokeWidth={1.5}
                  label={{ value: "alpha_hat", position: "top", fill: "#60a5fa", fontSize: 10, fontWeight: 700 }}
                />
                <ReferenceLine
                  x={pitcher.rawMeanDRE * 1000}
                  stroke="#f59e0b" strokeDasharray="3 3" strokeWidth={1}
                  label={{ value: "raw mean", position: "insideTopRight", fill: "#f59e0b", fontSize: 9 }}
                />
                <Area
                  type="monotone" dataKey="density" stroke="#60a5fa"
                  strokeWidth={2} fill="url(#posteriorGrad)"
                />
              </AreaChart>
            </ResponsiveContainer>
            <div style={{
              display: "flex", gap: 20, justifyContent: "center",
              marginTop: 12, flexWrap: "wrap",
            }}>
              {[
                { color: "#60a5fa", dash: "3 3", label: "Posterior mean (alpha_hat)" },
                { color: "#f59e0b", dash: "3 3", label: "Raw sample mean" },
                { color: "#ef4444", dash: "6 3", label: "League hyperprior (mu_alpha)" },
              ].map((l) => (
                <div key={l.label} style={{
                  display: "flex", alignItems: "center", gap: 6, fontSize: 10,
                }}>
                  <svg width={24} height={2}>
                    <line x1={0} y1={1} x2={24} y2={1}
                      stroke={l.color} strokeWidth={2} strokeDasharray={l.dash}
                    />
                  </svg>
                  <span style={{ color: "rgba(255,255,255,0.5)" }}>{l.label}</span>
                </div>
              ))}
            </div>
            <div style={{
              marginTop: 20, background: "rgba(96, 165, 250, 0.05)",
              border: "1px solid rgba(96, 165, 250, 0.12)",
              borderRadius: 8, padding: "14px 18px", fontSize: 11,
              color: "rgba(255,255,255,0.55)", lineHeight: 1.6,
            }}>
              <strong style={{ color: "#60a5fa" }}>Interpretation:</strong>{" "}
              The posterior mean alpha_hat = {(pitcher.posteriorMean * 1000).toFixed(1)} x 10^-3
              indicates {selectedPitcher}{" "}
              {pitcher.posteriorMean < 0 ? "prevents" : "allows"} approximately{" "}
              {Math.abs(pitcher.posteriorMean * 1000).toFixed(1)} milliRuns per pitch
              relative to the league-average intercept, after partial pooling.
              The 90% credible interval [{ci90[0]}, {ci90[1]}] quantifies uncertainty.
              {!pitcher.inModel && (
                <span style={{ color: "#f59e0b" }}>
                  {" "}This pitcher was not in the MCMC subsample. The posterior was computed
                  via conjugate normal-normal update using the model's estimated hyperprior
                  (mu_alpha = {(data.hyperprior.mu_alpha_mean * 1000).toFixed(1)},
                  sigma_alpha = {(data.hyperprior.sigma_alpha_mean * 1000).toFixed(1)}) and
                  {pitcher.nPitches.toLocaleString()} observed pitches.
                </span>
              )}
            </div>
          </div>
        )}

        {/* ── SHRINKAGE TAB ────────────────────────────────── */}
        {activeTab === "shrinkage" && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "#fff" }}>
                Bayesian Shrinkage: {pitcher.team} Pitching Staff
              </h2>
              <p style={{ margin: "4px 0 0", fontSize: 11, color: "rgba(255,255,255,0.4)" }}>
                Partial pooling pulls extreme raw means toward the league hyperprior mu_alpha.
                Pitchers with fewer pitches experience more shrinkage. Showing all {pitcher.team} pitchers.
                Diamond = raw mean. Bar = posterior mean.
              </p>
            </div>
            <ResponsiveContainer width="100%" height={Math.max(340, shrinkageData.length * 32)}>
              <ComposedChart
                data={shrinkageData} layout="vertical"
                margin={{ top: 10, right: 30, bottom: 20, left: 90 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  type="number" tick={{ fill: "#666", fontSize: 10 }}
                  label={{
                    value: "delta run expectancy x 10^3 per pitch",
                    position: "bottom", offset: 5, fill: "#555", fontSize: 10,
                  }}
                />
                <YAxis
                  dataKey="name" type="category"
                  tick={{ fill: "#999", fontSize: 11, fontWeight: 600 }} width={80}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.[0]) return null;
                    const d = payload[0].payload;
                    return (
                      <div style={tooltipStyle}>
                        <div style={{ fontWeight: 700, color: "#fff", marginBottom: 4 }}>
                          {d.fullName}
                        </div>
                        <div>Raw mean: <span style={{ color: "#f59e0b" }}>{d.raw.toFixed(1)}</span></div>
                        <div>Posterior: <span style={{ color: "#60a5fa" }}>{d.posterior.toFixed(1)}</span></div>
                        <div>
                          90% CI: <span style={{ color: "#a78bfa" }}>
                            [{d.ci90lo.toFixed(1)}, {d.ci90hi.toFixed(1)}]
                          </span>
                        </div>
                        <div>n = {d.nPitches.toLocaleString()} pitches</div>
                        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.35)", marginTop: 4 }}>
                          Shrinkage: {Math.abs(d.raw - d.posterior).toFixed(1)} toward mu_alpha
                          {!d.inModel && " (conjugate update)"}
                        </div>
                      </div>
                    );
                  }}
                />
                <ReferenceLine
                  x={leagueMean * 1000}
                  stroke="#ef4444" strokeDasharray="6 3" strokeWidth={1.5}
                />
                <Bar dataKey="posterior" barSize={14} radius={[0, 3, 3, 0]}>
                  {shrinkageData.map((d, i) => (
                    <Cell
                      key={i}
                      fill={d.selected ? "#60a5fa" : "rgba(96, 165, 250, 0.3)"}
                    />
                  ))}
                </Bar>
                <Scatter dataKey="raw" fill="#f59e0b" shape="diamond" legendType="diamond" />
              </ComposedChart>
            </ResponsiveContainer>
            <div style={{
              display: "flex", gap: 20, justifyContent: "center",
              marginTop: 8, flexWrap: "wrap",
            }}>
              {[
                { shape: "\u25a0", color: "#60a5fa", label: "Posterior mean (partial pooling)" },
                { shape: "\u25c6", color: "#f59e0b", label: "Raw sample mean (no pooling)" },
                { shape: "\u2502", color: "#ef4444", label: "League hyperprior mu_alpha" },
              ].map((l) => (
                <div key={l.label} style={{
                  display: "flex", alignItems: "center", gap: 6, fontSize: 10,
                }}>
                  <span style={{ color: l.color, fontSize: 12 }}>{l.shape}</span>
                  <span style={{ color: "rgba(255,255,255,0.5)" }}>{l.label}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── COUNT LEVERAGE TAB ───────────────────────────── */}
        {activeTab === "counts" && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "#fff" }}>
                Expected Delta Run Value by Count
              </h2>
              <p style={{ margin: "4px 0 0", fontSize: 11, color: "rgba(255,255,255,0.4)" }}>
                Model prediction for {selectedPitcher}'s best pitch in each count state.
                Pitcher-ahead counts compress run value; hitter-ahead counts amplify it.
              </p>
            </div>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={countData} margin={{ top: 10, right: 30, bottom: 30, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="count" tick={{ fill: "#999", fontSize: 10 }}
                  label={{
                    value: "Ball-Strike Count", position: "bottom",
                    offset: 10, fill: "#555", fontSize: 10,
                  }}
                />
                <YAxis
                  tick={{ fill: "#666", fontSize: 10 }}
                  label={{
                    value: "E[delta run exp] x 10^3", angle: -90,
                    position: "insideLeft", fill: "#555", fontSize: 10,
                  }}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.[0]) return null;
                    const d = payload[0].payload;
                    return (
                      <div style={tooltipStyle}>
                        <div style={{ fontWeight: 700, color: "#fff" }}>Count: {d.count}</div>
                        <div>
                          E[DRE] x 10^3:{" "}
                          <span style={{ color: d.expectedDRE < 0 ? "#34d399" : "#f87171" }}>
                            {d.expectedDRE.toFixed(1)}
                          </span>
                        </div>
                        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.35)", marginTop: 2 }}>
                          {d.pitcherAdvantage
                            ? "Pitcher-ahead count: compressed leverage"
                            : "Hitter-ahead count: amplified leverage"}
                        </div>
                      </div>
                    );
                  }}
                />
                <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
                <Bar dataKey="expectedDRE" radius={[3, 3, 0, 0]}>
                  {countData.map((d, i) => (
                    <Cell
                      key={i}
                      fill={d.expectedDRE < 0
                        ? "rgba(52, 211, 153, 0.7)"
                        : "rgba(248, 113, 113, 0.7)"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{
        padding: "16px 28px",
        borderTop: "1px solid rgba(255,255,255,0.04)",
        fontSize: 9, color: "rgba(255,255,255,0.2)",
        textAlign: "center", lineHeight: 1.6,
      }}>
        <div>
          Model: y ~ Normal(alpha[pitcher] + X*beta, sigma) | alpha[pitcher] ~ Normal(mu_alpha, sigma_alpha)
        </div>
        <div>
          Data: {data._meta.year} Statcast via pybaseball |{" "}
          {data._meta.totalPitches.toLocaleString()} pitches |{" "}
          {data._meta.totalPitchers} pitchers |{" "}
          {data._meta.modelType === "hierarchical_normal"
            ? "Hierarchical Normal"
            : data._meta.modelType}{" "}
          | PyMC {">"}= 5.10
        </div>
        <div>
          MCMC: {data._meta.samplerInfo.draws} draws x {data._meta.samplerInfo.chains} chains |{" "}
          {data.diagnostics.divergences} divergences
        </div>
        <div style={{ marginTop: 4 }}>
          Angel Rios | github.com/arios37
        </div>
      </div>
    </div>
  );
}

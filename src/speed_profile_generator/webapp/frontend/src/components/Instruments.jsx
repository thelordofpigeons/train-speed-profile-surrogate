import React from 'react';

// Reusable Needle Gauge Component
const NeedleGauge = ({ value, min, max, unit, size = 180, dangerZone = 0.9, color = '#3b82f6', label }) => {
    const range = max - min;
    const normalized = Math.min(Math.max((value - min) / range, 0), 1);
    const angle = normalized * 180 - 90; // -90 to 90 degrees
    const r = size / 2;
    const cx = r, cy = r;

    // Ticks
    const ticks = [];
    for (let i = 0; i <= 10; i++) {
        const val = min + (range * (i / 10));
        const ang = (i / 10) * 180 - 180;
        const rad = (ang * Math.PI) / 180;
        const x1 = cx + (r - 20) * Math.cos(rad);
        const y1 = cy + (r - 20) * Math.sin(rad);
        const x2 = cx + (r - 10) * Math.cos(rad);
        const y2 = cy + (r - 10) * Math.sin(rad);
        ticks.push(<line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#64748b" strokeWidth={i % 5 === 0 ? 2 : 1} />);
    }

    return (
        <div className="gauge-section">
            <span className="gauge-label">{label}</span>
            <div style={{ width: size, height: size / 1.5, position: 'relative', overflow: 'hidden' }}>
                <svg viewBox={`0 0 ${size} ${size}`} style={{ marginTop: size * 0.1 }}>
                    {/* Background Arc */}
                    <path
                        d={`M 10 ${r} A ${r - 10} ${r - 10} 0 0 1 ${size - 10} ${r}`}
                        fill="none"
                        stroke="#334155"
                        strokeWidth="12"
                        strokeLinecap="round"
                    />
                    {/* Active Arc (optional, implies fill) */}
                    {/* Ticks */}
                    {ticks}
                    {/* Redline */}
                    <path
                        d={`M ${cx + (r - 10) * Math.cos(Math.PI * (dangerZone - 1))} ${cy + (r - 10) * Math.sin(Math.PI * (dangerZone - 1))} A ${r - 10} ${r - 10} 0 0 1 ${size - 10} ${r}`}
                        fill="none"
                        stroke="#e74c3c"
                        strokeWidth="4"
                        opacity="0.5"
                    />
                    {/* Needle */}
                    <line
                        x1={cx} y1={r} x2={cx} y2={20}
                        stroke={color}
                        strokeWidth="4"
                        strokeLinecap="round"
                        style={{
                            transformOrigin: `${cx}px ${r}px`,
                            transform: `rotate(${angle}deg)`,
                            transition: 'transform 0.1s cubic-bezier(0.4, 0.0, 0.2, 1)'
                        }}
                    />
                    <circle cx={cx} cy={r} r="6" fill="#f8fafc" />
                </svg>
                <div style={{ position: 'absolute', bottom: 0, width: '100%', textAlign: 'center' }}>
                    <div style={{ fontSize: '1.8rem', fontWeight: 800, lineHeight: 1 }}>{value.toFixed(0)}</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600 }}>{unit}</div>
                </div>
            </div>
        </div>
    );
};

export default function Instruments({ currentV, currentA, currentJ, metrics, limits, isTerminal }) {
    const v_kmh = currentV * 3.6;
    // Accel Gauge: Center zero, range [-2, 2] roughly
    const a_range = 2.0;

    return (
        <div className="instrument-panel">
            <div className="glass-panel">
                <div className="panel-header" style={{ marginBottom: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.9rem', fontWeight: 800, letterSpacing: '0.1em', color: 'var(--text-muted)' }}>COCKPIT</span>
                    <div className={`status-chip ${isTerminal ? 'danger' : 'ok'}`}>
                        {isTerminal ? 'TERMINAL' : 'SYSTEM OK'}
                    </div>
                </div>

                <div className="cockpit-grid">
                    <NeedleGauge
                        label="Speed"
                        value={v_kmh}
                        min={0}
                        max={140}
                        unit="KM/H"
                        size={200}
                        color="var(--primary)"
                    />

                    <NeedleGauge
                        label="Acceleration"
                        value={currentA}
                        min={-2}
                        max={2}
                        unit="m/s²"
                        size={160}
                        dangerZone={0.8} // visual only here
                        color={Math.abs(currentA) > limits.a_max ? 'var(--warning)' : '#10b981'}
                    />
                </div>

                <div className="metrics-grid" style={{ display: 'grid', gap: '8px' }}>
                    <div className="metric-pill">
                        <span>Jerk</span>
                        <strong style={{ color: Math.abs(currentJ) > 0.5 ? 'var(--warning)' : 'var(--text)' }}>
                            {currentJ.toFixed(3)} m/s³
                        </strong>
                    </div>
                    <div className="metric-pill">
                        <span>Delay</span>
                        <strong>{metrics.delay.toFixed(1)} s</strong>
                    </div>
                    <div className="metric-pill">
                        <span>Energy</span>
                        <strong>{metrics.energy_mj.toFixed(2)} MJ</strong>
                    </div>
                </div>
            </div>
        </div>
    );
}

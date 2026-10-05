import React, { useMemo } from 'react';

const STATION_SPACING = 2000; // meters (virtual spacing if not mapped)

export default function TrackView({ currentX, currentV, series, meta, events, cameraMode = 'follow', zoom = 1.0 }) {
    const x_min = meta.x0_m;
    const x_max = meta.xf_m;

    // 1. Camera Logic
    // Wide: Show full route
    // Follow: Center on train
    // Station: Center on nearest station
    // Event: Center on next event?

    let viewBoxX = 0;
    let viewBoxWidth = 400 / zoom; // Base zoom width 400m

    if (cameraMode === 'wide') {
        viewBoxX = x_min - 100;
        viewBoxWidth = (x_max - x_min) + 200;
    } else if (cameraMode === 'follow') {
        viewBoxX = currentX - viewBoxWidth / 2;
    } else if (cameraMode === 'station') {
        // Find nearest station
        // Start or end terminal
        // If x < mid, show the start, else the end
        const mid = (x_min + x_max) / 2;
        const targetX = currentX < mid ? x_min : x_max;
        viewBoxX = targetX - viewBoxWidth / 2;
    } else if (cameraMode === 'event') {
        // Find nearest future event or stay on train
        const nextEvent = events.find(e => e.x_m > currentX);
        if (nextEvent && nextEvent.x_m - currentX < 500) {
            viewBoxX = nextEvent.x_m - viewBoxWidth / 2;
        } else {
            viewBoxX = currentX - viewBoxWidth / 2;
        }
    } else {
        viewBoxX = currentX - viewBoxWidth / 2;
    }

    // 2. Stations Map
    // We map meta.start_station_id -> x_min
    // meta.end_station_id -> x_max
    // Virtual stations in between?
    const startId = meta.start_station_id || "A";
    const endId = meta.end_station_id || "B";

    const stations = [
        { id: startId, x: x_min, label: `Station ${startId}`, type: 'terminal' },
        { id: endId, x: x_max, label: `Station ${endId}`, type: 'terminal' }
    ];

    // 3. Sleepers (Parallax)
    // To achieve parallax, we shift the texture coordinate or the objects based on camera pos
    // But for a simple SVG, we just render world-space objects.
    // "Parallax" usually means background moves slower.
    const sleeperSpacing = 8.0;
    const firstSleeper = Math.floor((viewBoxX - 100) / sleeperSpacing) * sleeperSpacing;
    const sleeperCount = Math.ceil((viewBoxWidth + 200) / sleeperSpacing);
    const sleepers = Array.from({ length: sleeperCount }, (_, i) => firstSleeper + i * sleeperSpacing);

    // 4. Phase / Narrative State
    // Determine active phase for visuals
    const distStart = meta.t_disturb; // Time
    // We need space-based triggers for static rendering, but "Brake Glow" is time-based state?
    // We have 'events' list.
    // Is train braking?
    const isBraking = currentV > 0.1 && series.exec_a_mps2[Math.floor(series.exec_t_s.length * (currentX - x_min) / (x_max - x_min))] < -0.5;
    // Approximation. Better to pass 'currentA' prop.
    // We can infer braking from events too.

    const isDisturbed = events.some(e => e.type === 'stop_brake_start' && Math.abs(currentX - e.x_m) < 200 && currentX <= e.x_m);

    return (
        <div className="track-svg-container" style={{ display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }}>

            {/* TOP: Timeline Ribbon */}
            <div className="timeline-ribbon">
                {/* Segments: Start -> Disturbance -> Stop -> Release -> End */}
                {/* Mapped by time or space? Time is better for playback ribbon. */}
                {/* But we want a "Phase Indicator". Let's render discrete phases if available. */}
                {/* For now, just render the robust ruler from V10 but styled as a ribbon */}
                <div className="ribbon-track">
                    {events.map((e, i) => {
                        // Position strictly by time %? Or Space?
                        // User asked for "Timeline Ribbon... proportional to phase duration"
                        // This implies Time.
                        const pct = (e.t_s / meta.tf_s) * 100;
                        return (
                            <div key={i} className="ribbon-marker" style={{ left: `${pct}%`, borderColor: e.color }}>
                                <div className="ribbon-icon" style={{ background: e.color }} />
                                <span className="ribbon-label">{e.label}</span>
                            </div>
                        )
                    })}
                </div>
            </div>

            {/* MAIN: SVG Logic */}
            <svg
                viewBox={`${viewBoxX} -60 ${viewBoxWidth} 120`}
                preserveAspectRatio="xMidYMid slice"
                style={{ flex: 1, width: '100%', background: '#0f172a' }}
            >
                <defs>
                    <linearGradient id="skyGrad" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0%" stopColor="#0f172a" />
                        <stop offset="100%" stopColor="#1e293b" />
                    </linearGradient>
                    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Sky */}
                <rect x={viewBoxX} y="-60" width={viewBoxWidth} height="120" fill="url(#skyGrad)" />

                {/* Parallax Mountains (Far Background) */}
                <g transform={`translate(${viewBoxX * 0.8}, 0)`}>
                    {/* Fake parallax by shifting opposite to camera? No, fixed object is world space. */}
                    {/* To make it move slower than foreground (viewBox), we need to translate it */}
                    {/* svg viewBox moves 1 unit right. We want bg to move 0.5 units right relative to screen. */}
                    {/* In world space coords, that means shifting it right by 0.5 * viewBoxX */}
                </g>

                {/* Track Bed */}
                <rect x={viewBoxX} y="-20" width={viewBoxWidth} height="40" fill="#1e293b" />

                {/* Sleepers */}
                {sleepers.map(s => (
                    <g key={s}>
                        <rect x={s} y="-15" width="4" height="30" fill="#334155" rx="1" />
                    </g>
                ))}

                {/* Hazardous Track Zone (Disturbance) */}
                {events.filter(e => e.type === 'stop_brake_start').map(e => (
                    <rect key={e.x_m} x={e.x_m} y="-16" width="100" height="32" fill="rgba(231, 76, 60, 0.2)" stroke="var(--danger)" strokeDasharray="10 5">
                        <animate attributeName="opacity" values="0.2;0.5;0.2" dur="1s" repeatCount="indefinite" />
                    </rect>
                ))}

                {/* Rails */}
                <path d={`M ${viewBoxX} -10 L ${viewBoxX + viewBoxWidth} -10`} stroke="#64748b" strokeWidth="2" />
                <path d={`M ${viewBoxX} 10 L ${viewBoxX + viewBoxWidth} 10`} stroke="#64748b" strokeWidth="2" />

                {/* Stations */}
                {stations.map((st, i) => (
                    <g key={i} transform={`translate(${st.x}, 0)`}>
                        {/* Platform */}
                        <rect x="-60" y="-45" width="120" height="20" rx="6" fill="#1e293b" stroke="#475569" strokeWidth="2" />
                        <rect x="-60" y="25" width="120" height="20" rx="6" fill="#1e293b" stroke="#475569" strokeWidth="2" />
                        {/* Label */}
                        <text x="0" y="-55" textAnchor="middle" fill="#f8fafc" fontSize="8" fontWeight="800" fontFamily="sans-serif" style={{ textShadow: '0 2px 4px black' }}>
                            {st.label.toUpperCase()}
                        </text>
                        {/* Dock Marker */}
                        <line x1="0" y1="-10" x2="0" y2="10" stroke="var(--success)" strokeWidth="4" strokeLinecap="round" />
                    </g>
                ))}

                {/* Events Pins (World Space) */}
                {/* 
                {events.map(e => (
                     <g key={e.x_m} transform={`translate(${e.x_m}, 0)`}>
                          <line y1="-30" y2="-15" stroke={e.color} strokeWidth="1" />
                          <circle cy="-32" r="2" fill={e.color} />
                     </g>
                ))}
                */}

                {/* TRAIN Avatar */}
                <g transform={`translate(${currentX}, 0)`}>
                    {/* Brake Glow */}
                    {(isBraking || isDisturbed) && (
                        <circle cx="-50" cy="0" r="30" fill="red" opacity="0.3" filter="url(#glow)">
                            <animate attributeName="opacity" values="0.2;0.6;0.2" dur="0.5s" repeatCount="indefinite" />
                        </circle>
                    )}

                    {/* Body */}
                    <path d="M -50 -14 L 40 -14 C 55 -14 60 -5 60 5 C 60 15 50 14 40 14 L -50 14 Z" fill="#3b82f6" stroke="#2563eb" strokeWidth="1" />
                    <rect x="-50" y="-14" width="5" height="28" fill="#1e3a8a" /> {/* Rear */}

                    {/* Windshield */}
                    <path d="M 40 -12 L 56 0 L 40 0 Z" fill="#93c5fd" opacity="0.8" />

                    {/* Headlight */}
                    <circle cx="58" cy="5" r="2" fill="#fbbf24" filter="url(#glow)" />
                    {/* Beam */}
                    <path d="M 60 5 L 120 25 L 120 -15 Z" fill="url(#headlightGrad)" opacity="0.2" />

                    <defs>
                        <radialGradient id="headlightGrad">
                            <stop offset="0%" stopColor="#fbbf24" stopOpacity="0.5" />
                            <stop offset="100%" stopColor="#fbbf24" stopOpacity="0" />
                        </radialGradient>
                    </defs>
                </g>

            </svg>
        </div>
    );
}

// NOTE: Add this CSS locally or in index.css
/*
.timeline-ribbon {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 48px;
    background: rgba(15, 23, 42, 0.9);
    border-bottom: 1px solid var(--border);
    z-index: 20;
    display: flex;
    align-items: center;
    padding: 0 20px;
    overflow: hidden;
}
.ribbon-track {
    position: relative;
    width: 100%;
    height: 4px;
    background: #334155;
    border-radius: 2px;
}
.ribbon-marker {
    position: absolute;
    top: -6px;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
}
.ribbon-icon {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 2px solid white;
    margin-bottom: 4px;
}
.ribbon-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    white-space: nowrap;
    opacity: 0.7;
}
*/

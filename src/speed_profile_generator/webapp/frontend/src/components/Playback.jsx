import React, { useMemo, useEffect, useState } from 'react';
import Sidebar from './Sidebar';
import TrackView from './TrackView';
import Instruments from './Instruments';

const interp = (xArr, yArr, x) => {
    if (!xArr || !yArr || xArr.length === 0) return 0;
    if (x <= xArr[0]) return yArr[0];
    if (x >= xArr[xArr.length - 1]) return yArr[yArr.length - 1];
    let i = 1;
    while (x > xArr[i]) i++;
    const x0 = xArr[i - 1], x1 = xArr[i];
    const y0 = yArr[i - 1], y1 = yArr[i];
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
};

export default function Playback({ result, status, simT, viewSettings, setViewSettings, setPlaying, setSimT, playbackSpeed, setPlaybackSpeed }) {
    if (!result) {
        if (status === 'running') return <div className="loading">Simulating...</div>;
        return <div className="placeholder">Run a simulation to view playback.</div>;
    }

    const { series, meta, metrics, events } = result;

    // Camera State
    const [cameraMode, setCameraMode] = useState('wide'); // 'wide', 'follow', 'station', 'event'
    const [zoom, setZoom] = useState(1.0);

    // Keyboard Shortcuts
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (status !== 'done') return;
            switch (e.key) {
                case ' ':
                    e.preventDefault();
                    setPlaying(p => !p);
                    break;
                case 'ArrowLeft':
                    setSimT(t => Math.max(0, t - 5));
                    break;
                case 'ArrowRight':
                    setSimT(t => Math.min(meta.tf_s, t + 5));
                    break;
                case '1': setPlaybackSpeed(0.5); break;
                case '2': setPlaybackSpeed(1); break;
                case '3': setPlaybackSpeed(2); break;
                case '4': setPlaybackSpeed(5); break;
                // Camera Shortcuts
                case 'f': setCameraMode('follow'); break;
                case 'w': setCameraMode('wide'); break;
                case 's': setCameraMode('station'); break;
                default: break;
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [status, meta.tf_s, setPlaying, setSimT, setPlaybackSpeed]);

    // 1. High-fidelity interpolation
    const currentX = useMemo(() => interp(series.exec_t_s, series.exec_x_m, simT), [simT, series]);
    const currentV = useMemo(() => interp(series.exec_t_s, series.exec_v_mps, simT), [simT, series]);
    const currentA = useMemo(() => interp(series.exec_t_s, series.exec_a_mps2, simT), [simT, series]);

    // Instant Jerk approximation
    const currentJ = useMemo(() => {
        const t_s = series.exec_t_s;
        const a_s = series.exec_a_mps2;
        if (t_s.length < 2) return 0;
        let i = 1;
        while (simT > t_s[i] && i < t_s.length - 1) i++;
        const da = a_s[i] - a_s[i - 1];
        const dt = t_s[i] - t_s[i - 1];
        return dt > 0 ? da / dt : 0;
    }, [simT, series]);

    // Narrative Logic
    const isTerminal = events.some(e => e.type === 'terminal_start' && simT >= e.t_s);

    return (
        <div className="playback-view">
            {/* 1. Left Sidebar (Shared) */}
            <Sidebar
                onSimulate={() => { }}
                isRunning={status === 'running'}
                viewSettings={viewSettings}
                setViewSettings={setViewSettings}
                hasGT={!!series.gt_v_mps_pos}
                compact={true}
            />

            {/* 2. Center: Large Track Animation */}
            <div className="track-view-panel">
                <div className="panel-header"
                    style={{
                        padding: '16px 24px',
                        borderBottom: '1px solid var(--border)',
                        background: 'rgba(15, 23, 42, 0.6)',
                        backdropFilter: 'blur(4px)',
                        position: 'absolute',
                        top: 0, left: 0, right: 0,
                        zIndex: 30, // Above ribbon
                        display: 'flex',
                        justifyContent: 'space-between'
                    }}>
                    <span style={{ fontSize: '0.8rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                        Live Visualizer
                    </span>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        {result.meta.run_id} / {result.meta.traj_id}
                    </span>
                </div>

                <TrackView
                    currentX={currentX}
                    currentV={currentV}
                    series={series}
                    meta={meta}
                    events={events}
                    cameraMode={cameraMode}
                    zoom={zoom}
                />

                {/* Camera Controls Overlay */}
                <div className="camera-controls">
                    <label>Camera System</label>
                    <div className="camera-row">
                        <select value={cameraMode} onChange={e => setCameraMode(e.target.value)}>
                            <option value="wide">Wide (Route)</option>
                            <option value="follow">Follow Train</option>
                            <option value="station">Station Focus</option>
                            <option value="event">Event Focus</option>
                        </select>
                        <button className="reset-cam-btn" onClick={() => { setCameraMode('wide'); setZoom(1.0); }}>Rest</button>
                    </div>
                    <div className="camera-row" style={{ marginTop: '4px' }}>
                        <span style={{ fontSize: '0.6rem', width: '24px' }}>{zoom}x</span>
                        <input
                            type="range"
                            className="zoom-slider"
                            min="0.5" max="3.0" step="0.1"
                            value={zoom}
                            onChange={e => setZoom(parseFloat(e.target.value))}
                        />
                    </div>
                </div>
            </div>

            {/* 3. Right: Premium Cockpit */}
            <Instruments
                currentV={currentV}
                currentA={currentA}
                currentJ={currentJ}
                metrics={metrics}
                limits={{ a_min: meta.a_min, a_max: meta.a_max }}
                isTerminal={isTerminal}
            />
        </div>
    );
}

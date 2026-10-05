import React, { useState, useEffect } from 'react';

const scenarios = [
    { id: 'none', label: 'Nominal (No disturbance)' },
    { id: 'stop', label: 'Temporary Stop (Disturbance)' },
    { id: 'replan', label: 'Manual Replan' }
];

export default function Sidebar({
    onSimulate,
    isRunning,
    viewSettings,
    setViewSettings,
    hasGT,
    compact = false,
    runs = [],
    selectedRun = '',
    onRunChange,
    trajectories = [],
    selectedTraj = '',
    onTrajChange
}) {
    const [params, setParams] = useState({
        scenario: 'stop',
        respect_limits: false,
        duration_policy: 'keep',
        h5_path: ''
    });

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setParams(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    const updateSetting = (key, val) => {
        setViewSettings(prev => ({ ...prev, [key]: val }));
    };

    // Validation
    const canSimulate = selectedRun && selectedTraj && !isRunning;

    return (
        <div className={`sidebar ${compact ? 'compact' : ''}`}>
            {!compact && (
                <>
                    <h2>Simulation Setup</h2>
                    <div className="form-group">
                        <label>Run ID (Planner Model)</label>
                        <select value={selectedRun} onChange={(e) => onRunChange(e.target.value)}>
                            <option value="">-- Select a Run --</option>
                            {runs.map(r => <option key={r.id} value={r.id}>{r.id}</option>)}
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Trajectory (Scenario Entry)</label>
                        <select value={selectedTraj} onChange={(e) => onTrajChange(e.target.value)} disabled={!selectedRun}>
                            <option value="">-- Select Trajectory --</option>
                            {trajectories.map(t => (
                                <option key={t.id} value={t.id}>{t.id} ({t.duration.toFixed(0)}s)</option>
                            ))}
                        </select>
                        {!selectedRun && <span className="hint">Select a run first</span>}
                    </div>

                    <div className="form-group">
                        <label>Disturbance Scenario</label>
                        <select name="scenario" value={params.scenario} onChange={handleChange}>
                            {scenarios.map(s => <option key={s.id} value={s.id}>{s.label}</option>)}
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Duration Policy</label>
                        <select name="duration_policy" value={params.duration_policy} onChange={handleChange}>
                            <option value="keep">Keep Nominal Duration</option>
                            <option value="stretch">Allow Stretch (Slower)</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                            <input type="checkbox" name="respect_limits" checked={params.respect_limits} onChange={handleChange} />
                            Respect Speed Limits
                        </label>
                    </div>

                    <button
                        className="primary-btn"
                        onClick={() => onSimulate({
                            ...params,
                            run_id: selectedRun,
                            traj_id: selectedTraj,
                            plan_visibility: viewSettings.planDisplay,
                            show_gt: viewSettings.showGT,
                            focus_zoom: viewSettings.focusMode
                        })}
                        disabled={!canSimulate}
                        style={{ opacity: canSimulate ? 1 : 0.5 }}
                    >
                        {isRunning ? 'Simulating...' : 'Run Simulation'}
                    </button>
                    {!canSimulate && !isRunning && (
                        <div className="error-text" style={{ fontSize: '0.7rem', color: 'var(--danger)', marginTop: '8px' }}>
                            Required: Run ID + Trajectory
                        </div>
                    )}

                    <div className="divider" />
                </>
            )}

            <h2>View Settings</h2>

            <div className="form-group">
                <label>Plan Visibility</label>
                <div className="radio-group">
                    <label>
                        <input type="radio" name="planDisplay" value="feasible"
                            checked={viewSettings.planDisplay === 'feasible'}
                            onChange={() => updateSetting('planDisplay', 'feasible')} />
                        Feasible only
                    </label>
                    <label>
                        <input type="radio" name="planDisplay" value="raw"
                            checked={viewSettings.planDisplay === 'raw'}
                            onChange={() => updateSetting('planDisplay', 'raw')} />
                        Raw AI only
                    </label>
                    <label>
                        <input type="radio" name="planDisplay" value="both"
                            checked={viewSettings.planDisplay === 'both'}
                            onChange={() => updateSetting('planDisplay', 'both')} />
                        Both plans
                    </label>
                </div>
            </div>

            <div className="form-group">
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                    <input type="checkbox" checked={viewSettings.showGT}
                        onChange={(e) => updateSetting('showGT', e.target.checked)}
                        disabled={!hasGT} />
                    Show Ground Truth (GT)
                </label>
            </div>

            <div className="form-group">
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                    <input type="checkbox" checked={viewSettings.focusMode}
                        onChange={(e) => updateSetting('focusMode', e.target.checked)} />
                    Focus Zoom
                </label>
            </div>
        </div>
    );
}

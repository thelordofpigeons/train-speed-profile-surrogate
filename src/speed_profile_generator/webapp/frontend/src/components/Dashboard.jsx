import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';

export default function Dashboard({ result, status, viewSettings, simT }) {
    if (!result) {
        if (status === 'running') return <div className="loading">Simulation Running...</div>;
        return <div className="placeholder">Select a run and click Simulate to start the experiment.</div>;
    }

    const { series, events, metrics, meta } = result;
    const { planDisplay, showGT, focusMode } = viewSettings;

    // Interpolate current X for x-axis cursor
    const interpX = (tArr, xArr, t) => {
        if (!tArr || !xArr || tArr.length === 0) return 0;
        if (t <= tArr[0]) return xArr[0];
        if (t >= tArr[tArr.length - 1]) return xArr[xArr.length - 1];
        let i = 1;
        while (t > tArr[i]) i++;
        const t0 = tArr[i - 1], t1 = tArr[i], x0 = xArr[i - 1], x1 = xArr[i];
        return x0 + (t - t0) * (x1 - x0) / (t1 - t0);
    };
    const currentX = interpX(series.exec_t_s, series.exec_x_m, simT);

    // 1. Find Replan Activation
    const replanEvent = events.find(e => e.type === 'replan_start');
    const t_replan = replanEvent ? replanEvent.t_s : (meta.scenario === 'none' ? 0 : Infinity);
    const x_replan = replanEvent ? replanEvent.x_m : (meta.scenario === 'none' ? 0 : Infinity);

    // 2. Alignment & Masking Helper
    const isAligned = (x, y) => x && y && x.length === y.length && x.length > 0;

    // Mask traces before activation
    const getMasked = (xArr, yArr, activationVal) => {
        if (!isAligned(xArr, yArr)) return null;
        const maskedY = yArr.map((v, i) => xArr[i] < activationVal ? null : v);
        return { x: xArr, y: maskedY };
    };

    // 3. Progressive Reveal (Time Domain)
    const getProgressive = (tArr, yArr, tCutoff) => {
        if (!isAligned(tArr, yArr)) return null;
        const idx = tArr.findIndex(t => t > tCutoff);
        const end = idx === -1 ? tArr.length : idx;
        return { x: tArr.slice(0, end), y: yArr.slice(0, end) };
    };

    // 4. Progressive Reveal (Space Domain - Executed only)
    // For Speed vs Position, we slice the executed trace by x corresponding to simT
    const getProgressiveSpatial = (xArr, yArr, xCutoff) => {
        if (!isAligned(xArr, yArr)) return null;
        const idx = xArr.findIndex(x => x > xCutoff);
        const end = idx === -1 ? xArr.length : idx;
        return { x: xArr.slice(0, end), y: yArr.slice(0, end) };
    };

    // 5. Shapes (Markers, Limits, Synced Cursor)
    const getShapes = (axis = 't') => {
        const list = [];
        const coord = axis === 't' ? simT : currentX;

        // Synced Cursor V8
        list.push({
            type: 'line', xref: 'x', yref: 'paper',
            x0: coord, x1: coord, y0: 0, y1: 1,
            line: { color: 'var(--primary)', width: 2 }
        });

        // Events
        events.forEach(e => {
            const evCoord = axis === 't' ? e.t_s : e.x_m;
            if (evCoord <= (axis === 't' ? simT : currentX)) {
                list.push({
                    type: 'line', xref: 'x', yref: 'paper',
                    x0: evCoord, x1: evCoord, y0: 0, y1: 1,
                    line: { color: e.color, width: 2, dash: 'dash' }
                });
            }
        });

        if (axis === 'accel') {
            list.push({
                type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: meta.a_max, y1: meta.a_max,
                line: { color: '#475569', width: 1, dash: 'dash' }
            });
            list.push({
                type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: meta.a_min, y1: meta.a_min,
                line: { color: '#475569', width: 1, dash: 'dash' }
            });
        }
        return list;
    };

    const commonLayout = {
        margin: { l: 60, r: 40, t: 40, b: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(15, 23, 42, 0.4)',
        font: { family: 'Outfit, sans-serif', color: '#f8fafc' },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' },
        autosize: true,
        xaxis: { gridcolor: '#334155', zerolinecolor: '#334155' },
        yaxis: { gridcolor: '#334155', zerolinecolor: '#334155' }
    };

    const v_mps_max = Math.max(...series.v_limit_mps, ...series.exec_v_mps, ...(series.plan_v_mps || []));
    const v_mps_range = [0, isFinite(v_mps_max) ? v_mps_max * 1.1 : 40];
    const t_s_range = [meta.t0_s, meta.tf_s];
    const x_m_range = [meta.x0_m - 20, meta.xf_m + 50];
    const a_mps2_range = [meta.a_min * 1.3, meta.a_max * 1.3];

    let x_view = x_m_range;
    let t_view = t_s_range;
    if (focusMode) {
        x_view = [currentX - 200, currentX + 200];
        t_view = [simT - 10, simT + 10];
    }

    const rawPlanTrace = getMasked(series.plan_x_m, series.plan_v_mps, x_replan);
    const feasiblePlanTrace = getMasked(series.feasible_plan_x_m, series.feasible_plan_v_mps, x_replan);

    const execProgT = getProgressive(series.exec_t_s, series.exec_v_mps, simT);
    const execProgX = getProgressiveSpatial(series.exec_x_m, series.exec_v_mps, currentX);
    const accelProgT = getProgressive(series.exec_t_s, series.exec_a_mps2, simT);

    return (
        <div className="dashboard">
            <div className="metrics-panel">
                <div className="card">
                    <span className="label">Delay</span>
                    <span className="value">{(metrics.delay || 0).toFixed(2)}s</span>
                </div>
                <div className="card">
                    <span className="label">Energy</span>
                    <span className="value">{(metrics.energy_mj || 0).toFixed(2)} MJ</span>
                </div>
                <div className="card">
                    <span className="label">Max Jerk</span>
                    <span className="value">{(result.j_app_max || metrics.max_jerk || 0).toFixed(3)}</span>
                </div>
            </div>

            <div className="plots-container">
                <div className="plot-wrapper">
                    <Plot
                        data={[
                            {
                                x: series.x_limit_m, y: series.v_limit_mps,
                                name: 'Limit', type: 'scatter', mode: 'lines',
                                line: { color: '#ff00ff', width: 2, shape: 'hv' }
                            },
                            showGT && isAligned(series.gt_x_m, series.gt_v_mps_pos) ? {
                                x: series.gt_x_m, y: series.gt_v_mps_pos,
                                name: 'GT', type: 'scatter', mode: 'lines',
                                line: { color: '#2ecc71', width: 2.5 }
                            } : null,
                            (planDisplay === 'raw' || planDisplay === 'both') && rawPlanTrace ? {
                                x: rawPlanTrace.x, y: rawPlanTrace.y,
                                name: 'Plan (Raw)', type: 'scatter', mode: 'lines',
                                line: { color: '#94a3b8', dash: 'dot', width: 1.5 }
                            } : null,
                            (planDisplay === 'feasible' || planDisplay === 'both') && feasiblePlanTrace ? {
                                x: feasiblePlanTrace.x, y: feasiblePlanTrace.y,
                                name: 'Plan (Feasible)', type: 'scatter', mode: 'lines',
                                line: { color: '#f39c12', dash: 'dash', width: 2 }
                            } : null,
                            execProgX ? {
                                x: execProgX.x, y: execProgX.y,
                                name: 'Executed', type: 'scatter', mode: 'lines',
                                line: { color: '#3b82f6', width: 3 }
                            } : null
                        ].filter(t => t !== null)}
                        layout={{ ...commonLayout, title: 'Speed vs Position', xaxis: { title: 'Position (m)', range: x_view }, yaxis: { title: 'Speed (m/s)', range: v_mps_range }, shapes: getShapes('x') }}
                        useResizeHandler={true} style={{ width: "100%", height: "100%" }}
                    />
                </div>

                <div className="plot-wrapper">
                    <Plot
                        data={[
                            showGT && isAligned(series.exec_t_s, series.gt_v_mps_time) ? {
                                x: series.exec_t_s, y: series.gt_v_mps_time,
                                name: 'GT', type: 'scatter', mode: 'lines',
                                line: { color: '#2ecc71', width: 2 }
                            } : null,
                            isAligned(series.exec_t_s, series.ref_v_mps_time) ? {
                                x: series.exec_t_s,
                                y: series.ref_v_mps_time.map((v, i) => series.exec_t_s[i] < t_replan || series.exec_t_s[i] > simT ? null : v),
                                name: 'Reference (Active)', type: 'scatter', mode: 'lines',
                                line: { color: '#28a745', dash: 'dot', width: 2 }
                            } : null,
                            execProgT ? {
                                x: execProgT.x, y: execProgT.y,
                                name: 'Executed', type: 'scatter', mode: 'lines',
                                line: { color: '#3b82f6', width: 3 }
                            } : null
                        ].filter(t => t !== null)}
                        layout={{ ...commonLayout, title: 'Speed vs Time', xaxis: { title: 'Time (s)', range: t_view }, yaxis: { title: 'Speed (m/s)', range: v_mps_range }, shapes: getShapes('t') }}
                        useResizeHandler={true} style={{ width: "100%", height: "100%" }}
                    />
                </div>

                <div className="plot-wrapper">
                    <Plot
                        data={[
                            accelProgT ? {
                                x: accelProgT.x, y: accelProgT.y,
                                name: 'Executed', type: 'scatter', mode: 'lines',
                                line: { color: '#e74c3c', width: 2 }
                            } : null
                        ].filter(t => t !== null)}
                        layout={{ ...commonLayout, title: 'Acceleration vs Time', xaxis: { title: 'Time (s)', range: t_view }, yaxis: { title: 'Accel (m/s²)', range: a_mps2_range }, shapes: getShapes('accel') }}
                        useResizeHandler={true} style={{ width: "100%", height: "100%" }}
                    />
                </div>
            </div>
        </div>
    );
}

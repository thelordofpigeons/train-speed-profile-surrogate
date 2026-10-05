const API_BASE = ''; // Proxy handles base URL

export async function fetchRuns() {
    const res = await fetch(`${API_BASE}/runs`);
    if (!res.ok) throw new Error('Failed to fetch runs');
    return res.json();
}

export async function fetchTrajectories(runId) {
    const res = await fetch(`${API_BASE}/runs/${runId}/trajectories`);
    if (!res.ok) throw new Error('Failed to fetch trajectories');
    return res.json();
}

export async function startSimulation(data) {
    const body = {
        run_id: data.run_id,
        traj_id: data.traj_id,
        scenario: data.scenario,
        respect_limits: data.respect_limits,
        duration_policy: data.duration_policy,
        plan_visibility: data.plan_visibility,
        show_gt: data.show_gt,
        focus_zoom: data.focus_zoom,
        // Config defaults
        // Omitted when empty so the backend default (SPEED_PROFILE_H5_PATH) applies
        h5_path: data.h5_path || undefined,
        downsample_points: 2000,
        animation_points: 600
    };

    const res = await fetch(`${API_BASE}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || errData.error || 'Failed to start simulation');
    }
    return res.json();
}

export async function getJobStatus(jobId) {
    const res = await fetch(`${API_BASE}/jobs/${jobId}`);
    if (!res.ok) {
        if (res.status === 404) throw new Error('404');
        throw new Error('Failed to fetch job status');
    }
    return res.json();
}

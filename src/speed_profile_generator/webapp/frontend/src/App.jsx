import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import Playback from './components/Playback';
import TransportBar from './components/TransportBar';
import { fetchRuns, fetchTrajectories, startSimulation, getJobStatus } from './api';

function App() {
    const [jobId, setJobId] = useState(null);
    const [status, setStatus] = useState('idle');
    const [fullResult, setFullResult] = useState(null);
    const [error, setError] = useState(null);
    const [view, setView] = useState('engineering');

    // Selectors State
    const [runs, setRuns] = useState([]);
    const [selectedRun, setSelectedRun] = useState('');
    const [trajectories, setTrajectories] = useState([]);
    const [selectedTraj, setSelectedTraj] = useState('');

    // GLOBAL PLAYBACK STATE
    const [playing, setPlaying] = useState(false);
    const [simT, setSimT] = useState(0);
    const [playbackSpeed, setPlaybackSpeed] = useState(1.0);

    // View Settings
    const [viewSettings, setViewSettings] = useState({
        focusMode: false,
        planDisplay: 'feasible',
        showGT: false
    });

    const pollIntervalRef = useRef(null);
    const animationRef = useRef(null);
    const lastTickRef = useRef(0);

    // Initial Load: Fetch Runs
    useEffect(() => {
        const loadRuns = async () => {
            try {
                const data = await fetchRuns();
                setRuns(data);
                if (data.length > 0) setSelectedRun(data[0].id);
            } catch (err) {
                console.error("Failed to load runs:", err);
            }
        };
        loadRuns();
    }, []);

    // Load Trajectories when Run changes
    useEffect(() => {
        if (selectedRun) {
            const loadTrajs = async () => {
                try {
                    const data = await fetchTrajectories(selectedRun);
                    setTrajectories(data);
                    if (data.length > 0) setSelectedTraj(data[0].id);
                    else setSelectedTraj('');
                } catch (err) {
                    console.error("Failed to load trajs:", err);
                    setTrajectories([]);
                    setSelectedTraj('');
                }
            };
            loadTrajs();
        }
    }, [selectedRun]);

    const handleSimulate = async (payload) => {
        try {
            setStatus('running');
            setFullResult(null);
            setError(null);
            setSimT(0);
            setPlaying(false);

            const res = await startSimulation(payload);
            setJobId(res.job_id);
        } catch (err) {
            console.error(err);
            setStatus('error');
            setError(err.message);
        }
    };

    // Job Polling
    useEffect(() => {
        if (jobId && status === 'running') {
            pollIntervalRef.current = setInterval(async () => {
                try {
                    const res = await getJobStatus(jobId);
                    if (res.state === 'done') {
                        setFullResult(res.result);
                        setStatus('done');
                        setPlaying(true);
                        clearInterval(pollIntervalRef.current);
                    } else if (res.state === 'error') {
                        setError(res.error);
                        setStatus('error');
                        clearInterval(pollIntervalRef.current);
                    }
                } catch (err) {
                    console.error('Polling error:', err);
                    if (err.message === '404') {
                        setError("Job lost on server. Please try again.");
                        setStatus('error');
                        clearInterval(pollIntervalRef.current);
                    }
                }
            }, 1000);
        }
        return () => clearInterval(pollIntervalRef.current);
    }, [jobId, status]);

    // ANIMATION LOOP
    const maxT = fullResult ? fullResult.meta.tf_s : 0;

    useEffect(() => {
        if (playing && simT < maxT) {
            const animate = (time) => {
                if (!lastTickRef.current) lastTickRef.current = time;
                const delta = (time - lastTickRef.current) / 1000;

                setSimT(t => {
                    const nextT = t + delta * playbackSpeed;
                    if (nextT >= maxT) {
                        setPlaying(false);
                        return maxT;
                    }
                    return nextT;
                });

                lastTickRef.current = time;
                animationRef.current = requestAnimationFrame(animate);
            };
            animationRef.current = requestAnimationFrame(animate);
        } else {
            cancelAnimationFrame(animationRef.current);
            lastTickRef.current = 0;
        }
        return () => cancelAnimationFrame(animationRef.current);
    }, [playing, maxT, playbackSpeed]);

    const onReset = () => {
        setPlaying(false);
        setSimT(0);
    };

    const onScrub = (val) => {
        setPlaying(false);
        setSimT(val);
    };

    return (
        <div className="app-container">
            <Sidebar
                onSimulate={handleSimulate}
                isRunning={status === 'running'}
                viewSettings={viewSettings}
                setViewSettings={setViewSettings}
                hasGT={!!fullResult?.series?.gt_v_mps_pos}
                runs={runs}
                selectedRun={selectedRun}
                onRunChange={setSelectedRun}
                trajectories={trajectories}
                selectedTraj={selectedTraj}
                onTrajChange={setSelectedTraj}
            />

            <div className="main-content">
                <header>
                    <div className="header-left">
                        <h1>VisuLab: Speed Profile Simulator</h1>
                        <nav className="tab-nav">
                            <button className={view === 'engineering' ? 'active' : ''} onClick={() => setView('engineering')}>Engineering Dashboard</button>
                            <button className={view === 'playback' ? 'active' : ''} onClick={() => setView('playback')}>Visual Playback</button>
                        </nav>
                    </div>
                    <div className="status-badge" data-status={status}>{status.toUpperCase()}</div>
                </header>

                {status === 'error' && <div className="error-banner">Error: {error}</div>}

                <div className="view-container">
                    {view === 'engineering' ? (
                        <div className="dashboard-wrapper">
                            <Dashboard
                                result={fullResult}
                                status={status}
                                viewSettings={viewSettings}
                                simT={simT}
                            />
                        </div>
                    ) : (
                        <Playback
                            result={fullResult}
                            status={status}
                            simT={simT}
                            viewSettings={viewSettings}
                            setViewSettings={setViewSettings}
                            playbackSpeed={playbackSpeed}
                        />
                    )}
                </div>

                {fullResult && (
                    <TransportBar
                        simT={simT}
                        maxT={maxT}
                        playing={playing}
                        setPlaying={setPlaying}
                        playbackSpeed={playbackSpeed}
                        setPlaybackSpeed={setPlaybackSpeed}
                        onReset={onReset}
                        onScrub={onScrub}
                        status={status.toUpperCase()}
                    />
                )}
            </div>
        </div>
    );
}

export default App;

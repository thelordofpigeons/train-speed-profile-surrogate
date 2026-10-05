import React from 'react';

export default function TransportBar({
    simT,
    maxT,
    playing,
    setPlaying,
    playbackSpeed,
    setPlaybackSpeed,
    onReset,
    onScrub,
    status
}) {
    const formatTime = (t) => t.toFixed(2).padStart(6, '0');

    return (
        <div className="transport-bar">
            <div className="transport-left">
                <button className="transport-btn" onClick={() => setPlaying(!playing)}>
                    {playing ? '⏸' : '▶'}
                </button>
                <button className="transport-btn" onClick={onReset}>
                    ↺
                </button>
            </div>

            <div className="transport-scrubber">
                <input
                    type="range"
                    min="0"
                    max={maxT}
                    step="0.01"
                    value={simT}
                    onChange={(e) => onScrub(parseFloat(e.target.value))}
                />
                <div className="time-info">
                    <span>{formatTime(simT)}s</span>
                    <span>{formatTime(maxT)}s</span>
                </div>
            </div>

            <div className="transport-right">
                <div className="speed-btns">
                    {[0.5, 1, 2, 5].map(s => (
                        <button
                            key={s}
                            className={playbackSpeed === s ? 'active' : ''}
                            onClick={() => setPlaybackSpeed(s)}
                        >
                            {s}x
                        </button>
                    ))}
                </div>

                <div className="status-pill" data-status={status}>
                    {status}
                </div>
            </div>
        </div>
    );
}

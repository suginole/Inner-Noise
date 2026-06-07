/**
 * AccelerateScreen — Inner Noise
 * Fast GA evolution mode — no rendering, just computation + fitness graph.
 * Design: Void Organism — minimal chrome, data-forward.
 */

import { useState, useEffect, useRef } from 'react';
import { useGameStore } from '../../../store/gameStore';
import FitnessGraph from '../../components/FitnessGraph';

const PRESETS = [
  { label: '×10',   value: 10 },
  { label: '×100',  value: 100 },
  { label: '×1000', value: 1000 },
];

export default function AccelerateScreen() {
  const { generation, history, runAccelerate, accelerateProgress, accelerateTarget } = useGameStore();
  const [isRunning, setIsRunning] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState(10);
  const [containerWidth, setContainerWidth] = useState(700);
  const containerRef = useRef<HTMLDivElement>(null);

  // Responsive graph width
  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(Math.floor(entry.contentRect.width) - 2);
      }
    });
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const handleRun = async () => {
    setIsRunning(true);
    await runAccelerate(selectedPreset);
    setIsRunning(false);
  };

  const progress = isRunning && accelerateTarget > 0
    ? Math.round((accelerateProgress / accelerateTarget) * 100)
    : 0;

  const currentLog = history[history.length - 1];
  const firstLog = history[0];

  // Calculate improvement
  const improvement = currentLog && firstLog
    ? ((currentLog.maxFitness - firstLog.maxFitness) / Math.abs(firstLog.maxFitness || 1) * 100)
    : 0;

  return (
    <div className="flex flex-col h-full gap-5">
      {/* Header stats */}
      <div className="flex items-center gap-8 flex-wrap">
        <div className="flex flex-col">
          <span className="text-[9px] font-mono tracking-widest" style={{ color: 'rgba(255,255,255,0.3)' }}>
            GENERATION
          </span>
          <span className="text-3xl font-mono font-bold" style={{ color: '#00e5c8' }}>
            {String(generation).padStart(5, '0')}
          </span>
        </div>
        {currentLog && (
          <>
            <StatBlock label="MAX FITNESS" value={currentLog.maxFitness.toFixed(3)} color="#00e5c8" />
            <StatBlock label="AVG FITNESS" value={currentLog.avgFitness.toFixed(3)} color="#f5a623" />
            <StatBlock
              label="IMPROVEMENT"
              value={`${improvement >= 0 ? '+' : ''}${improvement.toFixed(1)}%`}
              color={improvement >= 0 ? '#9b5de5' : '#ff6b6b'}
            />
            <StatBlock label="HISTORY" value={`${history.length} gens`} color="rgba(255,255,255,0.4)" />
          </>
        )}
      </div>

      {/* Fitness graph */}
      <div
        ref={containerRef}
        className="rounded overflow-hidden"
        style={{ border: '1px solid rgba(255,255,255,0.06)' }}
      >
        <FitnessGraph history={history} width={containerWidth} height={240} />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        {/* Preset buttons */}
        <div className="flex gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.value}
              onClick={() => setSelectedPreset(p.value)}
              disabled={isRunning}
              className="px-4 py-2 rounded text-sm font-mono transition-all disabled:opacity-40"
              style={{
                background: selectedPreset === p.value ? 'rgba(0,229,200,0.15)' : 'rgba(255,255,255,0.04)',
                color: selectedPreset === p.value ? '#00e5c8' : 'rgba(255,255,255,0.5)',
                border: `1px solid ${selectedPreset === p.value ? 'rgba(0,229,200,0.3)' : 'rgba(255,255,255,0.08)'}`,
              }}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Run button */}
        <button
          onClick={handleRun}
          disabled={isRunning}
          className="px-6 py-2 rounded text-sm font-mono font-bold transition-all disabled:opacity-40"
          style={{
            background: isRunning ? 'rgba(0,229,200,0.05)' : 'rgba(0,229,200,0.18)',
            color: '#00e5c8',
            border: '1px solid rgba(0,229,200,0.35)',
            boxShadow: isRunning ? 'none' : '0 0 16px rgba(0,229,200,0.1)',
          }}
        >
          {isRunning
            ? `▶ EVOLVING... ${progress}% (${accelerateProgress}/${accelerateTarget})`
            : `▶ RUN ${selectedPreset} GENERATIONS`}
        </button>
      </div>

      {/* Progress bar */}
      {isRunning && (
        <div className="w-full h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.05)' }}>
          <div
            className="h-full rounded-full"
            style={{
              width: `${progress}%`,
              background: 'linear-gradient(90deg, #00e5c8, #9b5de5)',
              boxShadow: '0 0 8px rgba(0,229,200,0.4)',
              transition: 'width 200ms ease-out',
            }}
          />
        </div>
      )}

      {/* Info */}
      <div
        className="rounded p-3"
        style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}
      >
        <p className="text-[10px] font-mono" style={{ color: 'rgba(255,255,255,0.25)' }}>
          Audio and rendering are suspended during acceleration. The GA runs synchronously at maximum CPU speed.
          Population: 50 agents · Tournament selection (k=3) · Uniform crossover · Gaussian mutation (σ=0.1, p=0.02)
        </p>
      </div>
    </div>
  );
}

function StatBlock({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[9px] font-mono tracking-widest" style={{ color: 'rgba(255,255,255,0.3)' }}>
        {label}
      </span>
      <span className="text-xl font-mono font-bold" style={{ color }}>
        {value}
      </span>
    </div>
  );
}

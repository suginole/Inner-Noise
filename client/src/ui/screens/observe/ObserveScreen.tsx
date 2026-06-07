/**
 * ObserveScreen — Inner Noise
 * Main game screen: watch the best agent navigate the maze in real-time.
 * Design: Void Organism — dark canvas center, floating instrument panels.
 *
 * Game loop:
 *   - tickObserve() runs at TICK_INTERVAL_MS
 *   - After MAX_FRAMES ticks, runGeneration() is called automatically
 *   - Audio syncs to bottleneck every tick
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { useGameStore } from '../../../store/gameStore';
import MazeCanvas from '../../components/MazeCanvas';
import NeuronDisplay from '../../components/NeuronDisplay';
import { audioEngine } from '../../../audio/audioEngine';
import { NEURON_COLORS } from '../../components/NeuronDisplay';

const TICK_INTERVAL_MS = 100; // 10fps agent movement
const FRAMES_PER_GEN = 500;

export default function ObserveScreen() {
  const {
    grid, activeAgent, activeFoodPositions, generation, history,
    isRunning, isAudioEnabled, tickObserve, runGeneration, setAudioEnabled,
    resetActiveAgent,
  } = useGameStore();

  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const frameCountRef = useRef(0);
  const [audioInitialized, setAudioInitialized] = useState(false);

  const tick = useCallback(() => {
    tickObserve();
    frameCountRef.current++;

    // Advance generation every FRAMES_PER_GEN ticks
    if (frameCountRef.current >= FRAMES_PER_GEN) {
      frameCountRef.current = 0;
      runGeneration();
    }
  }, [tickObserve, runGeneration]);

  // Game loop
  useEffect(() => {
    if (!isRunning) return;
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = setInterval(tick, TICK_INTERVAL_MS);
    return () => {
      if (tickRef.current) clearInterval(tickRef.current);
    };
  }, [isRunning, tick]);

  // Audio sync — update every tick
  useEffect(() => {
    if (!activeAgent || !isAudioEnabled || !audioInitialized) return;
    audioEngine.update(activeAgent.bottleneck);
  }, [activeAgent?.bottleneck, isAudioEnabled, audioInitialized]);

  const handleAudioToggle = async () => {
    if (!audioInitialized) {
      await audioEngine.init();
      await audioEngine.resume();
      setAudioInitialized(true);
    }
    const next = !isAudioEnabled;
    setAudioEnabled(next);
    audioEngine.setMuted(!next);
  };

  const bottleneck = activeAgent?.bottleneck ?? [0, 0, 0, 0];
  const currentLog = history[history.length - 1];
  const foodEaten = activeAgent?.foodEaten ?? 0;
  const fitness = activeAgent?.fitness ?? 0;
  const frame = activeAgent?.frame ?? 0;

  return (
    <div className="flex flex-col h-full gap-4">
      {/* HUD bar */}
      <div className="flex items-center justify-between px-1">
        <div className="flex gap-6">
          <HudItem label="GEN" value={String(generation).padStart(4, '0')} color="#00e5c8" />
          <HudItem label="FITNESS" value={fitness.toFixed(2)} color="#f5a623" />
          <HudItem label="FOOD" value={`${foodEaten}/5`} color="#f5a623" />
          <HudItem label="FRAME" value={`${String(frame).padStart(3, '0')}/${FRAMES_PER_GEN}`} color="rgba(255,255,255,0.4)" />
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={resetActiveAgent}
            className="flex items-center gap-2 px-3 py-1.5 rounded text-xs font-mono transition-all"
            style={{
              background: 'rgba(255,255,255,0.05)',
              color: 'rgba(255,255,255,0.4)',
              border: '1px solid rgba(255,255,255,0.1)',
            }}
          >
            ↺ RESET
          </button>
          <button
            onClick={handleAudioToggle}
            className="flex items-center gap-2 px-3 py-1.5 rounded text-xs font-mono transition-all"
            style={{
              background: isAudioEnabled && audioInitialized ? 'rgba(0,229,200,0.15)' : 'rgba(255,255,255,0.05)',
              color: isAudioEnabled && audioInitialized ? '#00e5c8' : 'rgba(255,255,255,0.4)',
              border: `1px solid ${isAudioEnabled && audioInitialized ? 'rgba(0,229,200,0.3)' : 'rgba(255,255,255,0.1)'}`,
            }}
          >
            {isAudioEnabled && audioInitialized ? '◉ AUDIO ON' : '○ AUDIO OFF'}
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Maze canvas */}
        <div className="flex-1 flex items-center justify-center">
          <div
            className="relative"
            style={{
              boxShadow: '0 0 40px rgba(0,229,200,0.08), 0 0 80px rgba(0,0,0,0.5)',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: '4px',
            }}
          >
            <MazeCanvas
              grid={grid}
              agent={activeAgent}
              foodPositions={activeFoodPositions}
              size={480}
            />
            {/* Generation advance indicator */}
            <div
              className="absolute bottom-2 left-2 right-2 h-0.5 rounded-full overflow-hidden"
              style={{ background: 'rgba(255,255,255,0.05)' }}
            >
              <div
                className="h-full rounded-full"
                style={{
                  width: `${(frame / FRAMES_PER_GEN) * 100}%`,
                  background: 'linear-gradient(90deg, #00e5c8, #9b5de5)',
                  transition: 'width 100ms linear',
                }}
              />
            </div>
          </div>
        </div>

        {/* Right panel */}
        <div className="w-52 flex flex-col gap-3">
          {/* Bottleneck neurons */}
          <Panel title="INNER NOISE">
            <NeuronDisplay bottleneck={bottleneck as [number,number,number,number]} />
          </Panel>

          {/* Waveform visualizer */}
          <Panel title="WAVEFORM">
            <WaveformViz bottleneck={bottleneck as [number,number,number,number]} />
          </Panel>

          {/* Phoneme legend */}
          <Panel title="PHONEME MAP">
            <div className="flex flex-col gap-1.5">
              {NEURON_COLORS.map((n) => (
                <div key={n.name} className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: n.color }} />
                  <div className="flex flex-col">
                    <span className="text-[10px] font-mono" style={{ color: 'rgba(255,255,255,0.5)' }}>
                      {n.label}
                    </span>
                    <span className="text-[9px] font-mono" style={{ color: 'rgba(255,255,255,0.25)' }}>
                      {n.meaning}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Panel>

          {/* Generation stats */}
          {currentLog && (
            <Panel title="STATS">
              <div className="flex flex-col gap-1">
                <StatRow label="MAX FIT" value={currentLog.maxFitness.toFixed(2)} color="#00e5c8" />
                <StatRow label="AVG FIT" value={currentLog.avgFitness.toFixed(2)} color="#f5a623" />
                <StatRow label="POP" value="50" color="rgba(255,255,255,0.4)" />
              </div>
            </Panel>
          )}

          {/* Audio note */}
          {!audioInitialized && (
            <div
              className="rounded p-2 text-[9px] font-mono text-center"
              style={{
                background: 'rgba(245,166,35,0.08)',
                border: '1px solid rgba(245,166,35,0.2)',
                color: 'rgba(245,166,35,0.7)',
              }}
            >
              Click AUDIO OFF to enable sound synthesis
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function HudItem({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[9px] font-mono tracking-widest" style={{ color: 'rgba(255,255,255,0.3)' }}>
        {label}
      </span>
      <span className="text-sm font-mono font-bold" style={{ color }}>
        {value}
      </span>
    </div>
  );
}

function StatRow({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-[10px] font-mono" style={{ color: 'rgba(255,255,255,0.35)' }}>{label}</span>
      <span className="text-[10px] font-mono" style={{ color }}>{value}</span>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      className="rounded p-3"
      style={{
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      <div className="text-[9px] font-mono tracking-widest mb-2" style={{ color: 'rgba(255,255,255,0.25)' }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function WaveformViz({ bottleneck }: { bottleneck: [number, number, number, number] }) {
  const colors = ['#f5a623', '#00e5c8', '#9b5de5', '#ff6b6b'];
  return (
    <div className="flex items-end gap-1 h-12">
      {bottleneck.map((val, i) => (
        <div key={i} className="flex-1 flex flex-col items-center justify-end h-full">
          <div
            className="w-full rounded-t"
            style={{
              height: `${Math.max(4, val * 100)}%`,
              backgroundColor: colors[i],
              opacity: 0.6 + val * 0.4,
              boxShadow: val > 0.5 ? `0 0 6px ${colors[i]}` : 'none',
              transition: 'height 80ms ease-out',
            }}
          />
        </div>
      ))}
    </div>
  );
}

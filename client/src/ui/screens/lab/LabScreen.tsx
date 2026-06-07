/**
 * LabScreen — Inner Noise
 * Analytical mode: inject forced bottleneck values, observe behavior.
 * Features: Piano roll timeline, per-neuron sliders, auto-labeling.
 * Design: Void Organism — instrument panel aesthetic.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { useGameStore } from '../../../store/gameStore';
import MazeCanvas from '../../components/MazeCanvas';
import NeuronDisplay from '../../components/NeuronDisplay';
import PianoRoll from '../../components/PianoRoll';
import { audioEngine } from '../../../audio/audioEngine';
import { NEURON_COLORS } from '../../components/NeuronDisplay';

const TICK_INTERVAL_MS = 150;

export default function LabScreen() {
  const {
    grid, activeAgent, activeFoodPositions, labBottleneckLog,
    forcedBottleneck, setForcedBottleneck, tickObserve, isAudioEnabled,
  } = useGameStore();

  const [localBN, setLocalBN] = useState<[number, number, number, number]>([0, 0, 0, 0]);
  const [isForcedActive, setIsForcedActive] = useState(false);
  const [audioInitialized, setAudioInitialized] = useState(false);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Game loop
  const tick = useCallback(() => {
    tickObserve();
  }, [tickObserve]);

  useEffect(() => {
    tickRef.current = setInterval(tick, TICK_INTERVAL_MS);
    return () => {
      if (tickRef.current) clearInterval(tickRef.current);
    };
  }, [tick]);

  // Audio sync
  useEffect(() => {
    if (!activeAgent || !isAudioEnabled || !audioInitialized) return;
    const bn = isForcedActive ? localBN : activeAgent.bottleneck;
    audioEngine.update(bn);
  }, [activeAgent?.bottleneck, isAudioEnabled, audioInitialized, isForcedActive, localBN]);

  // Update forced bottleneck in store
  useEffect(() => {
    if (isForcedActive) {
      setForcedBottleneck(localBN);
    } else {
      setForcedBottleneck(null);
    }
  }, [localBN, isForcedActive, setForcedBottleneck]);

  const handleAudioInit = async () => {
    if (!audioInitialized) {
      await audioEngine.init();
      await audioEngine.resume();
      setAudioInitialized(true);
    }
  };

  const bottleneck = isForcedActive ? localBN : (activeAgent?.bottleneck ?? [0, 0, 0, 0] as [number,number,number,number]);

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-mono font-bold" style={{ color: '#9b5de5' }}>
            LAB MODE — INTERNAL REPRESENTATION ANALYSIS
          </h2>
          <p className="text-[10px] font-mono mt-0.5" style={{ color: 'rgba(255,255,255,0.3)' }}>
            Inject forced activation values to observe behavioral responses (Human-in-the-Loop)
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleAudioInit}
            className="px-3 py-1.5 rounded text-xs font-mono transition-all"
            style={{
              background: audioInitialized ? 'rgba(0,229,200,0.1)' : 'rgba(255,255,255,0.05)',
              color: audioInitialized ? '#00e5c8' : 'rgba(255,255,255,0.4)',
              border: `1px solid ${audioInitialized ? 'rgba(0,229,200,0.2)' : 'rgba(255,255,255,0.1)'}`,
            }}
          >
            {audioInitialized ? '◉ AUDIO' : '○ INIT AUDIO'}
          </button>
          <button
            onClick={() => setIsForcedActive(!isForcedActive)}
            className="px-3 py-1.5 rounded text-xs font-mono font-bold transition-all"
            style={{
              background: isForcedActive ? 'rgba(155,93,229,0.2)' : 'rgba(255,255,255,0.05)',
              color: isForcedActive ? '#9b5de5' : 'rgba(255,255,255,0.4)',
              border: `1px solid ${isForcedActive ? 'rgba(155,93,229,0.4)' : 'rgba(255,255,255,0.1)'}`,
              boxShadow: isForcedActive ? '0 0 12px rgba(155,93,229,0.2)' : 'none',
            }}
          >
            {isForcedActive ? '⚡ FORCED ACTIVE' : '○ FORCE INACTIVE'}
          </button>
        </div>
      </div>

      {/* Main layout */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Left: maze + neuron display */}
        <div className="flex flex-col gap-3 shrink-0">
          <div
            className="relative"
            style={{
              boxShadow: isForcedActive
                ? '0 0 30px rgba(155,93,229,0.15)'
                : '0 0 20px rgba(0,229,200,0.05)',
              border: `1px solid ${isForcedActive ? 'rgba(155,93,229,0.2)' : 'rgba(255,255,255,0.06)'}`,
              borderRadius: '4px',
            }}
          >
            <MazeCanvas
              grid={grid}
              agent={activeAgent}
              foodPositions={activeFoodPositions}
              size={300}
            />
            {isForcedActive && (
              <div
                className="absolute top-2 left-2 text-[9px] font-mono px-2 py-0.5 rounded"
                style={{ background: 'rgba(155,93,229,0.85)', color: 'white' }}
              >
                ⚡ FORCED INPUT
              </div>
            )}
          </div>

          <div
            className="rounded p-3"
            style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}
          >
            <div className="text-[9px] font-mono tracking-widest mb-2" style={{ color: 'rgba(255,255,255,0.25)' }}>
              CURRENT ACTIVATION
            </div>
            <NeuronDisplay bottleneck={bottleneck as [number,number,number,number]} />
          </div>

          {/* Auto-labeling */}
          <div
            className="rounded p-3"
            style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}
          >
            <div className="text-[9px] font-mono tracking-widest mb-2" style={{ color: 'rgba(255,255,255,0.25)' }}>
              AUTO-LABEL — F22
            </div>
            <div className="grid grid-cols-4 gap-1.5">
              {NEURON_COLORS.map((n, i) => {
                const activation = bottleneck[i];
                return (
                  <div key={n.name} className="flex flex-col items-center gap-1">
                    <div
                      className="w-7 h-7 rounded-full flex items-center justify-center text-[9px] font-mono font-bold"
                      style={{
                        background: activation > 0.5 ? `rgba(${hexToRgb(n.color)},0.2)` : 'rgba(255,255,255,0.04)',
                        border: `1px solid ${activation > 0.5 ? n.color : 'rgba(255,255,255,0.08)'}`,
                        color: n.color,
                        boxShadow: activation > 0.7 ? `0 0 8px ${n.color}` : 'none',
                        transition: 'all 150ms ease',
                      }}
                    >
                      {n.name}
                    </div>
                    <span className="text-[8px] font-mono text-center leading-tight" style={{ color: 'rgba(255,255,255,0.35)' }}>
                      {activation > 0.7 ? n.meaning : '—'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right: sliders + piano roll */}
        <div className="flex-1 flex flex-col gap-3 min-w-0">
          {/* Forced pulse sliders */}
          <div
            className="rounded p-4"
            style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}
          >
            <div className="text-[9px] font-mono tracking-widest mb-3" style={{ color: 'rgba(255,255,255,0.25)' }}>
              FORCED PULSE INPUT — F20
            </div>
            <div className="flex flex-col gap-4">
              {NEURON_COLORS.map((n, i) => (
                <div key={n.name} className="flex items-center gap-3">
                  <span className="text-xs font-mono w-6 shrink-0" style={{ color: n.color }}>
                    {n.name}
                  </span>
                  <div className="flex-1 flex items-center gap-2">
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={Math.round(localBN[i] * 100)}
                      onChange={(e) => {
                        const val = parseInt(e.target.value) / 100;
                        setLocalBN((prev) => {
                          const next = [...prev] as [number, number, number, number];
                          next[i] = val;
                          return next;
                        });
                      }}
                      className="flex-1 cursor-pointer"
                      style={{
                        accentColor: n.color,
                      }}
                    />
                    <span className="text-xs font-mono w-10 text-right shrink-0" style={{ color: n.color }}>
                      {localBN[i].toFixed(2)}
                    </span>
                  </div>
                  <div className="w-28 flex flex-col text-right shrink-0">
                    <span className="text-[9px] font-mono" style={{ color: 'rgba(255,255,255,0.35)' }}>
                      {n.label}
                    </span>
                    <span className="text-[8px] font-mono" style={{ color: 'rgba(255,255,255,0.2)' }}>
                      {n.meaning}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Quick presets */}
            <div className="flex gap-2 mt-3 flex-wrap">
              {[
                { label: 'ALL ZERO', values: [0,0,0,0] as [number,number,number,number], color: 'rgba(255,255,255,0.3)' },
                { label: 'ALL MAX',  values: [1,1,1,1] as [number,number,number,number], color: '#9b5de5' },
                { label: 'FOOD',     values: [1,0,0.8,0] as [number,number,number,number], color: '#f5a623' },
                { label: 'AVOID',    values: [0,0.8,0,1] as [number,number,number,number], color: '#ff6b6b' },
                { label: 'FORWARD',  values: [0.3,0,1,0] as [number,number,number,number], color: '#9b5de5' },
              ].map((p) => (
                <button
                  key={p.label}
                  onClick={() => setLocalBN(p.values)}
                  className="px-2 py-1 rounded text-[10px] font-mono transition-all"
                  style={{
                    background: 'rgba(255,255,255,0.04)',
                    color: p.color,
                    border: '1px solid rgba(255,255,255,0.08)',
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Piano roll */}
          <div
            className="rounded p-3 flex-1"
            style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', minHeight: '180px' }}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="text-[9px] font-mono tracking-widest" style={{ color: 'rgba(255,255,255,0.25)' }}>
                PHONEME TIMELINE — F19
              </div>
              <div className="flex gap-2">
                {NEURON_COLORS.map((n) => (
                  <div key={n.name} className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: n.color, opacity: 0.8 }} />
                    <span className="text-[9px] font-mono" style={{ color: 'rgba(255,255,255,0.3)' }}>{n.name}</span>
                  </div>
                ))}
              </div>
            </div>
            <PianoRoll log={labBottleneckLog} width={520} height={160} />
            <div className="mt-1 text-[9px] font-mono" style={{ color: 'rgba(255,255,255,0.2)' }}>
              {labBottleneckLog.length} frames recorded · last 300 shown
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `${r},${g},${b}`;
}

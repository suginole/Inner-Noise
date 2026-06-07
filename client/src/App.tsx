/**
 * App — Inner Noise
 * Root layout with top navigation bar and mode routing.
 * Design: Void Organism — dark void, minimal chrome, mode tabs as instrument selectors.
 */

import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "./contexts/ThemeContext";
import ErrorBoundary from "./components/ErrorBoundary";
import { useGameStore, GameMode } from "./store/gameStore";
import { useEffect } from "react";
import ObserveScreen from "./ui/screens/observe/ObserveScreen";
import AccelerateScreen from "./ui/screens/accelerate/AccelerateScreen";
import LabScreen from "./ui/screens/lab/LabScreen";

const MODES: { id: GameMode; label: string; shortcut: string; color: string }[] = [
  { id: 'observe',    label: 'OBSERVE',    shortcut: 'O', color: '#00e5c8' },
  { id: 'accelerate', label: 'ACCELERATE', shortcut: 'A', color: '#f5a623' },
  { id: 'lab',        label: 'LAB',        shortcut: 'L', color: '#9b5de5' },
];

function App() {
  const { mode, setMode, loadFromStorage, generation } = useGameStore();

  useEffect(() => {
    loadFromStorage();
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === 'o' || e.key === 'O') setMode('observe');
      if (e.key === 'a' || e.key === 'A') setMode('accelerate');
      if (e.key === 'l' || e.key === 'L') setMode('lab');
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [setMode]);

  const currentMode = MODES.find((m) => m.id === mode)!;

  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="dark">
        <TooltipProvider>
          <Toaster />
          <div
            className="min-h-screen flex flex-col"
            style={{ background: '#080c10', color: 'rgba(255,255,255,0.85)' }}
          >
            {/* Top navigation bar */}
            <header
              className="flex items-center justify-between px-6 py-3 shrink-0"
              style={{
                borderBottom: '1px solid rgba(255,255,255,0.06)',
                background: 'rgba(8,12,16,0.95)',
                backdropFilter: 'blur(8px)',
              }}
            >
              {/* Logo */}
              <div className="flex items-center gap-3">
                <div
                  className="w-7 h-7 rounded flex items-center justify-center text-xs font-mono font-bold"
                  style={{
                    background: `rgba(${hexToRgb(currentMode.color)},0.15)`,
                    border: `1px solid rgba(${hexToRgb(currentMode.color)},0.3)`,
                    color: currentMode.color,
                    boxShadow: `0 0 12px rgba(${hexToRgb(currentMode.color)},0.2)`,
                    transition: 'all 300ms ease',
                  }}
                >
                  IN
                </div>
                <div>
                  <div className="text-sm font-bold tracking-wide" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                    Inner Noise
                  </div>
                  <div className="text-[9px] font-mono" style={{ color: 'rgba(255,255,255,0.25)' }}>
                    GA × NN × EMERGENT COMMUNICATION
                  </div>
                </div>
              </div>

              {/* Mode tabs */}
              <nav className="flex items-center gap-1">
                {MODES.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setMode(m.id)}
                    className="px-4 py-1.5 rounded text-xs font-mono tracking-widest transition-all"
                    style={{
                      background: mode === m.id ? `rgba(${hexToRgb(m.color)},0.12)` : 'transparent',
                      color: mode === m.id ? m.color : 'rgba(255,255,255,0.35)',
                      border: `1px solid ${mode === m.id ? `rgba(${hexToRgb(m.color)},0.3)` : 'transparent'}`,
                      boxShadow: mode === m.id ? `0 0 12px rgba(${hexToRgb(m.color)},0.1)` : 'none',
                    }}
                  >
                    {m.label}
                    <span className="ml-1.5 text-[9px] opacity-40">[{m.shortcut}]</span>
                  </button>
                ))}
              </nav>

              {/* Generation counter */}
              <div className="flex items-center gap-2">
                <span className="text-[9px] font-mono tracking-widest" style={{ color: 'rgba(255,255,255,0.25)' }}>
                  GEN
                </span>
                <span className="text-sm font-mono font-bold" style={{ color: '#00e5c8' }}>
                  {String(generation).padStart(5, '0')}
                </span>
              </div>
            </header>

            {/* Main content */}
            <main className="flex-1 p-6 overflow-auto">
              {mode === 'observe' && <ObserveScreen />}
              {mode === 'accelerate' && <AccelerateScreen />}
              {mode === 'lab' && <LabScreen />}
            </main>

            {/* Footer */}
            <footer
              className="px-6 py-2 flex items-center justify-between shrink-0"
              style={{
                borderTop: '1px solid rgba(255,255,255,0.04)',
                background: 'rgba(8,12,16,0.8)',
              }}
            >
              <span className="text-[9px] font-mono" style={{ color: 'rgba(255,255,255,0.15)' }}>
                Information Bottleneck Theory · Tishby et al. 2000 · Emergent Communication · Lazaridou & Baroni 2020
              </span>
              <span className="text-[9px] font-mono" style={{ color: 'rgba(255,255,255,0.15)' }}>
                v0.1.0 · Web Audio API · GA+NN · 50 agents
              </span>
            </footer>
          </div>
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `${r},${g},${b}`;
}

export default App;

/**
 * MazeCanvas — Inner Noise
 * Renders the 16×16 maze with dot-pixel aesthetic.
 * Design: Void Organism — dark void, neuron accent colors for agent glow.
 */

import { useEffect, useRef } from 'react';
import { Cell, MAZE_SIZE } from '../../core/maze/maze';
import { AgentState } from '../../core/agent/agent';

const COLORS = {
  bg:      '#080c10',
  wall:    '#0d1520',
  wallInner: '#0a1018',
  path:    '#0f1825',
  start:   '#0f2018',
  end:     '#0f1828',
  food:    '#f5a623',
  agent:   '#00e5c8',
  endMark: '#00e5c8',
  gridLine:'rgba(255,255,255,0.025)',
};

interface Props {
  grid: Cell[][];
  agent: AgentState | null;
  foodPositions: { x: number; y: number }[];
  size?: number;
}

export default function MazeCanvas({ grid, agent, foodPositions, size = 512 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trailRef = useRef<{ x: number; y: number }[]>([]);
  const lastAgentIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!canvasRef.current || grid.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cellSize = size / MAZE_SIZE;

    // Reset trail on agent change
    if (agent && agent.id !== lastAgentIdRef.current) {
      trailRef.current = [];
      lastAgentIdRef.current = agent.id;
    }

    // Update trail
    if (agent) {
      const trail = trailRef.current;
      const pos = agent.position;
      const last = trail[trail.length - 1];
      if (!last || last.x !== pos.x || last.y !== pos.y) {
        trail.push({ ...pos });
        if (trail.length > 24) trail.shift();
      }
    }

    // Clear
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, size, size);

    // Draw cells
    for (let y = 0; y < MAZE_SIZE; y++) {
      for (let x = 0; x < MAZE_SIZE; x++) {
        const cell = grid[y][x];
        const px = x * cellSize;
        const py = y * cellSize;

        if (cell.type === 'wall') {
          ctx.fillStyle = COLORS.wall;
          ctx.fillRect(px, py, cellSize, cellSize);
          ctx.fillStyle = COLORS.wallInner;
          ctx.fillRect(px + 1, py + 1, cellSize - 2, cellSize - 2);
        } else {
          ctx.fillStyle = cell.type === 'start' ? COLORS.start
            : cell.type === 'end' ? COLORS.end
            : COLORS.path;
          ctx.fillRect(px, py, cellSize, cellSize);

          // End marker: corner brackets
          if (cell.type === 'end') {
            const m = 3;
            const l = cellSize * 0.35;
            ctx.strokeStyle = COLORS.endMark;
            ctx.lineWidth = 1.5;
            ctx.globalAlpha = 0.6;
            // Top-left
            ctx.beginPath(); ctx.moveTo(px + m, py + m + l); ctx.lineTo(px + m, py + m); ctx.lineTo(px + m + l, py + m); ctx.stroke();
            // Top-right
            ctx.beginPath(); ctx.moveTo(px + cellSize - m - l, py + m); ctx.lineTo(px + cellSize - m, py + m); ctx.lineTo(px + cellSize - m, py + m + l); ctx.stroke();
            // Bottom-left
            ctx.beginPath(); ctx.moveTo(px + m, py + cellSize - m - l); ctx.lineTo(px + m, py + cellSize - m); ctx.lineTo(px + m + l, py + cellSize - m); ctx.stroke();
            // Bottom-right
            ctx.beginPath(); ctx.moveTo(px + cellSize - m - l, py + cellSize - m); ctx.lineTo(px + cellSize - m, py + cellSize - m); ctx.lineTo(px + cellSize - m, py + cellSize - m - l); ctx.stroke();
            ctx.globalAlpha = 1;
          }
        }

        // Food
        if (cell.hasFood && foodPositions.some((fp) => fp.x === x && fp.y === y)) {
          const cx = px + cellSize / 2;
          const cy = py + cellSize / 2;
          const r = cellSize * 0.2;

          // Glow
          const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 3);
          grad.addColorStop(0, 'rgba(245,166,35,0.5)');
          grad.addColorStop(1, 'rgba(245,166,35,0)');
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(cx, cy, r * 3, 0, Math.PI * 2);
          ctx.fill();

          // Core dot
          ctx.fillStyle = COLORS.food;
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Subtle grid lines
    ctx.strokeStyle = COLORS.gridLine;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= MAZE_SIZE; i++) {
      ctx.beginPath(); ctx.moveTo(i * cellSize, 0); ctx.lineTo(i * cellSize, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cellSize); ctx.lineTo(size, i * cellSize); ctx.stroke();
    }

    // Draw trail
    if (agent) {
      const trail = trailRef.current;
      for (let i = 0; i < trail.length - 1; i++) {
        const alpha = (i / trail.length) * 0.35;
        const tp = trail[i];
        ctx.fillStyle = `rgba(0,229,200,${alpha})`;
        const pad = cellSize * 0.3;
        ctx.fillRect(tp.x * cellSize + pad, tp.y * cellSize + pad, cellSize - pad * 2, cellSize - pad * 2);
      }

      // Draw agent
      const ax = agent.position.x * cellSize + cellSize / 2;
      const ay = agent.position.y * cellSize + cellSize / 2;
      const ar = cellSize * 0.32;

      // Outer glow
      const agentGrad = ctx.createRadialGradient(ax, ay, 0, ax, ay, ar * 3);
      agentGrad.addColorStop(0, 'rgba(0,229,200,0.45)');
      agentGrad.addColorStop(0.5, 'rgba(0,229,200,0.1)');
      agentGrad.addColorStop(1, 'rgba(0,229,200,0)');
      ctx.fillStyle = agentGrad;
      ctx.beginPath();
      ctx.arc(ax, ay, ar * 3, 0, Math.PI * 2);
      ctx.fill();

      // Agent body
      ctx.fillStyle = COLORS.agent;
      ctx.beginPath();
      ctx.arc(ax, ay, ar, 0, Math.PI * 2);
      ctx.fill();

      // Inner highlight
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.beginPath();
      ctx.arc(ax - ar * 0.28, ay - ar * 0.28, ar * 0.28, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      style={{ imageRendering: 'pixelated', display: 'block' }}
      className="rounded-sm"
    />
  );
}

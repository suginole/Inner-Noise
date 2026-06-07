/**
 * PianoRoll — Inner Noise
 * Displays bottleneck activation timeline (horizontal time, vertical N1–N4).
 * Design: Void Organism — each neuron row uses its accent color.
 */

import { useEffect, useRef } from 'react';
import { NEURON_COLORS } from './NeuronDisplay';

interface Props {
  log: number[][];  // Array of [n1,n2,n3,n4] frames
  width?: number;
  height?: number;
}

const ROW_LABELS = ['N1', 'N2', 'N3', 'N4'];

export default function PianoRoll({ log, width = 600, height = 160 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    const labelW = 28;
    const drawW = width - labelW;
    const rowH = height / 4;
    const THRESHOLD = 0.15;

    // Background
    ctx.fillStyle = '#080c10';
    ctx.fillRect(0, 0, width, height);

    // Row backgrounds
    for (let r = 0; r < 4; r++) {
      ctx.fillStyle = r % 2 === 0 ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0)';
      ctx.fillRect(labelW, r * rowH, drawW, rowH);
    }

    // Row labels
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    for (let r = 0; r < 4; r++) {
      ctx.fillStyle = NEURON_COLORS[r].color;
      ctx.fillText(ROW_LABELS[r], labelW / 2, r * rowH + rowH / 2 + 4);
    }

    if (log.length === 0) return;

    // Draw activation blocks
    const frameW = Math.max(1, drawW / Math.min(log.length, 300));
    const startIdx = Math.max(0, log.length - 300);

    for (let i = startIdx; i < log.length; i++) {
      const frame = log[i];
      const fx = labelW + (i - startIdx) * frameW;

      for (let r = 0; r < 4; r++) {
        const val = frame[r] ?? 0;
        if (val < THRESHOLD) continue;

        const alpha = Math.min(1, val * 1.2);
        const color = NEURON_COLORS[r].color;

        // Parse hex to rgba
        const hex = color.replace('#', '');
        const rr = parseInt(hex.substring(0, 2), 16);
        const gg = parseInt(hex.substring(2, 4), 16);
        const bb = parseInt(hex.substring(4, 6), 16);

        ctx.fillStyle = `rgba(${rr},${gg},${bb},${alpha})`;
        const blockH = rowH * val * 0.85;
        ctx.fillRect(
          fx,
          r * rowH + (rowH - blockH) / 2,
          Math.max(frameW - 0.5, 1),
          blockH
        );
      }
    }

    // Playhead (rightmost position)
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(width - 1, 0);
    ctx.lineTo(width - 1, height);
    ctx.stroke();

    // Separator lines between rows
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 0.5;
    for (let r = 1; r < 4; r++) {
      ctx.beginPath();
      ctx.moveTo(labelW, r * rowH);
      ctx.lineTo(width, r * rowH);
      ctx.stroke();
    }

  }, [log, width, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: `${width}px`, height: `${height}px` }}
      className="rounded-sm"
    />
  );
}

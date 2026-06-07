/**
 * FitnessGraph — Inner Noise
 * Canvas-based line chart showing avg/max fitness over generations.
 * Design: Void Organism — teal for max, amber for avg, dark void background.
 */

import { useEffect, useRef } from 'react';
import { GenerationLog } from '../../store/gameStore';

interface Props {
  history: GenerationLog[];
  width?: number;
  height?: number;
}

export default function FitnessGraph({ history, width = 600, height = 200 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length < 2) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    const pad = { top: 16, right: 16, bottom: 32, left: 48 };
    const w = width - pad.left - pad.right;
    const h = height - pad.top - pad.bottom;

    // Background
    ctx.fillStyle = '#080c10';
    ctx.fillRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (h / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + w, y);
      ctx.stroke();
    }

    const maxVal = Math.max(...history.map((h) => h.maxFitness), 1);
    const minVal = Math.min(...history.map((h) => h.avgFitness), 0);
    const range = maxVal - minVal || 1;

    const toX = (i: number) => pad.left + (i / (history.length - 1)) * w;
    const toY = (v: number) => pad.top + h - ((v - minVal) / range) * h;

    // Draw avg fitness line
    ctx.beginPath();
    ctx.strokeStyle = '#f5a623';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    history.forEach((log, i) => {
      if (i === 0) ctx.moveTo(toX(i), toY(log.avgFitness));
      else ctx.lineTo(toX(i), toY(log.avgFitness));
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw max fitness line
    ctx.beginPath();
    ctx.strokeStyle = '#00e5c8';
    ctx.lineWidth = 2;
    history.forEach((log, i) => {
      if (i === 0) ctx.moveTo(toX(i), toY(log.maxFitness));
      else ctx.lineTo(toX(i), toY(log.maxFitness));
    });
    ctx.stroke();

    // Area fill under max
    ctx.beginPath();
    history.forEach((log, i) => {
      if (i === 0) ctx.moveTo(toX(i), toY(log.maxFitness));
      else ctx.lineTo(toX(i), toY(log.maxFitness));
    });
    ctx.lineTo(toX(history.length - 1), pad.top + h);
    ctx.lineTo(toX(0), pad.top + h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + h);
    grad.addColorStop(0, 'rgba(0,229,200,0.15)');
    grad.addColorStop(1, 'rgba(0,229,200,0)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Y axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const v = minVal + (range / 4) * (4 - i);
      ctx.fillText(v.toFixed(1), pad.left - 6, pad.top + (h / 4) * i + 4);
    }

    // X axis labels
    ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(history.length / 5));
    for (let i = 0; i < history.length; i += step) {
      ctx.fillText(`${history[i].generation}`, toX(i), pad.top + h + 18);
    }

    // Legend
    ctx.textAlign = 'left';
    ctx.fillStyle = '#00e5c8';
    ctx.fillRect(pad.left, 4, 12, 3);
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText('Max', pad.left + 16, 10);

    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = '#f5a623';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pad.left + 80, 6);
    ctx.lineTo(pad.left + 92, 6);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText('Avg', pad.left + 96, 10);

  }, [history, width, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: `${width}px`, height: `${height}px` }}
      className="rounded-sm"
    />
  );
}

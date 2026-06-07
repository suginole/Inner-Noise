/**
 * NeuronDisplay — Inner Noise
 * Shows the 4 bottleneck neuron activation values as glowing bars.
 * Design: Void Organism — each neuron has its own accent color.
 */

interface Props {
  bottleneck: [number, number, number, number];
  size?: 'sm' | 'md' | 'lg';
}

const NEURON_COLORS = [
  { name: 'N1', color: '#f5a623', label: 'Open Vowel', meaning: 'Food / Positive' },
  { name: 'N2', color: '#00e5c8', label: 'Front Vowel', meaning: 'Threat / Negative' },
  { name: 'N3', color: '#9b5de5', label: 'Fricative',   meaning: 'Approach' },
  { name: 'N4', color: '#ff6b6b', label: 'Plosive',     meaning: 'Avoidance' },
];

export default function NeuronDisplay({ bottleneck, size = 'md' }: Props) {
  const heights = size === 'sm' ? 'h-1.5' : size === 'lg' ? 'h-3' : 'h-2';
  const textSize = size === 'sm' ? 'text-[10px]' : 'text-xs';

  return (
    <div className="flex flex-col gap-2">
      {NEURON_COLORS.map((n, i) => {
        const val = bottleneck[i];
        const pct = Math.round(val * 100);
        const isActive = val > 0.7;

        return (
          <div key={n.name} className="flex items-center gap-2">
            {/* Label */}
            <span
              className={`${textSize} font-mono w-6 shrink-0`}
              style={{ color: n.color }}
            >
              {n.name}
            </span>

            {/* Bar track */}
            <div className={`flex-1 bg-white/5 rounded-full ${heights} overflow-hidden relative`}>
              <div
                className={`${heights} rounded-full transition-all`}
                style={{
                  width: `${pct}%`,
                  backgroundColor: n.color,
                  boxShadow: isActive ? `0 0 8px ${n.color}` : 'none',
                  transition: 'width 80ms ease-out',
                }}
              />
            </div>

            {/* Value */}
            <span
              className={`${textSize} font-mono w-8 text-right shrink-0`}
              style={{ color: isActive ? n.color : 'rgba(255,255,255,0.4)' }}
            >
              {val.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export { NEURON_COLORS };

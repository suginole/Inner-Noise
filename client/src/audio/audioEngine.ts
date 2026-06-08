/**
 * Audio Engine — Inner Noise  v0.2.0
 *
 * ============================================================
 * ARCHITECTURE: Source-Filter Theory (Fant, 1960)
 * ============================================================
 *
 * WRONG (previous):
 *   N_i → OscillatorNode.frequency   ← changes PITCH per neuron → chord noise
 *
 * CORRECT (this implementation):
 *   Single SOURCE  : 1 sawtooth oscillator @ fixed 150 Hz  (glottis / vocal folds)
 *   FILTER MATRIX  : 3 BiquadFilter (bandpass) nodes in parallel
 *                    N1, N2, N3 control each filter's GAIN only  (formant shaping)
 *   CONSONANT BURST: N4 Δ-threshold triggers a 5ms white-noise burst  (plosive/fricative onset)
 *
 * Signal graph:
 *
 *   OscillatorNode (sawtooth, 150 Hz, fixed)
 *       │
 *       ├──> BPF_1 (center ~1000 Hz, [a] formant)  × GainNode_1  ← N1 gain
 *       ├──> BPF_2 (center ~2500 Hz, [i] formant)  × GainNode_2  ← N2 gain
 *       └──> BPF_3 (center ~3500 Hz, [u/s] region) × GainNode_3  ← N3 gain
 *                                                         │
 *   WhiteNoiseBurst (5ms, triggered on ΔN4 > threshold)  │
 *                                                         │
 *   All channels ──────────────────────────────────────> MixGain
 *                                                         │
 *                                              DynamicsCompressor
 *                                                         │
 *                                              MasterGain (0.35)
 *                                                         │
 *                                              AudioContext.destination
 *
 * ============================================================
 * Bottleneck → Acoustic mapping (revised):
 *
 *   N1  [0,1]  → GainNode_1.gain   (F1 region ~1000 Hz, open vowel [a])
 *   N2  [0,1]  → GainNode_2.gain   (F2 region ~2500 Hz, front vowel [i])
 *   N3  [0,1]  → GainNode_3.gain   (F3/noise region ~3500 Hz, approach/sibilant)
 *   N4  [0,1]  → ΔN4 > 0.15 in one frame → 5ms white-noise burst (plosive onset)
 *
 * Pitch (F0) is ALWAYS 150 Hz regardless of any neuron value.
 * The listener hears one voice whose timbre morphs — not a chord.
 * ============================================================
 */

export type Bottleneck = [number, number, number, number];

const SOURCE_FREQ   = 150;    // Hz — fixed glottal frequency (F0)
const MASTER_GAIN   = 0.35;
const FADE_TC       = 0.025;  // seconds — time-constant for smooth gain transitions
const BURST_DURATION = 0.005; // seconds — 5 ms consonant burst
const N4_DELTA_THRESHOLD = 0.15; // minimum ΔN4 per frame to trigger burst
const SILENCE_THRESHOLD  = 0.08; // all neurons below this → fade to silence

// Formant centre frequencies (Hz) — fixed, matching human vowel space
const F1_CENTER = 1000;   // N1: [a] — low F2, high F1
const F2_CENTER = 2500;   // N2: [i] — high F2
const F3_CENTER = 3500;   // N3: [s/u] — high F3 / sibilant region

const F_Q = 2.0; // BPF quality factor — narrow enough to shape timbre

export class AudioEngine {
  private ctx: AudioContext | null = null;

  // Source
  private glottis: OscillatorNode | null = null;
  private glottisGain: GainNode | null = null;

  // Filter bank (parallel BPFs)
  private bpf1: BiquadFilterNode | null = null;
  private bpf2: BiquadFilterNode | null = null;
  private bpf3: BiquadFilterNode | null = null;
  private fGain1: GainNode | null = null;
  private fGain2: GainNode | null = null;
  private fGain3: GainNode | null = null;

  // Mix / output
  private mixGain: GainNode | null = null;
  private compressor: DynamicsCompressorNode | null = null;
  private masterGain: GainNode | null = null;

  // N4 burst state
  private prevN4 = 0;

  private _running = false;
  private _muted   = false;

  get running() { return this._running; }
  get muted()   { return this._muted;   }

  // ─────────────────────────────────────────────
  // Initialisation
  // ─────────────────────────────────────────────

  async init(): Promise<void> {
    if (this.ctx) return;

    this.ctx = new AudioContext();

    // ── Output chain ──────────────────────────
    this.compressor = this.ctx.createDynamicsCompressor();
    this.compressor.threshold.value = -18;
    this.compressor.knee.value      = 10;
    this.compressor.ratio.value     = 4;
    this.compressor.attack.value    = 0.003;
    this.compressor.release.value   = 0.25;

    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = MASTER_GAIN;

    this.compressor.connect(this.masterGain);
    this.masterGain.connect(this.ctx.destination);

    // ── Mix bus ───────────────────────────────
    this.mixGain = this.ctx.createGain();
    this.mixGain.gain.value = 1.0;
    this.mixGain.connect(this.compressor);

    // ── Single glottal source (sawtooth @ F0) ─
    this.glottis = this.ctx.createOscillator();
    this.glottis.type = 'sawtooth';
    this.glottis.frequency.value = SOURCE_FREQ; // FIXED — never changes

    this.glottisGain = this.ctx.createGain();
    this.glottisGain.gain.value = 1.0;
    this.glottis.connect(this.glottisGain);

    // ── Filter bank (parallel) ────────────────
    // N1 → F1 (~1000 Hz, [a])
    this.bpf1 = this.ctx.createBiquadFilter();
    this.bpf1.type = 'bandpass';
    this.bpf1.frequency.value = F1_CENTER;
    this.bpf1.Q.value = F_Q;

    this.fGain1 = this.ctx.createGain();
    this.fGain1.gain.value = 0;

    this.glottisGain.connect(this.bpf1);
    this.bpf1.connect(this.fGain1);
    this.fGain1.connect(this.mixGain);

    // N2 → F2 (~2500 Hz, [i])
    this.bpf2 = this.ctx.createBiquadFilter();
    this.bpf2.type = 'bandpass';
    this.bpf2.frequency.value = F2_CENTER;
    this.bpf2.Q.value = F_Q;

    this.fGain2 = this.ctx.createGain();
    this.fGain2.gain.value = 0;

    this.glottisGain.connect(this.bpf2);
    this.bpf2.connect(this.fGain2);
    this.fGain2.connect(this.mixGain);

    // N3 → F3 (~3500 Hz, [s/u])
    this.bpf3 = this.ctx.createBiquadFilter();
    this.bpf3.type = 'bandpass';
    this.bpf3.frequency.value = F3_CENTER;
    this.bpf3.Q.value = F_Q;

    this.fGain3 = this.ctx.createGain();
    this.fGain3.gain.value = 0;

    this.glottisGain.connect(this.bpf3);
    this.bpf3.connect(this.fGain3);
    this.fGain3.connect(this.mixGain);

    // Start the glottis — it runs continuously
    this.glottis.start();

    this._running = true;
  }

  // ─────────────────────────────────────────────
  // Per-frame update  (call every ≤16 ms)
  // ─────────────────────────────────────────────

  update(bottleneck: Bottleneck): void {
    if (!this.ctx || !this._running || this._muted) return;

    const [n1, n2, n3, n4] = bottleneck;
    const now = this.ctx.currentTime;

    // ── Silence gate ──────────────────────────
    const allSilent = n1 < SILENCE_THRESHOLD &&
                      n2 < SILENCE_THRESHOLD &&
                      n3 < SILENCE_THRESHOLD &&
                      n4 < SILENCE_THRESHOLD;

    if (allSilent) {
      this.fGain1?.gain.setTargetAtTime(0, now, FADE_TC);
      this.fGain2?.gain.setTargetAtTime(0, now, FADE_TC);
      this.fGain3?.gain.setTargetAtTime(0, now, FADE_TC);
      this.prevN4 = n4;
      return;
    }

    // ── N1, N2, N3 → filter GAIN only ─────────
    // Frequency (F0 = 150 Hz) is NEVER touched.
    // Only the gain of each formant filter changes.
    this.fGain1?.gain.setTargetAtTime(n1, now, FADE_TC);
    this.fGain2?.gain.setTargetAtTime(n2, now, FADE_TC);
    this.fGain3?.gain.setTargetAtTime(n3, now, FADE_TC);

    // ── N4 → consonant burst on positive delta ─
    const deltaN4 = n4 - this.prevN4;
    if (deltaN4 > N4_DELTA_THRESHOLD) {
      this._triggerBurst(n4); // amplitude proportional to N4 value
    }
    this.prevN4 = n4;
  }

  // ─────────────────────────────────────────────
  // Consonant burst (5 ms white noise)
  // ─────────────────────────────────────────────

  private _triggerBurst(amplitude: number): void {
    if (!this.ctx || !this.mixGain) return;

    const sampleRate  = this.ctx.sampleRate;
    const burstFrames = Math.floor(sampleRate * BURST_DURATION);

    const buf = this.ctx.createBuffer(1, burstFrames, sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < burstFrames; i++) {
      data[i] = (Math.random() * 2 - 1);
    }

    // Envelope: linear fade-out over burst duration
    for (let i = 0; i < burstFrames; i++) {
      data[i] *= (1 - i / burstFrames);
    }

    const src  = this.ctx.createBufferSource();
    src.buffer = buf;

    const burstGain = this.ctx.createGain();
    burstGain.gain.value = Math.min(1.0, amplitude * 1.2);

    src.connect(burstGain);
    burstGain.connect(this.mixGain);
    src.start();
    src.stop(this.ctx.currentTime + BURST_DURATION + 0.001);
  }

  // ─────────────────────────────────────────────
  // Controls
  // ─────────────────────────────────────────────

  setMuted(muted: boolean): void {
    this._muted = muted;
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setTargetAtTime(
        muted ? 0 : MASTER_GAIN,
        this.ctx.currentTime,
        FADE_TC
      );
    }
  }

  async resume(): Promise<void> {
    if (this.ctx?.state === 'suspended') await this.ctx.resume();
  }

  async suspend(): Promise<void> {
    if (this.ctx?.state === 'running') await this.ctx.suspend();
  }

  destroy(): void {
    this.glottis?.stop();
    this.ctx?.close();
    this.ctx    = null;
    this._running = false;
  }
}

// Singleton
export const audioEngine = new AudioEngine();

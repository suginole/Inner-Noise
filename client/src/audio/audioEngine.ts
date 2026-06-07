/**
 * Audio Engine — Inner Noise
 * Synthesizes phoneme-like sounds from bottleneck neuron activations.
 * Zero external dependencies — pure Web Audio API.
 *
 * Neuron → Acoustic mapping:
 *   N1: Sine wave 100–300Hz (open vowel, food/positive)
 *   N2: High formant 800–3000Hz (front vowel, threat/negative)
 *   N3: White noise BPF 1–4kHz (fricative, approach)
 *   N4: Burst noise 2ms pulse (plosive, avoidance) — triggers at >0.7
 */

export type Bottleneck = [number, number, number, number];

const SILENCE_THRESHOLD = 0.1;
const MASTER_GAIN = 0.3;
const FADE_TIME = 0.05; // 50ms fade for smooth transitions

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private compressor: DynamicsCompressorNode | null = null;

  // N1: sine oscillator
  private osc1: OscillatorNode | null = null;
  private gain1: GainNode | null = null;

  // N2: high formant oscillator
  private osc2: OscillatorNode | null = null;
  private gain2: GainNode | null = null;

  // N3: white noise + bandpass filter
  private noiseBuffer: AudioBuffer | null = null;
  private noiseSource: AudioBufferSourceNode | null = null;
  private bpf3: BiquadFilterNode | null = null;
  private gain3: GainNode | null = null;

  // N4: burst noise
  private burstGain: GainNode | null = null;
  private lastN4: number = 0;

  private _running = false;
  private _muted = false;

  get running() { return this._running; }
  get muted() { return this._muted; }

  /** Initialize the audio context and graph */
  async init(): Promise<void> {
    if (this.ctx) return;

    this.ctx = new AudioContext();

    // Compressor → master gain → destination
    this.compressor = this.ctx.createDynamicsCompressor();
    this.compressor.threshold.value = -18;
    this.compressor.knee.value = 10;
    this.compressor.ratio.value = 4;
    this.compressor.attack.value = 0.003;
    this.compressor.release.value = 0.25;

    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = MASTER_GAIN;

    this.compressor.connect(this.masterGain);
    this.masterGain.connect(this.ctx.destination);

    // N1: Sine oscillator
    this.osc1 = this.ctx.createOscillator();
    this.osc1.type = 'sine';
    this.osc1.frequency.value = 150;
    this.gain1 = this.ctx.createGain();
    this.gain1.gain.value = 0;
    this.osc1.connect(this.gain1);
    this.gain1.connect(this.compressor);
    this.osc1.start();

    // N2: Sawtooth oscillator (richer harmonics for formant)
    this.osc2 = this.ctx.createOscillator();
    this.osc2.type = 'sawtooth';
    this.osc2.frequency.value = 1200;
    this.gain2 = this.ctx.createGain();
    this.gain2.gain.value = 0;
    this.osc2.connect(this.gain2);
    this.gain2.connect(this.compressor);
    this.osc2.start();

    // N3: White noise source (looping)
    const bufferSize = this.ctx.sampleRate * 2; // 2 seconds
    this.noiseBuffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = this.noiseBuffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1;
    }

    this.bpf3 = this.ctx.createBiquadFilter();
    this.bpf3.type = 'bandpass';
    this.bpf3.frequency.value = 2000;
    this.bpf3.Q.value = 1.5;

    this.gain3 = this.ctx.createGain();
    this.gain3.gain.value = 0;

    this._startNoiseLoop();
    this.bpf3.connect(this.gain3);
    this.gain3.connect(this.compressor);

    // N4: Burst gain (triggered on threshold)
    this.burstGain = this.ctx.createGain();
    this.burstGain.gain.value = 0;
    this.burstGain.connect(this.compressor);

    this._running = true;
  }

  private _startNoiseLoop() {
    if (!this.ctx || !this.noiseBuffer || !this.bpf3) return;
    this.noiseSource = this.ctx.createBufferSource();
    this.noiseSource.buffer = this.noiseBuffer;
    this.noiseSource.loop = true;
    this.noiseSource.connect(this.bpf3);
    this.noiseSource.start();
  }

  /** Update audio parameters from bottleneck values (call every frame ≤16ms) */
  update(bottleneck: Bottleneck): void {
    if (!this.ctx || !this._running || this._muted) return;

    const [n1, n2, n3, n4] = bottleneck;
    const now = this.ctx.currentTime;
    const fade = FADE_TIME;

    // Check silence
    const allSilent = n1 < SILENCE_THRESHOLD && n2 < SILENCE_THRESHOLD &&
                      n3 < SILENCE_THRESHOLD && n4 < SILENCE_THRESHOLD;

    if (allSilent) {
      this.gain1?.gain.setTargetAtTime(0, now, fade);
      this.gain2?.gain.setTargetAtTime(0, now, fade);
      this.gain3?.gain.setTargetAtTime(0, now, fade);
      return;
    }

    // N1: frequency 100–300Hz, gain proportional
    if (this.osc1 && this.gain1) {
      this.osc1.frequency.setTargetAtTime(100 + n1 * 200, now, fade);
      this.gain1.gain.setTargetAtTime(n1 * 0.8, now, fade);
    }

    // N2: frequency 800–3000Hz, gain proportional
    if (this.osc2 && this.gain2) {
      this.osc2.frequency.setTargetAtTime(800 + n2 * 2200, now, fade);
      this.gain2.gain.setTargetAtTime(n2 * 0.5, now, fade);
    }

    // N3: bandpass filter center 1–4kHz, gain proportional
    if (this.bpf3 && this.gain3) {
      this.bpf3.frequency.setTargetAtTime(1000 + n3 * 3000, now, fade);
      this.gain3.gain.setTargetAtTime(n3 * 0.6, now, fade);
    }

    // N4: burst trigger at > 0.7
    if (n4 > 0.7 && this.lastN4 <= 0.7 && this.ctx && this.burstGain) {
      this._triggerBurst();
    }
    this.lastN4 = n4;
  }

  private _triggerBurst(): void {
    if (!this.ctx || !this.burstGain || !this.noiseBuffer) return;

    // Create a short burst noise source
    const burstSource = this.ctx.createBufferSource();
    const burstBuf = this.ctx.createBuffer(1, Math.floor(this.ctx.sampleRate * 0.002), this.ctx.sampleRate);
    const d = burstBuf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    burstSource.buffer = burstBuf;

    const burstGain = this.ctx.createGain();
    burstGain.gain.value = 0.8;
    burstSource.connect(burstGain);
    burstGain.connect(this.compressor!);
    burstSource.start();
    burstSource.stop(this.ctx.currentTime + 0.002);
  }

  /** Toggle mute */
  setMuted(muted: boolean): void {
    this._muted = muted;
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setTargetAtTime(
        muted ? 0 : MASTER_GAIN,
        this.ctx.currentTime,
        FADE_TIME
      );
    }
  }

  /** Resume AudioContext (required after user gesture) */
  async resume(): Promise<void> {
    if (this.ctx?.state === 'suspended') {
      await this.ctx.resume();
    }
  }

  /** Suspend to save CPU */
  async suspend(): Promise<void> {
    if (this.ctx?.state === 'running') {
      await this.ctx.suspend();
    }
  }

  /** Clean up all nodes */
  destroy(): void {
    this.osc1?.stop();
    this.osc2?.stop();
    this.noiseSource?.stop();
    this.ctx?.close();
    this.ctx = null;
    this._running = false;
  }
}

// Singleton instance
export const audioEngine = new AudioEngine();

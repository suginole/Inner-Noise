/**
 * Neural Network Module — Inner Noise
 * Bottleneck feedforward network:
 *   Input(8) → Hidden(16, ReLU) → Bottleneck(4, Sigmoid) → Output(4, Softmax)
 *
 * The 4 bottleneck values [0,1] are the "Inner Noise" — sonified in real-time.
 */

export type Weights = number[][];

export interface NetworkWeights {
  w1: Weights;   // 8 → 16
  b1: number[];  // bias for hidden layer
  w2: Weights;   // 16 → 4 (bottleneck)
  b2: number[];  // bias for bottleneck
  w3: Weights;   // 4 → 4 (output)
  b3: number[];  // bias for output
}

/** Flatten all weights into a 1D array for GA operations */
export function flattenWeights(nw: NetworkWeights): number[] {
  const flat: number[] = [];
  for (const row of nw.w1) flat.push(...row);
  flat.push(...nw.b1);
  for (const row of nw.w2) flat.push(...row);
  flat.push(...nw.b2);
  for (const row of nw.w3) flat.push(...row);
  flat.push(...nw.b3);
  return flat;
}

/** Reconstruct NetworkWeights from a flat array */
export function unflattenWeights(flat: number[]): NetworkWeights {
  let idx = 0;
  const take = (n: number) => flat.slice(idx, (idx += n));

  const w1: Weights = Array.from({ length: 16 }, () => take(8));
  const b1 = take(16);
  const w2: Weights = Array.from({ length: 4 }, () => take(16));
  const b2 = take(4);
  const w3: Weights = Array.from({ length: 4 }, () => take(4));
  const b3 = take(4);

  return { w1, b1, w2, b2, w3, b3 };
}

/** Total number of weights */
export const WEIGHT_COUNT = 16 * 8 + 16 + 4 * 16 + 4 + 4 * 4 + 4; // = 128+16+64+4+16+4 = 232

/** Initialize random weights in [-1, 1] */
export function randomWeights(): NetworkWeights {
  const rand = () => (Math.random() * 2 - 1) * 0.5;
  const mat = (rows: number, cols: number): Weights =>
    Array.from({ length: rows }, () => Array.from({ length: cols }, rand));
  const vec = (n: number) => Array.from({ length: n }, rand);

  return {
    w1: mat(16, 8),
    b1: vec(16),
    w2: mat(4, 16),
    b2: vec(4),
    w3: mat(4, 4),
    b3: vec(4),
  };
}

function relu(x: number): number {
  return Math.max(0, x);
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function matVecMul(mat: Weights, vec: number[]): number[] {
  return mat.map((row) => row.reduce((sum, w, i) => sum + w * vec[i], 0));
}

function addBias(vec: number[], bias: number[]): number[] {
  return vec.map((v, i) => v + bias[i]);
}

export interface ForwardResult {
  bottleneck: [number, number, number, number];
  action: number; // 0=up, 1=right, 2=down, 3=left
  actionProbs: [number, number, number, number];
}

/**
 * Forward pass through the network.
 * @param input 8-element sensor vector
 * @param nw network weights
 * @param forcedBottleneck optional override for bottleneck values (Lab mode)
 */
export function forward(
  input: number[],
  nw: NetworkWeights,
  forcedBottleneck?: [number, number, number, number] | null
): ForwardResult {
  // Layer 1: input → hidden (ReLU)
  const h1 = addBias(matVecMul(nw.w1, input), nw.b1).map(relu);

  // Layer 2: hidden → bottleneck (Sigmoid)
  let bn: number[];
  if (forcedBottleneck) {
    bn = [...forcedBottleneck];
  } else {
    bn = addBias(matVecMul(nw.w2, h1), nw.b2).map(sigmoid);
  }

  // Layer 3: bottleneck → output (Softmax)
  const logits = addBias(matVecMul(nw.w3, bn), nw.b3);
  const probs = softmax(logits);

  // Argmax action selection
  const action = probs.indexOf(Math.max(...probs));

  return {
    bottleneck: bn as [number, number, number, number],
    action,
    actionProbs: probs as [number, number, number, number],
  };
}

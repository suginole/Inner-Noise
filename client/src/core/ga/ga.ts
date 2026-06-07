/**
 * Genetic Algorithm Module — Inner Noise
 * Tournament selection (k=3), uniform crossover, Gaussian mutation.
 * Elite preservation: top 2 individuals pass unchanged.
 */

import { NetworkWeights, flattenWeights, unflattenWeights, WEIGHT_COUNT } from '../nn/network';
import { runEpisode } from '../agent/agent';
import { Cell } from '../maze/maze';

export const POPULATION_SIZE = 50;
export const TOURNAMENT_K = 3;
export const MUTATION_RATE = 0.02;
export const MUTATION_SIGMA = 0.1;
export const ELITE_COUNT = 2;

export interface Individual {
  weights: NetworkWeights;
  fitness: number;
}

/** Initialize a random population */
export function initPopulation(): Individual[] {
  return Array.from({ length: POPULATION_SIZE }, () => {
    const flat = Array.from({ length: WEIGHT_COUNT }, () => (Math.random() * 2 - 1) * 0.5);
    return { weights: unflattenWeights(flat), fitness: 0 };
  });
}

/** Evaluate all individuals against the given maze */
export function evaluatePopulation(population: Individual[], grid: Cell[][]): Individual[] {
  return population.map((ind) => {
    const { fitness } = runEpisode(ind.weights, grid);
    return { ...ind, fitness };
  });
}

/** Tournament selection: pick k random, return best */
function tournamentSelect(population: Individual[]): Individual {
  let best = population[Math.floor(Math.random() * population.length)];
  for (let i = 1; i < TOURNAMENT_K; i++) {
    const candidate = population[Math.floor(Math.random() * population.length)];
    if (candidate.fitness > best.fitness) best = candidate;
  }
  return best;
}

/** Uniform crossover of two flat weight arrays */
function uniformCrossover(a: number[], b: number[]): number[] {
  return a.map((va, i) => (Math.random() < 0.5 ? va : b[i]));
}

/** Gaussian mutation */
function mutate(flat: number[]): number[] {
  return flat.map((w) => {
    if (Math.random() < MUTATION_RATE) {
      return w + (gaussianRandom() * MUTATION_SIGMA);
    }
    return w;
  });
}

/** Box-Muller transform for Gaussian random */
function gaussianRandom(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** Produce the next generation */
export function nextGeneration(population: Individual[]): Individual[] {
  // Sort descending by fitness
  const sorted = [...population].sort((a, b) => b.fitness - a.fitness);

  const newPop: Individual[] = [];

  // Elite preservation
  for (let i = 0; i < ELITE_COUNT; i++) {
    newPop.push({ weights: sorted[i].weights, fitness: 0 });
  }

  // Fill rest with crossover + mutation
  while (newPop.length < POPULATION_SIZE) {
    const parentA = tournamentSelect(sorted);
    const parentB = tournamentSelect(sorted);
    const flatA = flattenWeights(parentA.weights);
    const flatB = flattenWeights(parentB.weights);
    const child = mutate(uniformCrossover(flatA, flatB));
    newPop.push({ weights: unflattenWeights(child), fitness: 0 });
  }

  return newPop;
}

/** Get population statistics */
export function getStats(population: Individual[]): { avgFitness: number; maxFitness: number; bestWeights: NetworkWeights } {
  const fitnesses = population.map((i) => i.fitness);
  const maxFitness = Math.max(...fitnesses);
  const avgFitness = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
  const bestWeights = population.reduce((a, b) => (a.fitness > b.fitness ? a : b)).weights;
  return { avgFitness, maxFitness, bestWeights };
}

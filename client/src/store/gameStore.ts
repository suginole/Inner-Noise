/**
 * Game State Store — Inner Noise
 * Zustand store managing all game state across modes.
 * Design: Void Organism — dark, data-driven, organism-like state transitions.
 */

import { create } from 'zustand';
import { Cell } from '../core/maze/maze';
import { NetworkWeights } from '../core/nn/network';
import { AgentState, createAgent, stepAgent, newMazeWithFood } from '../core/agent/agent';
import {
  Individual,
  initPopulation,
  evaluatePopulation,
  nextGeneration,
  getStats,
} from '../core/ga/ga';

export type GameMode = 'observe' | 'accelerate' | 'lab';

export interface GenerationLog {
  generation: number;
  avgFitness: number;
  maxFitness: number;
  bottleneckLog: number[][];
}

export interface GameState {
  mode: GameMode;
  generation: number;
  population: Individual[];
  grid: Cell[][];
  foodPositions: { x: number; y: number }[];
  history: GenerationLog[];

  activeAgent: AgentState | null;
  activeFoodPositions: { x: number; y: number }[];

  forcedBottleneck: [number, number, number, number] | null;
  labBottleneckLog: number[][];

  isRunning: boolean;
  isAudioEnabled: boolean;
  accelerateTarget: number;
  accelerateProgress: number;

  initGame: () => void;
  setMode: (mode: GameMode) => void;
  tickObserve: () => void;
  runGeneration: () => void;
  runAccelerate: (generations: number) => Promise<void>;
  setForcedBottleneck: (bn: [number, number, number, number] | null) => void;
  setAudioEnabled: (enabled: boolean) => void;
  resetActiveAgent: () => void;
  loadFromStorage: () => void;
  saveToStorage: () => void;
}

const STORAGE_KEY_STATE = 'inner-noise:state';
const STORAGE_KEY_HISTORY = 'inner-noise:history';
const MAX_HISTORY = 1000;

function makeActiveAgent(weights: NetworkWeights, foodPositions: { x: number; y: number }[]): { agent: AgentState; food: { x: number; y: number }[] } {
  const agent = createAgent(weights);
  agent.position = { x: 1, y: 1 };
  return { agent, food: [...foodPositions] };
}

export const useGameStore = create<GameState>((set, get) => ({
  mode: 'observe',
  generation: 0,
  population: [],
  grid: [],
  foodPositions: [],
  history: [],
  activeAgent: null,
  activeFoodPositions: [],
  forcedBottleneck: null,
  labBottleneckLog: [],
  isRunning: false,
  isAudioEnabled: true,
  accelerateTarget: 10,
  accelerateProgress: 0,

  initGame: () => {
    const { grid, foodPositions } = newMazeWithFood();
    const population = initPopulation();
    const evaluated = evaluatePopulation(population, grid);
    const { maxFitness, avgFitness, bestWeights } = getStats(evaluated);

    const { agent, food } = makeActiveAgent(bestWeights, foodPositions);

    set({
      generation: 0,
      population: evaluated,
      grid,
      foodPositions,
      history: [{ generation: 0, avgFitness, maxFitness, bottleneckLog: [] }],
      activeAgent: agent,
      activeFoodPositions: food,
      labBottleneckLog: [],
      isRunning: true,
    });
  },

  setMode: (mode) => {
    const state = get();
    const { bestWeights } = getStats(state.population.length > 0 ? state.population : initPopulation());
    const { agent, food } = makeActiveAgent(bestWeights, state.foodPositions);
    set({
      mode,
      activeAgent: agent,
      activeFoodPositions: food,
      labBottleneckLog: [],
      forcedBottleneck: null,
    });
  },

  tickObserve: () => {
    const state = get();
    if (!state.activeAgent || !state.isRunning) return;

    const agent = state.activeAgent;
    const result = stepAgent(
      agent,
      state.grid,
      state.activeFoodPositions,
      state.forcedBottleneck
    );

    const newAgent: AgentState = {
      ...agent,
      position: result.newPos,
      bottleneck: result.forward.bottleneck,
      fitness: agent.fitness + result.fitnessGain,
      foodEaten: agent.foodEaten + (result.ateFood ? 1 : 0),
      frame: agent.frame + 1,
    };

    let newFoodPositions = state.activeFoodPositions;
    if (result.ateFood) {
      newFoodPositions = state.activeFoodPositions.filter(
        (fp) => !(fp.x === result.newPos.x && fp.y === result.newPos.y)
      );
    }

    // Update lab log (keep last 300 frames)
    const newLabLog =
      state.mode === 'lab'
        ? [...state.labBottleneckLog.slice(-299), [...result.forward.bottleneck]]
        : state.labBottleneckLog;

    // If agent exhausted or reached end, reset with same best weights
    if (newAgent.frame >= 500 || result.reachedEnd) {
      const { bestWeights } = getStats(state.population);
      const { agent: freshAgent, food: freshFood } = makeActiveAgent(bestWeights, state.foodPositions);
      set({
        activeAgent: freshAgent,
        activeFoodPositions: freshFood,
        labBottleneckLog: newLabLog,
      });
    } else {
      set({
        activeAgent: newAgent,
        activeFoodPositions: newFoodPositions,
        labBottleneckLog: newLabLog,
      });
    }
  },

  runGeneration: () => {
    const state = get();
    if (state.population.length === 0) return;

    const newPop = nextGeneration(state.population);
    const evaluated = evaluatePopulation(newPop, state.grid);
    const { maxFitness, avgFitness, bestWeights } = getStats(evaluated);

    const newGen = state.generation + 1;
    const log: GenerationLog = {
      generation: newGen,
      avgFitness,
      maxFitness,
      bottleneckLog: [],
    };

    const newHistory = [...state.history, log].slice(-MAX_HISTORY);
    const { agent, food } = makeActiveAgent(bestWeights, state.foodPositions);

    set({
      generation: newGen,
      population: evaluated,
      history: newHistory,
      activeAgent: agent,
      activeFoodPositions: food,
    });

    // Persist every 5 generations to avoid excessive writes
    if (newGen % 5 === 0) {
      get().saveToStorage();
    }
  },

  runAccelerate: async (generations: number) => {
    set({ accelerateTarget: generations, accelerateProgress: 0, isRunning: false });

    const state = get();
    let pop = state.population;
    let gen = state.generation;
    const grid = state.grid;
    const history = [...state.history];

    for (let i = 0; i < generations; i++) {
      pop = nextGeneration(pop);
      pop = evaluatePopulation(pop, grid);
      const { maxFitness, avgFitness } = getStats(pop);
      gen++;
      history.push({ generation: gen, avgFitness, maxFitness, bottleneckLog: [] });

      // Yield to UI every 10 generations
      if (i % 10 === 0) {
        set({ accelerateProgress: i + 1 });
        await new Promise((r) => setTimeout(r, 0));
      }
    }

    const { bestWeights } = getStats(pop);
    const { agent, food } = makeActiveAgent(bestWeights, get().foodPositions);

    set({
      population: pop,
      generation: gen,
      history: history.slice(-MAX_HISTORY),
      activeAgent: agent,
      activeFoodPositions: food,
      accelerateProgress: generations,
      isRunning: true,
    });

    get().saveToStorage();
  },

  setForcedBottleneck: (bn) => set({ forcedBottleneck: bn }),

  setAudioEnabled: (enabled) => set({ isAudioEnabled: enabled }),

  resetActiveAgent: () => {
    const state = get();
    if (state.population.length === 0) return;
    const { bestWeights } = getStats(state.population);
    const { agent, food } = makeActiveAgent(bestWeights, state.foodPositions);
    set({ activeAgent: agent, activeFoodPositions: food });
  },

  loadFromStorage: () => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY_STATE);
      const histRaw = localStorage.getItem(STORAGE_KEY_HISTORY);
      if (raw) {
        const saved = JSON.parse(raw) as Partial<GameState>;
        const history: GenerationLog[] = histRaw ? JSON.parse(histRaw) : [];

        // Reconstruct active agent from saved population
        let activeAgent = null;
        let activeFoodPositions: { x: number; y: number }[] = [];

        if (saved.population && saved.population.length > 0 && saved.foodPositions) {
          const { bestWeights } = getStats(saved.population);
          const result = makeActiveAgent(bestWeights, saved.foodPositions);
          activeAgent = result.agent;
          activeFoodPositions = result.food;
        }

        set({
          generation: saved.generation ?? 0,
          population: saved.population ?? [],
          grid: saved.grid ?? [],
          foodPositions: saved.foodPositions ?? [],
          mode: saved.mode ?? 'observe',
          history,
          activeAgent,
          activeFoodPositions,
          isRunning: true,
        });
        return;
      }
    } catch {
      // Fall through to init
    }
    get().initGame();
  },

  saveToStorage: () => {
    const state = get();
    try {
      const toSave = {
        generation: state.generation,
        population: state.population,
        grid: state.grid,
        foodPositions: state.foodPositions,
        mode: state.mode,
      };
      localStorage.setItem(STORAGE_KEY_STATE, JSON.stringify(toSave));
      localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(state.history));
    } catch {
      // Storage quota exceeded — silently ignore
    }
  },
}));

export const selectBottleneck = (s: GameState): [number, number, number, number] =>
  s.activeAgent?.bottleneck ?? [0, 0, 0, 0];

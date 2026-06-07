/**
 * Agent Module — Inner Noise
 * Handles agent sensors, movement, fitness calculation, and episode simulation.
 *
 * Sensor vector (8 values):
 *   [wallUp, wallRight, wallDown, wallLeft,    ← distance to wall (0=adjacent, 1=far)
 *    foodDirX, foodDirY,                        ← normalized direction to nearest food
 *    distToEnd_x, distToEnd_y]                  ← normalized direction to maze end
 */

import { nanoid } from 'nanoid';
import { Cell, MAZE_SIZE, generateMaze } from '../maze/maze';
import { NetworkWeights, ForwardResult, forward, randomWeights } from '../nn/network';

export const MAX_FRAMES = 500;
export const WALL_PENALTY = -0.1;
export const FOOD_REWARD = 1.0;
export const SURVIVAL_BONUS = 0.001;
export const END_REWARD = 5.0;

// Action index → [dx, dy]
const ACTION_DELTAS: [number, number][] = [
  [0, -1], // up
  [1, 0],  // right
  [0, 1],  // down
  [-1, 0], // left
];

export interface AgentState {
  id: string;
  weights: NetworkWeights;
  fitness: number;
  position: { x: number; y: number };
  bottleneck: [number, number, number, number];
  foodEaten: number;
  alive: boolean;
  frame: number;
}

export function createAgent(weights?: NetworkWeights): AgentState {
  return {
    id: nanoid(8),
    weights: weights ?? randomWeights(),
    fitness: 0,
    position: { x: 1, y: 1 },
    bottleneck: [0, 0, 0, 0],
    foodEaten: 0,
    alive: true,
    frame: 0,
  };
}

/** Compute the 8-element sensor vector for an agent */
export function computeSensors(
  pos: { x: number; y: number },
  grid: Cell[][],
  foodPositions: { x: number; y: number }[]
): number[] {
  const { x, y } = pos;

  // Wall distances (0 = wall adjacent, normalized to [0,1])
  const wallUp    = y > 0 && grid[y - 1][x].type !== 'wall' ? 1 : 0;
  const wallRight = x < MAZE_SIZE - 1 && grid[y][x + 1].type !== 'wall' ? 1 : 0;
  const wallDown  = y < MAZE_SIZE - 1 && grid[y + 1][x].type !== 'wall' ? 1 : 0;
  const wallLeft  = x > 0 && grid[y][x - 1].type !== 'wall' ? 1 : 0;

  // Direction to nearest food
  let foodDirX = 0;
  let foodDirY = 0;
  if (foodPositions.length > 0) {
    let minDist = Infinity;
    let nearest = foodPositions[0];
    for (const fp of foodPositions) {
      const d = Math.abs(fp.x - x) + Math.abs(fp.y - y);
      if (d < minDist) {
        minDist = d;
        nearest = fp;
      }
    }
    const dx = nearest.x - x;
    const dy = nearest.y - y;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    foodDirX = dx / len * 0.5 + 0.5; // normalize to [0,1]
    foodDirY = dy / len * 0.5 + 0.5;
  }

  // Direction to maze end (bottom-right: MAZE_SIZE-2, MAZE_SIZE-2)
  const endX = MAZE_SIZE - 2;
  const endY = MAZE_SIZE - 2;
  const dex = endX - x;
  const dey = endY - y;
  const endLen = Math.sqrt(dex * dex + dey * dey) || 1;
  const endDirX = dex / endLen * 0.5 + 0.5;
  const endDirY = dey / endLen * 0.5 + 0.5;

  return [wallUp, wallRight, wallDown, wallLeft, foodDirX, foodDirY, endDirX, endDirY];
}

export interface StepResult {
  forward: ForwardResult;
  newPos: { x: number; y: number };
  fitnessGain: number;
  ateFood: boolean;
  hitWall: boolean;
  reachedEnd: boolean;
}

/** Execute one step for an agent */
export function stepAgent(
  agent: AgentState,
  grid: Cell[][],
  foodPositions: { x: number; y: number }[],
  forcedBottleneck?: [number, number, number, number] | null
): StepResult {
  const sensors = computeSensors(agent.position, grid, foodPositions);
  const fwd = forward(sensors, agent.weights, forcedBottleneck);

  const [dx, dy] = ACTION_DELTAS[fwd.action];
  const nx = agent.position.x + dx;
  const ny = agent.position.y + dy;

  let fitnessGain = SURVIVAL_BONUS;
  let ateFood = false;
  let hitWall = false;
  let reachedEnd = false;
  let newPos = { ...agent.position };

  // Check bounds and wall collision
  if (nx < 0 || nx >= MAZE_SIZE || ny < 0 || ny >= MAZE_SIZE || grid[ny][nx].type === 'wall') {
    fitnessGain += WALL_PENALTY;
    hitWall = true;
  } else {
    newPos = { x: nx, y: ny };

    // Check food
    const foodIdx = foodPositions.findIndex((fp) => fp.x === nx && fp.y === ny);
    if (foodIdx !== -1) {
      fitnessGain += FOOD_REWARD;
      ateFood = true;
    }

    // Check end
    if (nx === MAZE_SIZE - 2 && ny === MAZE_SIZE - 2) {
      fitnessGain += END_REWARD;
      reachedEnd = true;
    }
  }

  return { forward: fwd, newPos, fitnessGain, ateFood, hitWall, reachedEnd };
}

/**
 * Run a full episode for an agent (used in GA evaluation).
 * Returns the total fitness and bottleneck log.
 */
export function runEpisode(
  weights: NetworkWeights,
  grid: Cell[][]
): { fitness: number; bottleneckLog: number[][] } {
  const agent = createAgent(weights);
  agent.position = { x: 1, y: 1 };

  // Collect food positions
  let foodPositions: { x: number; y: number }[] = [];
  for (let y = 0; y < MAZE_SIZE; y++) {
    for (let x = 0; x < MAZE_SIZE; x++) {
      if (grid[y][x].hasFood) foodPositions.push({ x, y });
    }
  }

  const bottleneckLog: number[][] = [];
  let fitness = 0;

  for (let frame = 0; frame < MAX_FRAMES; frame++) {
    const result = stepAgent(agent, grid, foodPositions);
    fitness += result.fitnessGain;
    agent.position = result.newPos;
    agent.bottleneck = result.forward.bottleneck;
    bottleneckLog.push([...result.forward.bottleneck]);

    if (result.ateFood) {
      foodPositions = foodPositions.filter(
        (fp) => !(fp.x === result.newPos.x && fp.y === result.newPos.y)
      );
    }

    if (result.reachedEnd) break;
  }

  return { fitness, bottleneckLog };
}

/** Generate a fresh maze and return both grid and food positions */
export function newMazeWithFood(): { grid: Cell[][]; foodPositions: { x: number; y: number }[] } {
  const grid = generateMaze();
  const foodPositions: { x: number; y: number }[] = [];
  for (let y = 0; y < MAZE_SIZE; y++) {
    for (let x = 0; x < MAZE_SIZE; x++) {
      if (grid[y][x].hasFood) foodPositions.push({ x, y });
    }
  }
  return { grid, foodPositions };
}

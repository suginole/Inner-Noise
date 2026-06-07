/**
 * Maze Module — Inner Noise
 * Generates a 16×16 grid maze using Recursive Backtracking.
 * Guarantees a solvable maze with exactly one solution path.
 */

export type CellType = 'wall' | 'path' | 'start' | 'end' | 'food';

export interface Cell {
  x: number;
  y: number;
  type: CellType;
  hasFood: boolean;
  visited: boolean;
}

export const MAZE_SIZE = 16;
export const FOOD_COUNT = 5;

// Direction vectors: [dx, dy]
const DIRECTIONS = [
  [0, -2], // up
  [2, 0],  // right
  [0, 2],  // down
  [-2, 0], // left
];

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function generateMaze(): Cell[][] {
  // Initialize all cells as walls
  const grid: Cell[][] = Array.from({ length: MAZE_SIZE }, (_, y) =>
    Array.from({ length: MAZE_SIZE }, (_, x) => ({
      x,
      y,
      type: 'wall' as CellType,
      hasFood: false,
      visited: false,
    }))
  );

  // Recursive backtracking — works on odd-indexed cells
  function carve(x: number, y: number) {
    grid[y][x].type = 'path';
    grid[y][x].visited = true;

    for (const [dx, dy] of shuffle(DIRECTIONS)) {
      const nx = x + dx;
      const ny = y + dy;
      const mx = x + dx / 2;
      const my = y + dy / 2;

      if (nx >= 0 && nx < MAZE_SIZE && ny >= 0 && ny < MAZE_SIZE && !grid[ny][nx].visited) {
        grid[my][mx].type = 'path';
        carve(nx, ny);
      }
    }
  }

  carve(1, 1);

  // Set start and end
  grid[1][1].type = 'start';
  grid[MAZE_SIZE - 2][MAZE_SIZE - 2].type = 'end';

  // Place food randomly on path cells (not start/end)
  const pathCells: { x: number; y: number }[] = [];
  for (let y = 0; y < MAZE_SIZE; y++) {
    for (let x = 0; x < MAZE_SIZE; x++) {
      if (grid[y][x].type === 'path') {
        pathCells.push({ x, y });
      }
    }
  }

  const shuffledPaths = shuffle(pathCells);
  for (let i = 0; i < Math.min(FOOD_COUNT, shuffledPaths.length); i++) {
    const { x, y } = shuffledPaths[i];
    grid[y][x].hasFood = true;
  }

  return grid;
}

/** Returns all walkable (non-wall) cell positions */
export function getWalkableCells(grid: Cell[][]): { x: number; y: number }[] {
  const cells: { x: number; y: number }[] = [];
  for (let y = 0; y < MAZE_SIZE; y++) {
    for (let x = 0; x < MAZE_SIZE; x++) {
      if (grid[y][x].type !== 'wall') {
        cells.push({ x, y });
      }
    }
  }
  return cells;
}

/** Returns the start position */
export function getStartPosition(_grid: Cell[][]): { x: number; y: number } {
  return { x: 1, y: 1 };
}

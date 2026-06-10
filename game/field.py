"""
field.py — キノコフィールド管理（Sage-Brute版）

地形生成: numpy完全ベクトル化・高速・Pythonループなし
キノコ: 12種（バイオーム×グレード×バリアント）+ 腐敗フラグ
"""
import numpy as np
import pygame
import copy
from config import *


class Mushroom:
    def __init__(self, pos, biome, grade, variant, is_rotten):
        self.pos       = pygame.Vector2(pos)
        self.biome     = biome
        self.grade     = grade
        self.variant   = variant
        self.is_rotten = is_rotten


class Field:
    _terrain_cache = {}

    def __init__(self, terrain_seed=TERRAIN_SEED, food_episode=0):
        self._seed  = terrain_seed
        self._hmap  = self._build_terrain()
        self._surf  = None
        self.mushrooms = []
        self._place_mushrooms(food_episode)
        self.start_pos = self._find_start()
        self.goal_pos  = self._find_goal()

    def _build_terrain(self):
        if self._seed in Field._terrain_cache:
            return Field._terrain_cache[self._seed]
        gs  = FIELD_SIZE // TILE
        rng = np.random.default_rng(self._seed)
        xs  = np.linspace(0, 1, gs, dtype=np.float32)
        ys  = np.linspace(0, 1, gs, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        hmap = np.zeros((gs, gs), dtype=np.float32)
        amp, freq = 1.0, TERRAIN_SCALE * gs
        # persistence=0.65（旧値0.5）で高周波成分を強め、バイオームが密に入り組む
        persistence = 0.65
        for octave in range(TERRAIN_OCTAVES):
            ox = rng.integers(0, 1000)
            oy = rng.integers(0, 1000)
            # 各オクターブで異なるパーミュテーションを使用（シード依存）
            hmap += amp * self._perlin(xx * freq + ox, yy * freq + oy,
                                       perm_seed=self._seed + octave * 1000)
            amp  *= persistence
            freq *= 2.0
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        Field._terrain_cache[self._seed] = hmap
        return hmap

    @staticmethod
    def _perlin(x, y, perm_seed: int = 42):
        xi = x.astype(int) & 255
        yi = y.astype(int) & 255
        xf = x - x.astype(int)
        yf = y - y.astype(int)
        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)
        def lerp(a, b, t):
            return a + t * (b - a)
        rng2 = np.random.default_rng(perm_seed)
        perm  = rng2.permutation(256).astype(np.uint8)
        perm2 = np.concatenate([perm, perm])
        def grad(h, x, y):
            h = h & 3
            u = np.where(h < 2, x, y)
            v = np.where(h < 2, y, x)
            return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)
        aa = perm2[perm2[xi]   + yi]
        ab = perm2[perm2[xi]   + yi + 1]
        ba = perm2[perm2[xi+1] + yi]
        bb = perm2[perm2[xi+1] + yi + 1]
        u, v = fade(xf), fade(yf)
        x1 = lerp(grad(aa, xf,   yf),   grad(ba, xf-1, yf),   u)
        x2 = lerp(grad(ab, xf,   yf-1), grad(bb, xf-1, yf-1), u)
        return lerp(x1, x2, v)

    def _height_at(self, wx, wy):
        gs = FIELD_SIZE // TILE
        gx = int(wx / TILE) % gs
        gy = int(wy / TILE) % gs
        return float(self._hmap[gy, gx])

    def biome_at(self, wx, wy):
        h = self._height_at(wx, wy)
        if h < BIOME_THRESHOLDS[0]: return 'W'
        if h < BIOME_THRESHOLDS[1]: return 'G'
        return 'M'

    def _place_mushrooms(self, episode):
        rng = np.random.default_rng(FOOD_SEED + episode)
        self.mushrooms = []
        placed = 0
        attempts = 0
        while placed < FOOD_COUNT and attempts < FOOD_COUNT * 20:
            attempts += 1
            wx = rng.uniform(50, FIELD_SIZE - 50)
            wy = rng.uniform(50, FIELD_SIZE - 50)
            if rng.random() > MUSHROOM_DENSITY * 10:
                continue
            biome   = self.biome_at(wx, wy)
            grade   = rng.choice(['normal', 'premium'], p=[0.7, 0.3])
            variant = int(rng.integers(1, 3))
            is_rot  = bool(rng.random() < ROT_PROBABILITY)
            self.mushrooms.append(
                Mushroom((wx, wy), biome, grade, variant, is_rot))
            placed += 1

    def reset_foods(self, food_episode=0):
        self._place_mushrooms(food_episode)

    def _find_start(self):
        return (FIELD_SIZE * 0.1, FIELD_SIZE * 0.5)

    def _find_goal(self):
        return (FIELD_SIZE * 0.9, FIELD_SIZE * 0.5)

    def get_surface(self):
        if self._surf is not None:
            return self._surf
        gs = FIELD_SIZE // TILE
        arr = np.zeros((gs, gs, 3), dtype=np.uint8)
        lo, hi = BIOME_THRESHOLDS
        w_mask = self._hmap < lo
        g_mask = (self._hmap >= lo) & (self._hmap < hi)
        m_mask = self._hmap >= hi
        arr[w_mask] = BIOME_COLORS['W']
        arr[g_mask] = BIOME_COLORS['G']
        arr[m_mask] = BIOME_COLORS['M']
        # TILE倍に拡大
        arr_big = np.repeat(np.repeat(arr, TILE, axis=0), TILE, axis=1)
        self._surf = pygame.surfarray.make_surface(
            arr_big.transpose(1, 0, 2))
        return self._surf

    def clone_foods(self):
        """地形共有・キノコリストのみコピー"""
        clone = object.__new__(Field)
        clone.__dict__.update(self.__dict__)
        clone.mushrooms = [copy.copy(m) for m in self.mushrooms]
        return clone

    # ---- 旧インターフェース互換 ----
    def height_at(self, wx, wy):
        return self._height_at(wx, wy)

    def gradient_at(self, wx, wy):
        d = float(TILE)
        return ((self._height_at(wx + d, wy) - self._height_at(wx - d, wy)) / (2 * d),
                (self._height_at(wx, wy + d) - self._height_at(wx, wy - d)) / (2 * d))

    def is_mountain(self, wx, wy):
        return self._height_at(wx, wy) >= BIOME_THRESHOLDS[1]

    def is_valley(self, wx, wy):
        return self._height_at(wx, wy) <= BIOME_THRESHOLDS[0]

    @property
    def hmap(self):
        return self._hmap

    @property
    def terrain_seed(self):
        return self._seed

    @property
    def pass_pos(self):
        return (FIELD_SIZE // 2, FIELD_SIZE // 2)

    # ---- car.py 互換メソッド ----
    @property
    def foods(self):
        """car.py互換: mushrooms を (pos, is_premium) タプルリストとして返す。"""
        return [(m.pos, m.grade == 'premium') for m in self.mushrooms]

    def collect_food(self, wx, wy):
        """car.py互換: 座標付近のキノコを回収して (collected, is_premium) を返す。"""
        pos = pygame.Vector2(wx, wy)
        for i, m in enumerate(self.mushrooms):
            if pos.distance_to(m.pos) < FOOD_RADIUS:
                m = self.mushrooms.pop(i)
                return True, (m.grade == 'premium')
        return False, False

    def respawn_food(self):
        """car.py互換: キノコを1個補充する。"""
        rng = np.random.default_rng()
        wx = rng.uniform(50, FIELD_SIZE - 50)
        wy = rng.uniform(50, FIELD_SIZE - 50)
        biome = self.biome_at(wx, wy)
        grade = 'premium' if rng.random() < 0.3 else 'normal'
        variant = int(rng.integers(1, 3))
        is_rot = bool(rng.random() < ROT_PROBABILITY)
        self.mushrooms.append(Mushroom((wx, wy), biome, grade, variant, is_rot))

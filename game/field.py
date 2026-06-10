"""
field.py — キノコフィールド管理（Sage-Brute版）

キノコ設計:
  12種（各バイオーム4種 × 3バイオーム）+ 腐敗フラグ
  SAGEの弁別視野: 12種を識別可能・腐敗は識別不可
  BRUTEの弁別視野: 腐敗のみ識別可能・12種は識別不可

中毒・回復ルール:
  中毒: 同一種を連続TOXIC_COUNT(3)回摂取で発症 (-30)
  回復: 異なるバイオームの普通種2種を直近HISTORY_LEN(5)回内に揃えると回復
  ゴール: ENERGY_GOAL(+15) + 摂取履歴リセット
"""
import math
import random
import numpy as np
import pygame
from config import *


# ---- パーリンノイズ（numpy完全ベクトル化版）----
class PerlinNoise:
    """numpyベクトル化されたパーリンノイズ。
    グリッド全体を一括計算するのでピュアPythonループ比導で約10倍高速。
    """
    def __init__(self, seed: int = 0):
        rng = random.Random(seed)
        p = list(range(256))
        rng.shuffle(p)
        self.perm = np.array(p * 2, dtype=np.int32)

    def _grad2d(self, h: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h = h & 3
        return np.where(h == 0,  x + y,
               np.where(h == 1, -x + y,
               np.where(h == 2,  x - y,
                                 -x - y)))

    def noise_grid(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """2Dグリッド全体のノイズ値を一括計算する。xs/ysは(rows, cols)形状。"""
        xi = xs.astype(np.int32) & 255
        yi = ys.astype(np.int32) & 255
        xf = xs - np.floor(xs)
        yf = ys - np.floor(ys)
        # fade
        u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
        v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)
        p = self.perm
        aa = p[p[xi]     + yi]
        ab = p[p[xi]     + yi + 1]
        ba = p[p[xi + 1] + yi]
        bb = p[p[xi + 1] + yi + 1]
        x1 = xf - 1.0
        y1 = yf - 1.0
        g_aa = self._grad2d(aa, xf, yf)
        g_ba = self._grad2d(ba, x1, yf)
        g_ab = self._grad2d(ab, xf, y1)
        g_bb = self._grad2d(bb, x1, y1)
        lerp_a = g_aa + u * (g_ba - g_aa)
        lerp_b = g_ab + u * (g_bb - g_ab)
        return lerp_a + v * (lerp_b - lerp_a)

    def octave_grid(self, xs: np.ndarray, ys: np.ndarray,
                    octaves: int = 6, persistence: float = 0.5,
                    scale: float = 1.0) -> np.ndarray:
        """octave合成したノイズグリッド。"""
        val = np.zeros_like(xs, dtype=np.float32)
        amp, freq, mx = 1.0, scale, 0.0
        for _ in range(octaves):
            val += self.noise_grid(xs * freq, ys * freq).astype(np.float32) * amp
            mx  += amp; amp *= persistence; freq *= 2.0
        return val / mx


# ---- キノコエンティティ ----
# ================================================================
# キノコエンコード関数
# ================================================================
def encode_mushroom(biome: str, grade: str, variant: int, is_rotten: bool) -> np.ndarray:
    """キノコを構造化6次元ベクトルにエンコードする。
    [0:3] バイオーム one-hot: 沼(1,0,0)/平地(0,1,0)/山(0,0,1)
    [3]   栄養価: 普通=0.0 / 高栄養=1.0
    [4]   バリアント: 種①=0.0 / 種②=1.0
    [5]   腐敗: 新鮮=0.0 / 腐敗=1.0
    """
    enc = np.zeros(6, dtype=np.float32)
    biome_idx = {'W': 0, 'G': 1, 'M': 2}.get(biome, 0)
    enc[biome_idx] = 1.0
    enc[3] = 1.0 if grade == 'premium' else 0.0
    enc[4] = 1.0 if variant == 2 else 0.0
    enc[5] = 1.0 if is_rotten else 0.0
    return enc

def sage_vision(enc: np.ndarray) -> np.ndarray:
    """SAGEが受け取る弁別視野: バイオーム・腐敗をマスク。
    [0:3]=0固定 / [3]=栄養価 / [4]=バリアント / [5]=0固定
    """
    masked = enc.copy()
    masked[0:3] = 0.0
    masked[5]   = 0.0
    return masked

def brute_vision(enc: np.ndarray) -> np.ndarray:
    """BRUTEが受け取る弁別視野: 栄養価・バリアントをマスク。
    [0:3]=バイオーム / [3]=0固定 / [4]=0固定 / [5]=腐敗
    """
    masked = enc.copy()
    masked[3] = 0.0
    masked[4] = 0.0
    return masked


class Mushroom:
    """フィールド上のキノコ1個。"""
    __slots__ = ('pos', 'species_key', 'species_idx', 'biome', 'grade', 'is_rotten')

    def __init__(self, pos: pygame.Vector2, species_key: tuple, species_idx: int, is_rotten: bool):
        self.pos         = pos
        self.species_key = species_key          # ('W', 'normal', 1) など
        self.species_idx = species_idx          # 0〜11
        self.biome       = species_key[0]       # 'W'/'G'/'M'
        self.grade       = species_key[1]       # 'normal'/'premium'
        self.is_rotten   = is_rotten


class Field:
    """ワールド全体の地形・キノコ・ゴールを管理する。"""

    _terrain_cache: dict = {}
    _surface_cache: dict = {}
    _pass_cache:    dict = {}

    def __init__(self, terrain_seed: int = TERRAIN_SEED, food_episode: int = 0):
        self.terrain_seed = terrain_seed
        self.food_episode = food_episode

        if terrain_seed not in Field._terrain_cache:
            Field._terrain_cache[terrain_seed] = self._build_terrain(terrain_seed)
            pass_rng = random.Random(terrain_seed + 999)
            pass_y   = pass_rng.randint(int(WORLD_H * 0.3), int(WORLD_H * 0.7))
            Field._pass_cache[terrain_seed] = (WORLD_W // 2, pass_y)

        self.hmap     = Field._terrain_cache[terrain_seed]
        self.pass_pos = Field._pass_cache[terrain_seed]

        self.start_pos = (WORLD_W * 0.15, WORLD_H * 0.5)
        self.goal_pos  = (WORLD_W * 0.85, WORLD_H * 0.5)

        self.mushrooms: list[Mushroom] = []
        self._food_rng = random.Random(FOOD_SEED + food_episode)
        self._place_mushrooms()

    # ----------------------------------------------------------------
    @staticmethod
    def _build_terrain(seed: int) -> np.ndarray:
        """numpyベクトル化で地形を一括生成する（旧比約10倍高速）。"""
        pn   = PerlinNoise(seed)
        cols = WORLD_W // TILE
        rows = WORLD_H // TILE

        # グリッド座標をnumpy配列で一括生成
        gx_arr = np.arange(cols, dtype=np.float32) * TILE   # (cols,)
        gy_arr = np.arange(rows, dtype=np.float32) * TILE   # (rows,)
        xs, ys = np.meshgrid(gx_arr, gy_arr)                # (rows, cols)

        # octave合成ノイズを一括計算
        raw = pn.octave_grid(xs, ys, octaves=TERRAIN_OCTAVES, scale=TERRAIN_SCALE)
        lo, hi = raw.min(), raw.max()
        hmap = (raw - lo) / (hi - lo + 1e-8)

        # 山脈を追加（numpyベクトル化）
        pass_rng = random.Random(seed + 999)
        pass_y   = pass_rng.randint(int(WORLD_H * 0.3), int(WORLD_H * 0.7))
        cx = WORLD_W // 2; ridge_w = 80

        wy_arr = gy_arr                          # (rows,) 各行のy座標
        wx_arr = gx_arr                          # (cols,) 各列のx座標

        # 峰の通路以外の行を選択
        pass_mask_rows = np.abs(wy_arr - pass_y) >= PASS_WIDTH // 2   # (rows,)
        dist_ridge = np.abs(wx_arr - cx)                               # (cols,)
        ridge_mask_cols = dist_ridge < ridge_w                         # (cols,)
        strength_cols = np.where(ridge_mask_cols,
                                  1.0 - dist_ridge / ridge_w,
                                  0.0).astype(np.float32)              # (cols,)

        # ブロードキャストで山脈を追加
        add = np.outer(pass_mask_rows.astype(np.float32),
                       strength_cols) * 0.55                           # (rows, cols)
        hmap = np.clip(hmap + add, 0.0, 1.0)
        return hmap

    # ----------------------------------------------------------------
    def _biome_at(self, h: float) -> str:
        """高さ値からバイオーム文字を返す。W=沼/G=平地/M=山"""
        lo, hi = BIOME_THRESHOLDS
        if h < lo:  return 'W'
        if h < hi:  return 'G'
        return 'M'

    def _place_mushrooms(self):
        """グリッドベース均一配置でキノコを配置する。"""
        self.mushrooms.clear()
        rng = self._food_rng
        margin = 100

        xs = list(range(margin, WORLD_W - margin, FOOD_GRID_SPACING))
        ys = list(range(margin, WORLD_H - margin, FOOD_GRID_SPACING))

        candidates = []
        for gx in xs:
            for gy in ys:
                jx = rng.uniform(-FOOD_JITTER, FOOD_JITTER)
                jy = rng.uniform(-FOOD_JITTER, FOOD_JITTER)
                cx = max(margin, min(WORLD_W - margin, gx + jx))
                cy = max(margin, min(WORLD_H - margin, gy + jy))
                candidates.append((cx, cy))

        rng.shuffle(candidates)
        for cx, cy in candidates[:FOOD_COUNT]:
            h = self.height_at(cx, cy)
            biome = self._biome_at(h)
            # バイオームに合った種を選択
            biome_species = [k for k in MUSHROOM_SPECIES_LIST if k[0] == biome]
            species_key = rng.choice(biome_species)
            species_idx = MUSHROOM_SPECIES_LIST.index(species_key)
            is_rotten   = (rng.random() < ROT_PROBABILITY)
            self.mushrooms.append(
                Mushroom(pygame.Vector2(cx, cy), species_key, species_idx, is_rotten)
            )

    def reset_foods(self, food_episode: int | None = None):
        if food_episode is not None:
            self.food_episode = food_episode
        self._food_rng = random.Random(FOOD_SEED + self.food_episode)
        self._place_mushrooms()

    def clone_foods(self) -> 'Field':
        """各エージェント用の独立したフィールドコピーを返す。"""
        clone = Field.__new__(Field)
        clone.terrain_seed = self.terrain_seed
        clone.food_episode = self.food_episode
        clone.hmap         = self.hmap
        clone.pass_pos     = self.pass_pos
        clone.start_pos    = self.start_pos
        clone.goal_pos     = self.goal_pos
        clone.mushrooms    = [
            Mushroom(pygame.Vector2(m.pos), m.species_key, m.species_idx, m.is_rotten)
            for m in self.mushrooms
        ]
        clone._food_rng = random.Random(FOOD_SEED + self.food_episode)
        return clone

    # ----------------------------------------------------------------
    def height_at(self, wx: float, wy: float) -> float:
        gx = max(0, min(int(wx / TILE), self.hmap.shape[1] - 1))
        gy = max(0, min(int(wy / TILE), self.hmap.shape[0] - 1))
        return float(self.hmap[gy, gx])

    def gradient_at(self, wx: float, wy: float) -> tuple[float, float]:
        d = float(TILE)
        return ((self.height_at(wx + d, wy) - self.height_at(wx - d, wy)) / (2 * d),
                (self.height_at(wx, wy + d) - self.height_at(wx, wy - d)) / (2 * d))

    def biome_at(self, wx: float, wy: float) -> str:
        return self._biome_at(self.height_at(wx, wy))

    def biome_onehot(self, wx: float, wy: float) -> list[float]:
        b = self.biome_at(wx, wy)
        return [1.0 if b == 'W' else 0.0,
                1.0 if b == 'G' else 0.0,
                1.0 if b == 'M' else 0.0]

    def is_mountain(self, wx: float, wy: float) -> bool:
        return self.height_at(wx, wy) >= MOUNTAIN_THRESHOLD

    def is_valley(self, wx: float, wy: float) -> bool:
        return self.height_at(wx, wy) <= VALLEY_THRESHOLD

    # ----------------------------------------------------------------
    def collect_mushroom(self, wx: float, wy: float) -> Mushroom | None:
        """指定座標付近のキノコを回収して返す。なければNone。"""
        pos = pygame.Vector2(wx, wy)
        for i, m in enumerate(self.mushrooms):
            if pos.distance_to(m.pos) < MUSHROOM_RADIUS:
                return self.mushrooms.pop(i)
        return None

    def nearest_mushroom(self, wx: float, wy: float) -> Mushroom | None:
        """最寄りのキノコを返す（当たり判定外でも）。"""
        if not self.mushrooms:
            return None
        pos = pygame.Vector2(wx, wy)
        return min(self.mushrooms, key=lambda m: pos.distance_to(m.pos))

    def mushroom_in_focus(self, wx: float, wy: float, angle_deg: float,
                          focus_range: float = FOCUS_RANGE) -> Mushroom | None:
        """弁別視野（正面直線上）にあるキノコを返す。"""
        if not self.mushrooms:
            return None
        pos = pygame.Vector2(wx, wy)
        angle_rad = math.radians(angle_deg)
        forward   = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad))
        best, best_d = None, focus_range
        for m in self.mushrooms:
            diff = m.pos - pos
            d    = diff.length()
            if d > focus_range or d < 1e-6:
                continue
            dot = diff.normalize().dot(forward)
            if dot > 0.97:   # ±約14度以内
                if d < best_d:
                    best, best_d = m, d
        return best

    def respawn_mushroom(self):
        """キノコを1個補充する。"""
        rng = self._food_rng
        margin = 100
        x = rng.uniform(margin, WORLD_W - margin)
        y = rng.uniform(margin, WORLD_H - margin)
        h = self.height_at(x, y)
        biome = self._biome_at(h)
        biome_species = [k for k in MUSHROOM_SPECIES_LIST if k[0] == biome]
        species_key = rng.choice(biome_species)
        species_idx = MUSHROOM_SPECIES_LIST.index(species_key)
        is_rotten   = (rng.random() < ROT_PROBABILITY)
        self.mushrooms.append(
            Mushroom(pygame.Vector2(x, y), species_key, species_idx, is_rotten)
        )

    # ----------------------------------------------------------------
    def get_surface(self) -> pygame.Surface:
        if self.terrain_seed in Field._surface_cache:
            return Field._surface_cache[self.terrain_seed]

        rows, cols = self.hmap.shape
        rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
        h = self.hmap
        lo, hi = BIOME_THRESHOLDS

        # バイオーム新色でグラデーションあり（地形の勾配がわかりやすい）
        cW = np.array(BIOME_COLORS['W'], dtype=np.float32)
        cG = np.array(BIOME_COLORS['G'], dtype=np.float32)
        cM = np.array(BIOME_COLORS['M'], dtype=np.float32)

        w_mask = h < lo
        m_mask = h >= hi
        g_mask = ~w_mask & ~m_mask

        # 沼エリア: 深い水色→水色（深さで勾配表現）
        t_w = np.clip(h / (lo + 1e-8), 0, 1)
        for c in range(3):
            dark_w = cW[c] * 0.5
            rgb[w_mask, c] = (dark_w + (cW[c] - dark_w) * t_w[w_mask]).astype(np.uint8)

        # 山エリア: 平地色→山色（高さで勾配表現）
        t_m = np.clip((h - hi) / (1.0 - hi + 1e-8), 0, 1)
        for c in range(3):
            rgb[m_mask, c] = (cG[c] + (cM[c] - cG[c]) * t_m[m_mask]).astype(np.uint8)

        # 平地エリア: 水色→ミルク色（高さで勾配表現）
        t_g = np.clip((h - lo) / (hi - lo + 1e-8), 0, 1)
        for c in range(3):
            rgb[g_mask, c] = (cW[c] + (cG[c] - cW[c]) * t_g[g_mask]).astype(np.uint8)

        rgb_big = np.repeat(np.repeat(rgb, TILE, axis=0), TILE, axis=1)
        surf = pygame.surfarray.make_surface(rgb_big.transpose(1, 0, 2))
        Field._surface_cache[self.terrain_seed] = surf
        return surf

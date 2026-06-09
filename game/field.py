"""
field.py — 地形生成・餌・ゴール管理

設計方針:
  - 地形（hmap）は terrain_seed で完全固定。クラスレベルキャッシュにより
    同じシードなら何度インスタンス化しても再計算しない。
  - 餌は food_episode（エピソード番号）ごとに FOOD_SEED ベースで再配置。
    地形依存確率（谷バイアス）で出現し、山の上には高級餌が出現する。
"""
import math
import random
import numpy as np
import pygame
from config import *


# ---- シンプルなパーリンノイズ実装 ----
def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def _lerp(a, b, t):
    return a + t * (b - a)

def _grad(h, x, y):
    h &= 3
    if h == 0: return  x + y
    if h == 1: return -x + y
    if h == 2: return  x - y
    return -x - y

class PerlinNoise:
    def __init__(self, seed=0):
        rng = random.Random(seed)
        p = list(range(256))
        rng.shuffle(p)
        self.perm = p * 2

    def noise(self, x, y):
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        u, v = _fade(xf), _fade(yf)
        p = self.perm
        aa = p[p[xi]   + yi]
        ab = p[p[xi]   + yi + 1]
        ba = p[p[xi+1] + yi]
        bb = p[p[xi+1] + yi + 1]
        return _lerp(
            _lerp(_grad(aa, xf,   yf),   _grad(ba, xf-1, yf),   u),
            _lerp(_grad(ab, xf,   yf-1), _grad(bb, xf-1, yf-1), u),
            v)

    def octave(self, x, y, octaves=6, persistence=0.5, scale=1.0):
        val, amp, freq, mx = 0.0, 1.0, scale, 0.0
        for _ in range(octaves):
            val += self.noise(x * freq, y * freq) * amp
            mx  += amp
            amp *= persistence
            freq *= 2.0
        return val / mx


class Field:
    """ワールド全体の地形・餌・ゴールを管理する。

    地形（hmap）は terrain_seed で完全固定。
    餌はエピソードごとに food_episode で再配置される。
    foods の各要素は (Vector2, is_premium: bool) のタプル。
    """

    # 地形はクラスレベルキャッシュ（同じシードなら再計算不要）
    _terrain_cache: dict = {}   # terrain_seed -> hmap (np.ndarray)
    _surface_cache: dict = {}   # terrain_seed -> pygame.Surface
    _pass_cache:    dict = {}   # terrain_seed -> (pass_x, pass_y)  峰の中心座標

    def __init__(self, terrain_seed: int = 42, food_episode: int = 0):
        self.terrain_seed = terrain_seed
        self.food_episode = food_episode

        # ---- 地形（完全固定・キャッシュ） ----
        if terrain_seed not in Field._terrain_cache:
            Field._terrain_cache[terrain_seed] = self._build_terrain(terrain_seed)
            # 峰の中心座標もキャッシュ
            pass_rng = random.Random(terrain_seed + 999)
            pass_y   = pass_rng.randint(int(WORLD_H * 0.3), int(WORLD_H * 0.7))
            Field._pass_cache[terrain_seed] = (WORLD_W // 2, pass_y)

        self.hmap = Field._terrain_cache[terrain_seed]

        # 峰の中心座標（峰出口報酬エリアの生成に使用）
        self.pass_pos: tuple[int, int] = Field._pass_cache[terrain_seed]

        # ---- スタート・ゴール位置（地形固定） ----
        self.start_pos = (WORLD_W * 0.15, WORLD_H * 0.5)
        self.goal_pos  = (WORLD_W * 0.85, WORLD_H * 0.5)

        # ---- 餌の配置（エピソードごとに再配置） ----
        # foods: list of (Vector2, is_premium)
        self.foods: list[tuple[pygame.Vector2, bool]] = []
        self._food_rng = random.Random(FOOD_SEED + food_episode)
        self._place_foods()

    # ----------------------------------------------------------------
    @staticmethod
    def _build_terrain(seed: int) -> np.ndarray:
        """地形の高さマップを生成する（初回のみ）。"""
        pn   = PerlinNoise(seed)
        cols = WORLD_W // TILE
        rows = WORLD_H // TILE
        raw  = np.zeros((rows, cols), dtype=np.float32)
        for gy in range(rows):
            for gx in range(cols):
                raw[gy, gx] = pn.octave(
                    gx * TILE, gy * TILE,
                    octaves=TERRAIN_OCTAVES,
                    scale=TERRAIN_SCALE
                )
        lo, hi = raw.min(), raw.max()
        hmap = (raw - lo) / (hi - lo + 1e-8)

        # 中央に山脈を追加（峠の位置もシードで固定）
        pass_rng = random.Random(seed + 999)
        pass_y   = pass_rng.randint(int(WORLD_H * 0.3), int(WORLD_H * 0.7))
        cx       = WORLD_W // 2
        ridge_w  = 80
        for gy in range(rows):
            wy = gy * TILE
            if abs(wy - pass_y) < PASS_WIDTH // 2:
                continue
            for gx in range(cols):
                dist_ridge = abs(gx * TILE - cx)
                if dist_ridge < ridge_w:
                    strength = 1.0 - dist_ridge / ridge_w
                    hmap[gy, gx] = min(1.0, hmap[gy, gx] + strength * 0.55)
        return hmap

    # ----------------------------------------------------------------
    def _place_foods(self):
        """地形依存確率で餌を配置する。
        - 谷（低地）: 通常餌が高確率で出現
        - 山の上（FOOD_MOUNTAIN_THRESH以上）: 高級餌が出現
        - 峰出口エリア（峰の左右 PASS_REWARD_RADIUS px内）: 超高級餌を集中配置
        """
        self.foods.clear()
        placed   = 0
        attempts = 0
        rng      = self._food_rng

        # ---- 峰出口エリアに超高級餌を集中配置 ----
        # 峰の左側（スタート側）と右側（ゴール側）の出口仙近に集中
        px, py = self.pass_pos
        for side_dx in (-PASS_REWARD_RADIUS * 0.6, PASS_REWARD_RADIUS * 0.6):
            for _ in range(PASS_REWARD_COUNT):
                ox = rng.gauss(px + side_dx, PASS_REWARD_RADIUS * 0.3)
                oy = rng.gauss(py, PASS_REWARD_RADIUS * 0.4)
                ox = max(100, min(WORLD_W - 100, ox))
                oy = max(100, min(WORLD_H - 100, oy))
                self.foods.append((pygame.Vector2(ox, oy), True))
                placed += 1

        # ---- 通常配置 ----
        while placed < FOOD_COUNT and attempts < FOOD_COUNT * 30:
            attempts += 1
            x = rng.uniform(100, WORLD_W - 100)
            y = rng.uniform(100, WORLD_H - 100)
            h = self.height_at(x, y)

            # 高級餌ゾーン（山の上）
            if h >= FOOD_MOUNTAIN_THRESH:
                if rng.random() < 0.4:
                    self.foods.append((pygame.Vector2(x, y), True))
                    placed += 1
                continue

            # 通常餌ゾーン（谷バイアス）
            if h <= VALLEY_THRESHOLD:
                prob = FOOD_VALLEY_BIAS
            elif h >= MOUNTAIN_THRESHOLD:
                prob = 0.05
            else:
                t    = (h - VALLEY_THRESHOLD) / (MOUNTAIN_THRESHOLD - VALLEY_THRESHOLD)
                prob = FOOD_VALLEY_BIAS * (1.0 - t) + 0.05 * t

            if rng.random() < prob:
                self.foods.append((pygame.Vector2(x, y), False))
                placed += 1

    def reset_foods(self, food_episode: int | None = None):
        """餅をリセットして再配置する（エピソード開始時に呼ぶ）。"""
        if food_episode is not None:
            self.food_episode = food_episode
        self._food_rng = random.Random(FOOD_SEED + self.food_episode)
        self._place_foods()

    def clone_foods(self) -> 'Field':
        """現在の餅配置をコピーした独立なFieldインスタンスを返す。

        GA評価時に各エージェントに渡すことで、
        他のエージェントの餅取得に影響されず独立した取得状況を保証する。
        地形（hmap）・スタート・ゴール・峰座標は共有（変更なし）。
        餅リストのみディープコピーする。
        """
        import copy
        clone = Field.__new__(Field)
        # 地形・座標は元インスタンスと共有（読み取り専用なので安全）
        clone.terrain_seed = self.terrain_seed
        clone.food_episode = self.food_episode
        clone.hmap         = self.hmap          # numpy配列は共有（変更なし）
        clone.pass_pos     = self.pass_pos
        clone.start_pos    = self.start_pos
        clone.goal_pos     = self.goal_pos
        # 餅リストはディープコピー（各エージェントが独立に取得・削除できる）
        clone.foods = [(pygame.Vector2(pos), is_p) for pos, is_p in self.foods]
        # 餅補充用RNGは同じシードから再生成（補充順序が元と同じになる）
        clone._food_rng = random.Random(FOOD_SEED + self.food_episode)
        return clone

    # ----------------------------------------------------------------
    def height_at(self, wx: float, wy: float) -> float:
        gx = int(wx / TILE)
        gy = int(wy / TILE)
        gx = max(0, min(gx, self.hmap.shape[1] - 1))
        gy = max(0, min(gy, self.hmap.shape[0] - 1))
        return float(self.hmap[gy, gx])

    def gradient_at(self, wx: float, wy: float) -> tuple[float, float]:
        d = float(TILE)
        dh_x = (self.height_at(wx + d, wy) - self.height_at(wx - d, wy)) / (2 * d)
        dh_y = (self.height_at(wx, wy + d) - self.height_at(wx, wy - d)) / (2 * d)
        return dh_x, dh_y

    def is_mountain(self, wx: float, wy: float) -> bool:
        return self.height_at(wx, wy) >= MOUNTAIN_THRESHOLD

    def is_valley(self, wx: float, wy: float) -> bool:
        return self.height_at(wx, wy) <= VALLEY_THRESHOLD

    # ----------------------------------------------------------------
    def collect_food(self, wx: float, wy: float) -> tuple[bool, bool]:
        """指定座標付近の餌を回収する。
        Returns: (collected: bool, is_premium: bool)
        """
        pos = pygame.Vector2(wx, wy)
        for i, (food_pos, is_premium) in enumerate(self.foods):
            if pos.distance_to(food_pos) < FOOD_RADIUS:
                self.foods.pop(i)
                return True, is_premium
        return False, False

    def respawn_food(self):
        """餌を1個補充する（地形依存確率）。"""
        rng = self._food_rng
        for _ in range(300):
            x = rng.uniform(100, WORLD_W - 100)
            y = rng.uniform(100, WORLD_H - 100)
            h = self.height_at(x, y)

            if h >= FOOD_MOUNTAIN_THRESH:
                if rng.random() < 0.4:
                    self.foods.append((pygame.Vector2(x, y), True))
                    return
                continue

            if h <= VALLEY_THRESHOLD:
                prob = FOOD_VALLEY_BIAS
            elif h >= MOUNTAIN_THRESHOLD:
                prob = 0.05
            else:
                t    = (h - VALLEY_THRESHOLD) / (MOUNTAIN_THRESHOLD - VALLEY_THRESHOLD)
                prob = FOOD_VALLEY_BIAS * (1.0 - t) + 0.05 * t

            if rng.random() < prob:
                self.foods.append((pygame.Vector2(x, y), False))
                return

    # ----------------------------------------------------------------
    def get_surface(self) -> pygame.Surface:
        """地形サーフェスを生成してキャッシュする（terrain_seedごとに1回のみ）。
        内部では numpyでカラーマップを一括生成し、高速に Surface に転送する。
        """
        if self.terrain_seed in Field._surface_cache:
            return Field._surface_cache[self.terrain_seed]

        import numpy as np
        rows, cols = self.hmap.shape
        # numpyで全ピクセルのカラー配列を一括生成（ループ不要）
        rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
        h = self.hmap

        # 山地帯
        mt_mask = h >= MOUNTAIN_THRESHOLD
        t_mt = np.clip((h - MOUNTAIN_THRESHOLD) / (1.0 - MOUNTAIN_THRESHOLD + 1e-8), 0, 1)
        rgb[mt_mask, 0] = (C_PLAIN[0] + (C_MOUNTAIN[0] - C_PLAIN[0]) * t_mt[mt_mask]).astype(np.uint8)
        rgb[mt_mask, 1] = (C_PLAIN[1] + (C_MOUNTAIN[1] - C_PLAIN[1]) * t_mt[mt_mask]).astype(np.uint8)
        rgb[mt_mask, 2] = (C_PLAIN[2] + (C_MOUNTAIN[2] - C_PLAIN[2]) * t_mt[mt_mask]).astype(np.uint8)

        # 谷地帯
        vl_mask = h <= VALLEY_THRESHOLD
        t_vl = np.clip(1.0 - h / (VALLEY_THRESHOLD + 1e-8), 0, 1) * 0.5
        c2 = (20, 40, 25)
        rgb[vl_mask, 0] = (C_VALLEY[0] + (c2[0] - C_VALLEY[0]) * t_vl[vl_mask]).astype(np.uint8)
        rgb[vl_mask, 1] = (C_VALLEY[1] + (c2[1] - C_VALLEY[1]) * t_vl[vl_mask]).astype(np.uint8)
        rgb[vl_mask, 2] = (C_VALLEY[2] + (c2[2] - C_VALLEY[2]) * t_vl[vl_mask]).astype(np.uint8)

        # 平地帯
        pl_mask = ~mt_mask & ~vl_mask
        t_pl = np.clip((h - VALLEY_THRESHOLD) / (MOUNTAIN_THRESHOLD - VALLEY_THRESHOLD + 1e-8), 0, 1)
        rgb[pl_mask, 0] = (C_VALLEY[0] + (C_PLAIN[0] - C_VALLEY[0]) * t_pl[pl_mask]).astype(np.uint8)
        rgb[pl_mask, 1] = (C_VALLEY[1] + (C_PLAIN[1] - C_VALLEY[1]) * t_pl[pl_mask]).astype(np.uint8)
        rgb[pl_mask, 2] = (C_VALLEY[2] + (C_PLAIN[2] - C_VALLEY[2]) * t_pl[pl_mask]).astype(np.uint8)

        # TILEサイズに拡大（numpy repeat）
        rgb_big = np.repeat(np.repeat(rgb, TILE, axis=0), TILE, axis=1)

        # numpy配列から pygame.Surface へ変換
        surf = pygame.surfarray.make_surface(rgb_big.transpose(1, 0, 2))

        Field._surface_cache[self.terrain_seed] = surf
        return surf


def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )

"""
field.py — 地形生成・餌・ゴール管理
パーリンノイズで高さマップを生成し、谷/山/峠を配置する。
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
    """ワールド全体の地形・餌・ゴールを管理する。"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng  = random.Random(seed)
        self._pn  = PerlinNoise(seed)

        # 高さマップ（0〜1に正規化）
        cols = WORLD_W // TILE
        rows = WORLD_H // TILE
        raw = np.zeros((rows, cols), dtype=np.float32)
        for gy in range(rows):
            for gx in range(cols):
                wx = gx * TILE
                wy = gy * TILE
                raw[gy, gx] = self._pn.octave(
                    wx, wy,
                    octaves=TERRAIN_OCTAVES,
                    scale=TERRAIN_SCALE
                )
        lo, hi = raw.min(), raw.max()
        self.hmap = (raw - lo) / (hi - lo + 1e-8)   # 0〜1

        # 巨大な山脈をマップ中央に追加（スリット付き）
        self._carve_mountain_range()

        # スタート・ゴール位置
        self.start_pos = (WORLD_W * 0.15, WORLD_H * 0.5)
        self.goal_pos  = (WORLD_W * 0.85, WORLD_H * 0.5)

        # 餌の配置
        self.foods: list[pygame.Vector2] = []
        self._place_foods()

        # 地形サーフェス（描画用キャッシュ）
        self._surface: pygame.Surface | None = None

    # ----------------------------------------------------------------
    def _carve_mountain_range(self):
        """マップ中央に山脈を追加し、1箇所だけ峠（スリット）を作る。"""
        pass_y = self.rng.randint(
            int(WORLD_H * 0.3),
            int(WORLD_H * 0.7)
        )
        cx = WORLD_W // 2
        cols = WORLD_W // TILE
        rows = WORLD_H // TILE
        ridge_w = 80  # 山脈の幅 (px)

        for gy in range(rows):
            wy = gy * TILE
            dist_pass = abs(wy - pass_y)
            if dist_pass < PASS_WIDTH // 2:
                continue   # 峠は削らない
            for gx in range(cols):
                wx = gx * TILE
                dist_ridge = abs(wx - cx)
                if dist_ridge < ridge_w:
                    strength = 1.0 - dist_ridge / ridge_w
                    self.hmap[gy, gx] = min(
                        1.0,
                        self.hmap[gy, gx] + strength * 0.55
                    )

    # ----------------------------------------------------------------
    def _place_foods(self):
        """谷バイアスで餌を配置する。"""
        placed = 0
        attempts = 0
        while placed < FOOD_COUNT and attempts < FOOD_COUNT * 20:
            attempts += 1
            x = self.rng.uniform(100, WORLD_W - 100)
            y = self.rng.uniform(100, WORLD_H - 100)
            h = self.height_at(x, y)
            # 谷（低い場所）に高確率で配置
            if h < VALLEY_THRESHOLD:
                prob = FOOD_VALLEY_BIAS
            else:
                prob = 1.0 - FOOD_VALLEY_BIAS
            if self.rng.random() < prob:
                self.foods.append(pygame.Vector2(x, y))
                placed += 1

    # ----------------------------------------------------------------
    def height_at(self, wx: float, wy: float) -> float:
        """ワールド座標 (wx, wy) の高さ（0〜1）を返す。"""
        gx = int(wx / TILE)
        gy = int(wy / TILE)
        gx = max(0, min(gx, self.hmap.shape[1] - 1))
        gy = max(0, min(gy, self.hmap.shape[0] - 1))
        return float(self.hmap[gy, gx])

    def gradient_at(self, wx: float, wy: float) -> tuple[float, float]:
        """ワールド座標の勾配ベクトル (dx, dy) を返す（中央差分）。"""
        d = float(TILE)
        dh_x = (self.height_at(wx + d, wy) - self.height_at(wx - d, wy)) / (2 * d)
        dh_y = (self.height_at(wx, wy + d) - self.height_at(wx, wy - d)) / (2 * d)
        return dh_x, dh_y

    def is_mountain(self, wx: float, wy: float) -> bool:
        return self.height_at(wx, wy) >= MOUNTAIN_THRESHOLD

    def is_valley(self, wx: float, wy: float) -> bool:
        return self.height_at(wx, wy) <= VALLEY_THRESHOLD

    # ----------------------------------------------------------------
    def collect_food(self, wx: float, wy: float) -> bool:
        """指定座標付近の餌を回収する。回収できたら True を返す。"""
        pos = pygame.Vector2(wx, wy)
        for food in self.foods:
            if pos.distance_to(food) < FOOD_RADIUS:
                self.foods.remove(food)
                return True
        return False

    def respawn_food(self):
        """餌を1個補充する（谷バイアス）。"""
        for _ in range(200):
            x = self.rng.uniform(100, WORLD_W - 100)
            y = self.rng.uniform(100, WORLD_H - 100)
            h = self.height_at(x, y)
            if h < VALLEY_THRESHOLD:
                prob = FOOD_VALLEY_BIAS
            else:
                prob = 1.0 - FOOD_VALLEY_BIAS
            if self.rng.random() < prob:
                self.foods.append(pygame.Vector2(x, y))
                return

    # ----------------------------------------------------------------
    def get_surface(self) -> pygame.Surface:
        """地形サーフェスを生成してキャッシュする（初回のみ描画）。"""
        if self._surface is not None:
            return self._surface

        surf = pygame.Surface((WORLD_W, WORLD_H))
        rows, cols = self.hmap.shape
        for gy in range(rows):
            for gx in range(cols):
                h = float(self.hmap[gy, gx])
                if h >= MOUNTAIN_THRESHOLD:
                    t = (h - MOUNTAIN_THRESHOLD) / (1.0 - MOUNTAIN_THRESHOLD + 1e-8)
                    c = _lerp_color(C_PLAIN, C_MOUNTAIN, t)
                elif h <= VALLEY_THRESHOLD:
                    t = 1.0 - h / (VALLEY_THRESHOLD + 1e-8)
                    c = _lerp_color(C_VALLEY, (20, 40, 25), t * 0.5)
                else:
                    t = (h - VALLEY_THRESHOLD) / (MOUNTAIN_THRESHOLD - VALLEY_THRESHOLD)
                    c = _lerp_color(C_VALLEY, C_PLAIN, t)
                rect = pygame.Rect(gx * TILE, gy * TILE, TILE, TILE)
                surf.fill(c, rect)

        self._surface = surf
        return surf


def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )

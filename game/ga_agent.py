"""
ga_agent.py — GAゲノム管理・エージェント（Sage-Brute版）

GAGenome: SageNN + BruteNN の全重みをフラットベクトルで保持。
  合計 ≈ 5,415次元

エピソード処理:
  - 摂取履歴・中毒カウント・回復判定を管理
  - SAGEとBRUTEの観測ベクトルを構築
  - ボトルネック経由で行動を取得
"""
import math
import random
import numpy as np
import pygame
from config import *
from game.sage  import SageNN
from game.brute import BruteNN
from game.bottleneck import Bottleneck
from game.field import Field, Mushroom, encode_mushroom, sage_vision, brute_vision


# ================================================================
class GAGenome:
    """SAGE + BRUTE の全重みをフラットベクトルで保持するゲノム。"""

    def __init__(self, rng: random.Random | None = None):
        np_rng = np.random.default_rng(rng.randint(0, 2**31) if rng else None)
        self.sage  = SageNN(rng=np_rng)
        self.brute = BruteNN(rng=np_rng)
        self.fitness:    float = 0.0
        self.species_id: int   = -1

    def flat(self) -> np.ndarray:
        return np.concatenate([self.sage.flat(), self.brute.flat()])

    @staticmethod
    def total_flat_size() -> int:
        return SageNN.param_count() + BruteNN.param_count()

    def load_flat(self, arr: np.ndarray):
        n = SageNN.param_count()
        self.sage.load_flat(arr[:n])
        self.brute.load_flat(arr[n:])

    def reset_episode(self):
        self.sage.reset_episode()
        self.brute.reset_episode()

    @staticmethod
    def from_flat(v: np.ndarray) -> 'GAGenome':
        g = GAGenome.__new__(GAGenome)
        g.fitness    = 0.0
        g.species_id = -1
        g.sage  = SageNN.__new__(SageNN)
        g.brute = BruteNN.__new__(BruteNN)
        # 最小限の初期化
        rng = np.random.default_rng(0)
        tmp_s = SageNN(rng=rng)
        tmp_b = BruteNN(rng=rng)
        g.sage  = tmp_s
        g.brute = tmp_b
        g.load_flat(v)
        return g

    def mutate(self, rate: float, std: float) -> 'GAGenome':
        v = self.flat().copy()
        mask = np.random.random(v.shape) < rate
        v[mask] += np.random.normal(0, std, mask.sum())
        return GAGenome.from_flat(v)

    @staticmethod
    def crossover(a: 'GAGenome', b: 'GAGenome') -> 'GAGenome':
        va, vb = a.flat(), b.flat()
        mask = np.random.random(va.shape) < 0.5
        return GAGenome.from_flat(np.where(mask, va, vb))

    def distance(self, other: 'GAGenome') -> float:
        return float(np.linalg.norm(self.flat() - other.flat()))


# ================================================================
class SageBruteAgent:
    """1エピソードを実行するエージェント。"""

    def __init__(self, genome: GAGenome, field: Field,
                 start_pos: tuple[float, float]):
        self.genome = genome
        self.field  = field

        # 車体状態
        self.pos   = pygame.Vector2(*start_pos)
        self.angle = 0.0   # 度
        self.speed = 0.0
        self.energy = INIT_ENERGY
        self.alive  = True

        # ボトルネック
        self.bn = Bottleneck(genome.sage, genome.brute)
        self.bn.reset(prefill=True)

        # 摂取履歴・中毒管理
        self.intake_history: list[tuple] = []   # (species_key,) 直近HISTORY_LEN個
        self.toxic_count:    int = 0             # 同一種連続摂取カウント
        self.last_species:   tuple | None = None
        self.is_toxic:       bool = False        # 中毒状態

        # スコア
        self.total_reward:   float = 0.0
        self.goal_count:     int   = 0
        self.food_collected: int   = 0
        self.frame:          int   = 0

    def _build_sage_obs(self, received_pulse: int) -> np.ndarray:
        """SAGE用観測ベクトル（11次元）を構築する。"""
        # 弁別視野: 歩化エンコードしてSAGEマスクを適用
        focus_m = self.field.mushroom_in_focus(self.pos.x, self.pos.y, self.angle)
        if focus_m is not None:
            enc = encode_mushroom(focus_m.biome, focus_m.grade, focus_m.species_key[2], focus_m.is_rotten)
            vision = sage_vision(enc)   # [0:3]=0固定 / [3]=栄養価 / [4]=バリアント / [5]=0固定
        else:
            vision = np.zeros(MUSHROOM_ENC_DIM, dtype=np.float32)

        # ゴール角度・距離
        goal = pygame.Vector2(*self.field.goal_pos)
        diff = goal - self.pos
        dist_norm = min(1.0, diff.length() / (WORLD_W * 1.4))
        angle_to_goal = math.atan2(diff.y, diff.x)
        angle_diff = (math.radians(self.angle) - angle_to_goal + math.pi) % (2 * math.pi) - math.pi
        angle_norm = angle_diff / math.pi

        # エネルギー
        energy_norm = max(0.0, min(1.0, self.energy / MAX_ENERGY))

        # 受信パルス
        p_bits = [float((received_pulse >> 1) & 1), float(received_pulse & 1)]

        return np.concatenate([vision, [angle_norm, dist_norm, energy_norm], p_bits]).astype(np.float32)

    def _build_brute_obs(self, received_pulse: int) -> np.ndarray:
        """BRUTE用観測ベクトル（11次元）を構築する。"""
        # 弁別視野: 歩化エンコードしてBRUTEマスクを適用
        focus_m = self.field.mushroom_in_focus(self.pos.x, self.pos.y, self.angle)
        if focus_m is not None:
            enc = encode_mushroom(focus_m.biome, focus_m.grade, focus_m.species_key[2], focus_m.is_rotten)
            vision = brute_vision(enc)  # [0:3]=バイオーム / [3]=0固定 / [4]=0固定 / [5]=腐敗
        else:
            vision = np.zeros(MUSHROOM_ENC_DIM, dtype=np.float32)

        # 視覚レイ（±45度3本）
        # BRUTE_OBS_DIM=11: vision(6) + rays(3) + p_bits(2) = 11
        rays = []
        angle_rad = math.radians(self.angle)
        n_rays = BRUTE_OBS_DIM - MUSHROOM_ENC_DIM - 2   # = 11 - 6 - 2 = 3
        for i in range(n_rays):
            t = i / max(1, n_rays - 1)
            ray_angle = angle_rad + math.radians(VISION_ANGLE_DEG * (2 * t - 1))
            ray_val = 0.0
            for step in range(1, int(VISION_RANGE / 20) + 1):
                rx = self.pos.x + math.cos(ray_angle) * step * 20
                ry = self.pos.y + math.sin(ray_angle) * step * 20
                if not (0 <= rx < WORLD_W and 0 <= ry < WORLD_H):
                    ray_val = step * 20 / VISION_RANGE
                    break
            rays.append(ray_val)

        # 受信パルス
        p_bits = [float((received_pulse >> 1) & 1), float(received_pulse & 1)]

        return np.concatenate([vision, rays, p_bits]).astype(np.float32)

    def step(self) -> dict:
        """1フレーム処理。"""
        if not self.alive:
            return {'done': True, 'reward': 0.0}

        # 観測ベクトル構築（受信パルスは前フレームのもの）
        last_pulse = self.bn._last_pulse
        obs_sage  = self._build_sage_obs(last_pulse)
        obs_brute = self._build_brute_obs(last_pulse)

        # ボトルネック経由で行動取得
        action = self.bn.step(obs_sage, obs_brute)
        accel, steer, brake = float(action[0]), float(action[1]), float(action[2])

        # 車体物理更新
        self._physics_step(accel, steer, brake)

        # エネルギー消費
        self.energy -= ENERGY_DECAY

        # キノコ収集判定
        m = self.field.collect_mushroom(self.pos.x, self.pos.y)
        reward = 0.0
        if m is not None:
            reward += self._eat_mushroom(m)
            self.food_collected += 1
            self.field.respawn_mushroom()

        # ゴール判定
        goal = pygame.Vector2(*self.field.goal_pos)
        if (self.pos - goal).length() < GOAL_RADIUS:
            reward += ENERGY_GOAL
            self.energy = min(MAX_ENERGY, self.energy + ENERGY_GOAL)
            self.goal_count += 1
            self.intake_history.clear()
            self.toxic_count = 0
            self.last_species = None

        # 生存報酬
        reward += REWARD_SURVIVE
        if self.speed > IDLE_SPEED_THRESH:
            reward += REWARD_MOVE * self.speed

        self.total_reward += reward
        self.frame += 1

        # 死亡判定
        done = False
        if self.energy <= 0.0:
            self.alive = False
            reward += PENALTY_DEATH
            self.total_reward += PENALTY_DEATH
            done = True

        return {'done': done, 'reward': reward}

    def _physics_step(self, accel: float, steer: float, brake: float):
        """簡易車体物理。"""
        turn = (steer - 0.5) * 2.0 * CAR_TURN_SPEED
        self.angle = (self.angle + turn * self.speed / max(0.1, CAR_MAX_SPEED)) % 360

        target_speed = accel * CAR_MAX_SPEED * (1.0 - brake * CAR_BRAKE)
        self.speed += (target_speed - self.speed) * CAR_ACCEL
        self.speed *= (1.0 - CAR_FRICTION)
        self.speed  = max(0.0, min(CAR_MAX_SPEED, self.speed))

        angle_rad = math.radians(self.angle)
        self.pos.x += math.cos(angle_rad) * self.speed
        self.pos.y += math.sin(angle_rad) * self.speed
        self.pos.x = max(0, min(WORLD_W - 1, self.pos.x))
        self.pos.y = max(0, min(WORLD_H - 1, self.pos.y))

    def _eat_mushroom(self, m: Mushroom) -> float:
        """キノコを食べてエネルギー変化と報酬を計算する。"""
        if m.is_rotten:
            self.energy += ENERGY_ROTTEN
            self.energy = min(MAX_ENERGY, self.energy)
            return ENERGY_ROTTEN

        # 中毒チェック（同一種連続 TOXIC_COUNT 回）
        sk = m.species_key
        self.intake_history.append(sk)
        if len(self.intake_history) > HISTORY_LEN:
            self.intake_history.pop(0)

        # 連続カウント
        count = 0
        for h in reversed(self.intake_history):
            if h == sk:
                count += 1
            else:
                break
        self.toxic_count = count

        if count >= TOXIC_COUNT:
            self.energy += ENERGY_TOXIC
            self.energy = min(MAX_ENERGY, self.energy)
            return ENERGY_TOXIC

        # 回復チェック（直近 HISTORY_LEN 内に異バイオームの普通种2種）
        normals_in_history = [h for h in self.intake_history if h[1] == 'normal']
        biomes_seen = set(h[0] for h in normals_in_history)
        if len(biomes_seen) >= 2:
            gain = ENERGY_NORMAL * 1.5   # 回復ボーナス
        else:
            gain = MUSHROOM_SPECIES.get(sk, ENERGY_NORMAL)

        self.energy += gain
        self.energy = min(MAX_ENERGY, self.energy)
        return gain


# ================================================================
class Species:
    _id_counter = 0

    def __init__(self, representative: GAGenome):
        self.id = Species._id_counter
        Species._id_counter += 1
        self.representative = representative
        self.members: list[GAGenome] = [representative]
        self.best_fitness: float = representative.fitness
        self.stagnation: int = 0

    def update(self):
        if not self.members:
            return
        best = max(m.fitness for m in self.members)
        if best > self.best_fitness:
            self.best_fitness = best
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.representative = max(self.members, key=lambda m: m.fitness)


# ================================================================
class GeneticAlgorithm:
    SPECIES_THRESH   = 12.0
    STAGNATION_LIMIT = 15

    def __init__(self, pop_size: int = GA_POP_SIZE, seed: int = 0):
        self.rng         = random.Random(seed)
        self.generation  = 0
        self.pop_size    = pop_size
        self.elite_count = GA_ELITE

        self.population: list[GAGenome] = [
            GAGenome(rng=self.rng) for _ in range(pop_size)
        ]
        self.species: list[Species] = []

        self.best_fitness_history:  list[float] = []
        self.avg_fitness_history:   list[float] = []
        self.species_count_history: list[int]   = []

        self._mut_rate = GA_MUTATION_RATE
        self._mut_std  = GA_MUTATION_STD
        self._prev_best = -1e9

    def _speciate(self):
        for sp in self.species:
            sp.members.clear()
        for genome in self.population:
            placed = False
            for sp in self.species:
                if genome.distance(sp.representative) < self.SPECIES_THRESH:
                    sp.members.append(genome)
                    genome.species_id = sp.id
                    placed = True
                    break
            if not placed:
                new_sp = Species(genome)
                genome.species_id = new_sp.id
                self.species.append(new_sp)
        self.species = [sp for sp in self.species if sp.members]
        for sp in self.species:
            sp.update()
        if len(self.species) > 1:
            best_sp = max(self.species, key=lambda s: s.best_fitness)
            self.species = [
                sp for sp in self.species
                if sp.stagnation < self.STAGNATION_LIMIT or sp is best_sp
            ]

    def _adaptive_mutation(self):
        if not self.best_fitness_history:
            return
        current_best = self.best_fitness_history[-1]
        if current_best <= self._prev_best:
            self._mut_rate = min(0.5, self._mut_rate * 1.15)
            self._mut_std  = min(1.0, self._mut_std  * 1.10)
        else:
            self._mut_rate = max(0.05, self._mut_rate * 0.95)
            self._mut_std  = max(0.05, self._mut_std  * 0.95)
        self._prev_best = current_best

    def evolve(self):
        self._speciate()
        fits = [g.fitness for g in self.population]
        self.best_fitness_history.append(max(fits))
        self.avg_fitness_history.append(sum(fits) / len(fits))
        self.species_count_history.append(len(self.species))
        self._adaptive_mutation()

        new_pop: list[GAGenome] = []
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        new_pop.extend(sorted_pop[:self.elite_count])

        fits_min = min(fits)
        total_adj = sum(
            sum(m.fitness - fits_min + 1e-6 for m in sp.members)
            for sp in self.species
        )

        while len(new_pop) < self.pop_size:
            sp = self._select_species(total_adj, fits_min)
            if sp is None or not sp.members:
                parent = self._tournament(self.population)
                child = parent.mutate(self._mut_rate, self._mut_std)
            elif len(sp.members) == 1 or self.rng.random() < 0.25:
                parent = self._tournament(sp.members)
                child = parent.mutate(self._mut_rate, self._mut_std)
            else:
                p1 = self._tournament(sp.members)
                p2 = self._tournament(sp.members)
                child = GAGenome.crossover(p1, p2).mutate(
                    self._mut_rate * 0.5, self._mut_std * 0.5)
            new_pop.append(child)

        self.population = new_pop[:self.pop_size]
        self.generation += 1

    def _select_species(self, total_adj: float, fits_min: float):
        if not self.species or total_adj <= 0:
            return None
        r = self.rng.random() * total_adj
        acc = 0.0
        for sp in self.species:
            acc += sum(m.fitness - fits_min + 1e-6 for m in sp.members)
            if acc >= r:
                return sp
        return self.species[-1]

    def _tournament(self, pool: list[GAGenome], k: int = 3) -> GAGenome:
        candidates = self.rng.choices(pool, k=min(k, len(pool)))
        return max(candidates, key=lambda g: g.fitness)

    def get_best(self) -> GAGenome:
        return max(self.population, key=lambda g: g.fitness)

    def get_stats(self) -> dict:
        fits = [g.fitness for g in self.population]
        return {
            'generation': self.generation,
            'best':       max(fits),
            'avg':        sum(fits) / len(fits),
            'worst':      min(fits),
            'species':    len(self.species),
            'mut_rate':   self._mut_rate,
            'mut_std':    self._mut_std,
        }

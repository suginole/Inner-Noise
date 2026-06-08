"""
ga_agent.py — GA（遠伝的アルゴリズム）エージェント

現在のフェーズ: GAによるゲーム攻略可能性の検証。
ボトルネックは使用しない。
将来: RNN-GAフェーズでボトルネックを導入する。
"""
import random
import numpy as np
from game.agent import Agent
from game.car import Car
from game.field import Field
from config import *

# 観測ベクトルの全次元 = 基本6 + 視野VISION_RAYS + 弁別視野1
OBS_DIM = 6 + VISION_RAYS + 1


# ================================================================
class GAGenome:
    """
    3層フィードフォワードNN のゲノム。
    OBS_DIM → H1(16) → H2(12) → 3(行動)

    アクティベーション記録（可視化用）も保持する。
    """
    H1 = 16
    H2 = 12

    def __init__(self, rng: random.Random | None = None):
        if rng is None:
            rng = random.Random()
        s = 0.5
        self.W1   = np.array([[rng.gauss(0, s) for _ in range(OBS_DIM)] for _ in range(self.H1)])
        self.b1   = np.zeros(self.H1)
        self.W2   = np.array([[rng.gauss(0, s) for _ in range(self.H1)] for _ in range(self.H2)])
        self.b2   = np.zeros(self.H2)
        self.W3   = np.array([[rng.gauss(0, s) for _ in range(self.H2)] for _ in range(3)])
        self.b3   = np.zeros(3)

        self.fitness: float = 0.0
        self.species_id: int = -1

        # アクティベーション記録（可視化用）
        self.last_input_act:  list[float] = [0.0] * OBS_DIM
        self.last_hidden_act: list[float] = [0.0] * self.H1   # 第1隠れ層を表示
        self.last_output_act: list[float] = [0.5, 0.5, 0.0]
        self._last_bn_pulse:  list[int]   = [0, 0, 0, 0]

    # ----------------------------------------------------------------
    def forward(self, obs: list[float]) -> list[float]:
        x  = np.array(obs, dtype=np.float32)
        h1 = np.tanh(self.W1 @ x  + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        out = 1.0 / (1.0 + np.exp(-(self.W3 @ h2 + self.b3)))

        self.last_input_act  = x.tolist()
        self.last_hidden_act = h1.tolist()
        self.last_output_act = out.tolist()
        return out.tolist()

    # ----------------------------------------------------------------
    def flat(self) -> np.ndarray:
        """全重みを1次元ベクトルに変換（距離計算・種分化用）。"""
        return np.concatenate([
            self.W1.ravel(), self.b1,
            self.W2.ravel(), self.b2,
            self.W3.ravel(), self.b3,
        ])

    @staticmethod
    def from_flat(v: np.ndarray, rng=None) -> "GAGenome":
        g = GAGenome.__new__(GAGenome)
        g.fitness = 0.0
        g.species_id = -1
        g.last_input_act  = [0.0] * OBS_DIM
        g.last_hidden_act = [0.0] * GAGenome.H1
        g.last_output_act = [0.5, 0.5, 0.0]
        g._last_bn_pulse  = [0, 0, 0, 0]
        i = 0
        def take(n):
            nonlocal i
            r = v[i:i+n]; i += n; return r
        g.W1   = take(GAGenome.H1 * OBS_DIM).reshape(GAGenome.H1, OBS_DIM)
        g.b1   = take(GAGenome.H1)
        g.W2   = take(GAGenome.H2 * GAGenome.H1).reshape(GAGenome.H2, GAGenome.H1)
        g.b2   = take(GAGenome.H2)
        g.W3   = take(3 * GAGenome.H2).reshape(3, GAGenome.H2)
        g.b3   = take(3)
        return g

    # ----------------------------------------------------------------
    def mutate(self, rate: float, std: float) -> "GAGenome":
        v = self.flat().copy()
        mask = np.random.random(v.shape) < rate
        v[mask] += np.random.normal(0, std, mask.sum())
        return GAGenome.from_flat(v)

    @staticmethod
    def crossover(a: "GAGenome", b: "GAGenome") -> "GAGenome":
        va, vb = a.flat(), b.flat()
        mask = np.random.random(va.shape) < 0.5
        return GAGenome.from_flat(np.where(mask, va, vb))

    def distance(self, other: "GAGenome") -> float:
        """2個体間のユークリッド距離（種分化の判定に使用）。"""
        return float(np.linalg.norm(self.flat() - other.flat()))


# ================================================================
class Species:
    """種（スペシーズ）。近い個体をグループ化して多様性を維持する。"""
    _id_counter = 0

    def __init__(self, representative: GAGenome):
        self.id = Species._id_counter
        Species._id_counter += 1
        self.representative = representative
        self.members: list[GAGenome] = [representative]
        self.best_fitness: float = representative.fitness
        self.stagnation: int = 0   # 改善なしの世代数

    def update(self):
        if not self.members:
            return
        best = max(m.fitness for m in self.members)
        if best > self.best_fitness:
            self.best_fitness = best
            self.stagnation = 0
        else:
            self.stagnation += 1
        # 代表個体を更新（最高適応度の個体）
        self.representative = max(self.members, key=lambda m: m.fitness)


# ================================================================
class GeneticAlgorithm:
    """
    本格的なGA。
    - 3層NN ゲノム
    - エリート保存
    - トーナメント選択
    - 一様交叉 + 適応的突然変異
    - 種分化（Speciation）: 距離閾値で個体をグループ化し、
      停滞した種を淘汰して多様性を維持する
    """

    SPECIES_THRESH  = 8.0    # 種分化の距離閾値
    STAGNATION_LIMIT = 15    # この世代数改善なしで種を淘汰

    def __init__(self, pop_size: int = GA_POP_SIZE, seed: int = 0):
        self.rng        = random.Random(seed)
        self.generation = 0
        self.pop_size   = pop_size
        self.elite_count = GA_ELITE

        self.population: list[GAGenome] = [
            GAGenome(rng=self.rng) for _ in range(pop_size)
        ]
        self.species: list[Species] = []

        # 統計履歴
        self.best_fitness_history: list[float] = []
        self.avg_fitness_history:  list[float] = []
        self.species_count_history: list[int]  = []

        # 適応的突然変異パラメータ
        self._mut_rate = GA_MUTATION_RATE
        self._mut_std  = GA_MUTATION_STD
        self._prev_best = -1e9

    # ----------------------------------------------------------------
    def _speciate(self):
        """個体を種に割り当てる。"""
        # 既存の種のメンバーをクリア
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

        # 空の種を除去
        self.species = [sp for sp in self.species if sp.members]

        # 各種を更新（停滞カウント・代表更新）
        for sp in self.species:
            sp.update()

        # 長期停滞した種を淘汰（最良種は保護）
        if len(self.species) > 1:
            best_sp = max(self.species, key=lambda s: s.best_fitness)
            self.species = [
                sp for sp in self.species
                if sp.stagnation < self.STAGNATION_LIMIT or sp is best_sp
            ]

    # ----------------------------------------------------------------
    def _adaptive_mutation(self):
        """適応的突然変異率の調整。"""
        if not self.best_fitness_history:
            return
        current_best = self.best_fitness_history[-1]
        if current_best <= self._prev_best:
            # 改善なし → 突然変異率を上げる
            self._mut_rate = min(0.5, self._mut_rate * 1.15)
            self._mut_std  = min(1.0, self._mut_std  * 1.10)
        else:
            # 改善あり → 突然変異率を下げる
            self._mut_rate = max(0.05, self._mut_rate * 0.95)
            self._mut_std  = max(0.05, self._mut_std  * 0.95)
        self._prev_best = current_best

    # ----------------------------------------------------------------
    def evolve(self):
        """1世代分の進化を行う。"""
        # 種分化
        self._speciate()

        # 統計
        fits = [g.fitness for g in self.population]
        best = max(fits)
        avg  = sum(fits) / len(fits)
        self.best_fitness_history.append(best)
        self.avg_fitness_history.append(avg)
        self.species_count_history.append(len(self.species))

        # 適応的突然変異率の更新
        self._adaptive_mutation()

        # 新世代を生成
        new_pop: list[GAGenome] = []

        # エリート保存（全体からトップN）
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        new_pop.extend(sorted_pop[:self.elite_count])

        # 各種から比例的に子孫を生成
        total_adj_fit = 0.0
        for sp in self.species:
            sp_fits = [m.fitness - min(fits) + 1e-6 for m in sp.members]
            total_adj_fit += sum(sp_fits)

        while len(new_pop) < self.pop_size:
            # 種をランダムに選択（適応度比例）
            sp = self._select_species(total_adj_fit)
            if sp is None or len(sp.members) == 0:
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

    def _select_species(self, total_adj_fit: float) -> "Species | None":
        if not self.species or total_adj_fit <= 0:
            return None
        r = self.rng.random() * total_adj_fit
        acc = 0.0
        fits_min = min(g.fitness for g in self.population)
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
            "generation":   self.generation,
            "best":         max(fits),
            "avg":          sum(fits) / len(fits),
            "worst":        min(fits),
            "species":      len(self.species),
            "mut_rate":     self._mut_rate,
            "mut_std":      self._mut_std,
        }


# ================================================================
class GAAgent(Agent):
    """ゲノムを使って行動するエージェント。
    現在はボトルネックなし。毎フレーム forward() で直接行動を決定。
    """

    def __init__(self, car: Car, field: Field, genome: GAGenome):
        super().__init__(car, field)
        self.genome = genome

    def act(self) -> tuple[float, float, float]:
        obs    = self.car.get_observation(self.field)
        action = self.genome.forward(obs)
        return tuple(max(0.0, min(1.0, v)) for v in action[:3])

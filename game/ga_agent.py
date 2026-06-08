"""
ga_agent.py — シンプルGA学習エージェント
観測ベクトル(6次元) → 線形変換 → 行動(3次元) のシンプルなゲノム。
ボトルネックモジュールを経由して行動を生成する。
将来: ゲノムの中身をRNN+NEATに差し替える。
"""
import numpy as np
import random
from game.agent import Agent
from game.car import Car
from game.field import Field
from game.bottleneck import Bottleneck
from config import *

# 観測ベクトルの全次元 = 基本6 + 視野VISION_RAYS
OBS_DIM = 6 + VISION_RAYS


class GAGenome:
    """
    シンプルGAのゲノム。
    観測(OBS_DIM) → 隠れ層(12) → 行動(3) の2層NN。
    ボトルネック通信路の重みも含む。
    """
    def __init__(self, rng: random.Random | None = None):
        if rng is None:
            rng = random.Random()
        # W1: (12, OBS_DIM), b1: (12,)
        # W2: (3, 12), b2: (3,)
        # bn_w: (3, 4)  ボトルネックデコード重み
        self.W1   = np.array([[rng.gauss(0, 0.5) for _ in range(OBS_DIM)] for _ in range(12)])
        self.b1   = np.array([rng.gauss(0, 0.3) for _ in range(12)])
        self.W2   = np.array([[rng.gauss(0, 0.5) for _ in range(12)] for _ in range(3)])
        self.b2   = np.array([rng.gauss(0, 0.3) for _ in range(3)])
        self.bn_w = np.array([[rng.gauss(0, 0.5) for _ in range(4)] for _ in range(3)])
        self.fitness = 0.0
        # アクティベーション記録（初期値）
        self.last_input_act:  list[float] = [0.0] * OBS_DIM
        self.last_hidden_act: list[float] = [0.0] * 12
        self.last_output_act: list[float] = [0.5, 0.5, 0.0]

    def forward(self, obs: list[float]) -> list[float]:
        """観測ベクトルから行動を計算する（ボトルネックを経由しない直接版）。"""
        x = np.array(obs, dtype=np.float32)
        h = np.tanh(self.W1 @ x + self.b1)
        out = self.W2 @ h + self.b2
        action = 1.0 / (1.0 + np.exp(-out))
        # アクティベーションを記録（可視化用）
        self.last_input_act  = x.tolist()          # 観測ベクトル（正規化済み）
        self.last_hidden_act = h.tolist()           # 隠れ層（tanh: -1〜1）
        self.last_output_act = action.tolist()      # 出力層（sigmoid: 0〜1）
        return action.tolist()

    def mutate(self, rate: float = GA_MUTATION_RATE,
               std: float = GA_MUTATION_STD) -> "GAGenome":
        child = GAGenome.__new__(GAGenome)
        child.W1   = self.W1.copy()
        child.b1   = self.b1.copy()
        child.W2   = self.W2.copy()
        child.b2   = self.b2.copy()
        child.bn_w = self.bn_w.copy()
        child.fitness = 0.0
        for arr in [child.W1, child.b1, child.W2, child.b2, child.bn_w]:
            mask = np.random.random(arr.shape) < rate
            arr += mask * np.random.normal(0, std, arr.shape)
        return child

    @staticmethod
    def crossover(a: "GAGenome", b: "GAGenome") -> "GAGenome":
        child = GAGenome.__new__(GAGenome)
        child.fitness = 0.0
        for attr in ["W1", "b1", "W2", "b2", "bn_w"]:
            pa, pb = getattr(a, attr), getattr(b, attr)
            mask = np.random.random(pa.shape) < 0.5
            setattr(child, attr, np.where(mask, pa, pb))
        return child


class GAAgent(Agent):
    """
    GAゲノムを使って行動するエージェント。
    ボトルネックモジュールを経由する（スタブ）。
    """

    def __init__(self, car: Car, field: Field, genome: GAGenome):
        super().__init__(car, field)
        self.genome     = genome
        self.bottleneck = Bottleneck(weights=genome.bn_w)

    def act(self) -> tuple[float, float, float]:
        obs = self.car.get_observation(self.field)
        # 感覚系 → ボトルネック
        self.bottleneck.push(obs)
        dt = 1.0 / FPS
        pulse, mode = self.bottleneck.tick(dt)
        # 発話ターン中はボトルネック経由の行動を使う
        if mode == "speak":
            action = self.bottleneck.get_action()
            # 行動をゲノムの出力アクティベーションに反映
            self.genome.last_output_act = list(action)
        else:
            # 傾聴ターン中は直接ゲノムで行動（スタブ）
            action = self.genome.forward(obs)
        # 現在のパルスをゲノムに保存（可視化用）
        self.genome._last_bn_pulse = self.bottleneck.get_current_pulse()
        return tuple(max(0.0, min(1.0, v)) for v in action[:3])

    def reset(self):
        super().reset()
        self.bottleneck.reset()


# ----------------------------------------------------------------
class GeneticAlgorithm:
    """
    シンプルなGA（エリート選択 + 突然変異 + 交叉）。
    将来: NEAT に差し替えるためのラッパー。
    """

    def __init__(self, pop_size: int = GA_POP_SIZE, seed: int = 0):
        self.rng = random.Random(seed)
        self.generation = 0
        self.population: list[GAGenome] = [
            GAGenome(rng=self.rng) for _ in range(pop_size)
        ]
        self.pop_size    = pop_size
        self.elite_count = GA_ELITE
        self.best_fitness_history: list[float] = []
        self.avg_fitness_history:  list[float] = []

    def evolve(self):
        """1世代分の進化を行う。"""
        # 適応度でソート（降順）
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        best = self.population[0].fitness
        avg  = sum(g.fitness for g in self.population) / len(self.population)
        self.best_fitness_history.append(best)
        self.avg_fitness_history.append(avg)

        # エリート保存
        new_pop = self.population[:self.elite_count]

        # 残りを交叉・突然変異で補充
        while len(new_pop) < self.pop_size:
            if self.rng.random() < 0.6:
                # 交叉
                p1 = self._tournament()
                p2 = self._tournament()
                child = GAGenome.crossover(p1, p2).mutate()
            else:
                # 突然変異のみ
                parent = self._tournament()
                child = parent.mutate()
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1

    def _tournament(self, k: int = 3) -> GAGenome:
        candidates = self.rng.choices(self.population, k=k)
        return max(candidates, key=lambda g: g.fitness)

    def get_best(self) -> GAGenome:
        return max(self.population, key=lambda g: g.fitness)

    def get_stats(self) -> dict:
        fits = [g.fitness for g in self.population]
        return {
            "generation": self.generation,
            "best":  max(fits),
            "avg":   sum(fits) / len(fits),
            "worst": min(fits),
        }

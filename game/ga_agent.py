"""
ga_agent.py — バッファGRU挿入型ゲノムを使ったGAエージェント

GAGenome: SensoryNN + MotorNN の全重みをフラットベクトルとして保持。
  合計 2,729次元

GAが進化させるもの:
  - 第三層FF重み（W3, b3）
  - バッファGRU重み（9行列）
  - 記憶GRU重み（9行列）
  - γ（記憶GRU上位6次元の初期値）
  - 通常FF重み（W_bypass, b_bypass）
  - 第一層FF重み（W1, b1）
  - センサリー: パルス符号化FF重み（W_encode, b_encode）
  - モーター: 出力FF重み（W_out, b_out）

オンライン更新（GAが進化させない）:
  - 記憶GRUの隠れ状態（エピソード内のみ）
  - バッファGRUの隠れ状態・buf_out（エピソード内のみ）
"""
import random
import numpy as np
from game.agent import Agent
from game.car import Car
from game.field import Field
from game.rnn_bottleneck import SensoryNN, MotorNN, RNNBottleneck
from config import *

# 観測ベクトルの全次元
OBS_DIM = 6 + VISION_RAYS + 1   # = 12


# ================================================================
class GAGenome:
    """
    バッファGRU挿入型ゲノム。
    SensoryNN + MotorNN の全重みをフラットベクトルで保持する。
    """

    def __init__(self, rng: random.Random | None = None):
        np_rng = np.random.default_rng(
            rng.randint(0, 2**31) if rng else None
        )
        self.sensory = SensoryNN(rng=np_rng)
        self.motor   = MotorNN(rng=np_rng)

        self.fitness:    float = 0.0
        self.species_id: int   = -1

        # アクティベーション記録（可視化用）
        self.last_input_act:        list[float] = [0.0] * OBS_DIM
        self.last_l3_act:           list[float] = [0.0] * L3_OUT_DIM       # 第三層FF出力(24)
        self.last_sensory_buf:      list[float] = [0.0] * BUF_GRU_DIM      # センサリーバッファGRU(5)
        self.last_sensory_buf_active: bool = False
        self.last_sensory_gru:      list[float] = [0.0] * MEM_GRU_DIM      # 感覚記憶GRU(12)
        self.last_pulse_act:        list[int]   = [0, 0]                    # パルス符号化後
        self.last_motor_l3_act:     list[float] = [0.0] * L3_OUT_DIM       # モーター第三層FF(24)
        self.last_motor_buf:        list[float] = [0.0] * BUF_GRU_DIM      # モーターバッファGRU(5)
        self.last_motor_buf_active: bool = False
        self.last_motor_gru:        list[float] = [0.0] * MEM_GRU_DIM      # 運動記憶GRU(12)
        self.last_output_act:       list[float] = [0.5, 0.5, 0.0]          # 出力
        # 互換性維持用エイリアス（旧名→新名）
        self.last_cortex_act:       list[float] = self.last_l3_act
        self.last_sensory_integ:    list[float] = [0.0] * L1_OUT_DIM       # (24)
        self.last_embed_act:        list[float] = self.last_motor_l3_act
        self.last_motor_integ:      list[float] = [0.0] * L1_OUT_DIM       # (24)
        self.last_motor_cortex_act: list[float] = [0.0] * L1_OUT_DIM       # (24)
        self.last_hidden_act:       list[float] = self.last_l3_act
        self.last_hidden2_act:      list[float] = self.last_sensory_gru

    # ----------------------------------------------------------------
    def flat(self) -> np.ndarray:
        return np.concatenate([self.sensory.flat(), self.motor.flat()])

    @staticmethod
    def total_flat_size() -> int:
        return SensoryNN.flat_size() + MotorNN.flat_size()

    @staticmethod
    def from_flat(v: np.ndarray) -> "GAGenome":
        g = GAGenome.__new__(GAGenome)
        g.fitness    = 0.0
        g.species_id = -1
        g.last_input_act          = [0.0] * OBS_DIM
        g.last_l3_act             = [0.0] * L3_OUT_DIM       # (24)
        g.last_sensory_buf        = [0.0] * BUF_GRU_DIM      # (5)
        g.last_sensory_buf_active = False
        g.last_sensory_gru        = [0.0] * MEM_GRU_DIM      # (12)
        g.last_pulse_act          = [0, 0]
        g.last_motor_l3_act       = [0.0] * L3_OUT_DIM       # (24)
        g.last_motor_buf          = [0.0] * BUF_GRU_DIM      # (5)
        g.last_motor_buf_active   = False
        g.last_motor_gru          = [0.0] * MEM_GRU_DIM      # (12)
        g.last_output_act         = [0.5, 0.5, 0.0]
        g.last_cortex_act         = g.last_l3_act
        g.last_sensory_integ      = [0.0] * L1_OUT_DIM       # (24)
        g.last_embed_act          = g.last_motor_l3_act
        g.last_motor_integ        = [0.0] * L1_OUT_DIM       # (24)
        g.last_motor_cortex_act   = [0.0] * L1_OUT_DIM       # (24)
        g.last_hidden_act         = g.last_l3_act
        g.last_hidden2_act        = g.last_sensory_gru

        ss = SensoryNN.flat_size()

        g.sensory = SensoryNN.__new__(SensoryNN)
        g.sensory.last_l3_act      = [0.0] * L3_OUT_DIM      # (24)
        g.sensory.last_buf_act     = [0.0] * BUF_GRU_DIM     # (5)
        g.sensory.last_buf_active  = False
        g.sensory.last_gru_act     = [0.0] * MEM_GRU_DIM     # (12)
        g.sensory.last_bypass_act  = [0.0] * BYPASS_FF_DIM   # (16)
        g.sensory.last_integ_act   = [0.0] * L1_OUT_DIM      # (24)
        g.sensory.last_pulse       = [0, 0]
        g.sensory.last_input_act   = g.sensory.last_l3_act
        g.sensory.last_cortex_act  = g.sensory.last_l3_act
        g.sensory._h_mem           = np.zeros(MEM_GRU_DIM)
        g.sensory._h_buf           = np.zeros(BUF_GRU_DIM)
        g.sensory._buf_out         = np.zeros(BUF_GRU_DIM)
        g.sensory.gamma            = np.zeros(GRU_INHERIT_DIM)
        g.sensory.load_flat(v[:ss])
        g.sensory._h_mem           = g.sensory._init_h_mem()

        g.motor = MotorNN.__new__(MotorNN)
        g.motor.last_l3_act        = [0.0] * L3_OUT_DIM      # (24)
        g.motor.last_buf_act       = [0.0] * BUF_GRU_DIM     # (5)
        g.motor.last_buf_active    = False
        g.motor.last_gru_act       = [0.0] * MEM_GRU_DIM     # (12)
        g.motor.last_bypass_act    = [0.0] * BYPASS_FF_DIM   # (16)
        g.motor.last_integ_act     = [0.0] * L1_OUT_DIM      # (24)
        g.motor.last_output_act    = [0.5, 0.5, 0.0]
        g.motor.last_embed_act     = g.motor.last_l3_act
        g.motor.last_cortex_act    = g.motor.last_integ_act
        g.motor._h_mem             = np.zeros(MEM_GRU_DIM)
        g.motor._h_buf             = np.zeros(BUF_GRU_DIM)
        g.motor._buf_out           = np.zeros(BUF_GRU_DIM)
        g.motor.gamma              = np.zeros(GRU_INHERIT_DIM)
        g.motor.load_flat(v[ss:])
        g.motor._h_mem             = g.motor._init_h_mem()
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
        return float(np.linalg.norm(self.flat() - other.flat()))

    def update_activations(self, bn, obs: list[float]):
        """全層のアクティベーションをゲノムに同期する（可視化用）。"""
        self.last_input_act          = list(obs)
        self.last_l3_act             = list(getattr(bn.sensory, 'last_l3_act',      [0.0]*L3_OUT_DIM))
        self.last_sensory_buf        = list(getattr(bn.sensory, 'last_buf_act',     [0.0]*BUF_GRU_DIM))
        self.last_sensory_buf_active = bool(getattr(bn.sensory, 'last_buf_active',  False))
        self.last_sensory_gru        = list(getattr(bn, 'last_sensory_gru',         [0.0]*MEM_GRU_DIM))
        self.last_sensory_integ      = list(getattr(bn.sensory, 'last_integ_act',   [0.0]*L1_OUT_DIM))
        self.last_pulse_act          = list(getattr(bn.sensory, 'last_pulse',       [0, 0]))
        self.last_motor_l3_act       = list(getattr(bn.motor, 'last_l3_act',        [0.0]*L3_OUT_DIM))
        self.last_motor_buf          = list(getattr(bn.motor, 'last_buf_act',       [0.0]*BUF_GRU_DIM))
        self.last_motor_buf_active   = bool(getattr(bn.motor, 'last_buf_active',    False))
        self.last_motor_gru          = list(getattr(bn, 'last_motor_gru',           [0.0]*MEM_GRU_DIM))
        self.last_motor_integ        = list(getattr(bn.motor, 'last_integ_act',     [0.0]*L1_OUT_DIM))
        self.last_motor_cortex_act   = list(getattr(bn.motor, 'last_integ_act',     [0.0]*L1_OUT_DIM))
        self.last_output_act         = list(getattr(bn, 'last_output',              [0.5, 0.5, 0.0]))
        # エイリアスも更新
        self.last_cortex_act   = self.last_l3_act
        self.last_embed_act    = self.last_motor_l3_act
        self.last_hidden_act   = self.last_l3_act
        self.last_hidden2_act  = self.last_sensory_gru


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

        self.best_fitness_history:   list[float] = []
        self.avg_fitness_history:    list[float] = []
        self.species_count_history:  list[int]   = []

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
        best = max(fits)
        avg  = sum(fits) / len(fits)
        self.best_fitness_history.append(best)
        self.avg_fitness_history.append(avg)
        self.species_count_history.append(len(self.species))
        self._adaptive_mutation()

        new_pop: list[GAGenome] = []
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        new_pop.extend(sorted_pop[:self.elite_count])

        total_adj_fit = 0.0
        fits_min = min(fits)
        for sp in self.species:
            total_adj_fit += sum(m.fitness - fits_min + 1e-6 for m in sp.members)

        while len(new_pop) < self.pop_size:
            sp = self._select_species(total_adj_fit, fits_min)
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

    def _select_species(self, total_adj_fit: float, fits_min: float):
        if not self.species or total_adj_fit <= 0:
            return None
        r = self.rng.random() * total_adj_fit
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
            "generation": self.generation,
            "best":       max(fits),
            "avg":        sum(fits) / len(fits),
            "worst":      min(fits),
            "species":    len(self.species),
            "mut_rate":   self._mut_rate,
            "mut_std":    self._mut_std,
        }


# ================================================================
class GAAgent(Agent):
    """バッファGRUゲノムを使って行動するエージェント。"""

    def __init__(self, car: Car, field: Field, genome: GAGenome):
        super().__init__(car, field)
        self.genome = genome
        self.bn = RNNBottleneck(genome.sensory, genome.motor)
        self.bn.reset()

    def act(self) -> tuple[float, float, float]:
        obs    = self.car.get_observation(self.field)
        action = self.bn.step(obs)
        self.genome.update_activations(self.bn, obs)
        return tuple(max(0.0, min(1.0, v)) for v in action[:3])

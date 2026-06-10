"""
ga_agent.py — GAゲノム管理・エージェント（Sage-Brute版）
"""
import math
import copy
import numpy as np
import pygame
from config import *
from game.sage     import SageNN
from game.brute    import BruteNN
from game.bottleneck import Bottleneck


class GAGenome:
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.sage    = SageNN(rng)
        self.brute   = BruteNN(rng)
        self.fitness = 0.0
        self.species_id = -1

    # ---- renderer互換プロパティ ----
    # renderer.pyは genome.sensory / genome.motor を期待する
    @property
    def sensory(self): return self.sage
    @property
    def motor(self):   return self.brute

    # rendererが getattr(genome, 'last_*') で参照するアクティベーション属性
    # SAGE→sensory側のエイリアス
    @property
    def last_input_act(self):         return [0.0] * SAGE_OBS_DIM
    @property
    def last_l3_act(self):            return self.sage.last_l3_act
    @property
    def last_sensory_buf(self):       return self.sage.last_buf_act
    @property
    def last_sensory_buf_active(self): return self.sage.last_buf_active
    @property
    def last_sensory_gru(self):       return self.sage.last_gru_act
    @property
    def last_pulse_act(self):
        p = self.sage.last_pulse
        return [(p >> 1) & 1, p & 1]
    # BRUTE→motor側のエイリアス
    @property
    def last_motor_l3_act(self):      return self.brute.last_l3_act
    @property
    def last_motor_buf(self):         return self.brute.last_buf_act
    @property
    def last_motor_buf_active(self):  return self.brute.last_buf_active
    @property
    def last_motor_gru(self):         return self.brute.last_gru_act
    @property
    def last_output_act(self):        return self.brute.last_output_act

    def flat(self):
        return np.concatenate([self.sage.flat(), self.brute.flat()])

    def load_flat(self, arr):
        n = self.sage.param_count()
        self.sage.load_flat(arr[:n])
        self.brute.load_flat(arr[n:])

    def reset_episode(self):
        self.sage.reset_episode()
        self.brute.reset_episode()

    @staticmethod
    def total_flat_size():
        tmp_rng = np.random.default_rng(0)
        s = SageNN(tmp_rng)
        b = BruteNN(tmp_rng)
        return s.param_count() + b.param_count()

    @staticmethod
    def from_flat(arr):
        g = GAGenome()
        g.load_flat(arr)
        return g


class SageBruteAgent:
    def __init__(self, genome, field, start_pos):
        self.genome         = genome
        self.field          = field
        self.pos            = pygame.Vector2(*start_pos)
        self.angle          = 0.0
        self.speed          = 0.0
        self.energy         = INIT_ENERGY
        self.alive          = True
        self.total_reward   = 0.0
        self.intake_history = []
        self.toxic_count    = 0
        self.bn = Bottleneck(genome.sage, genome.brute)
        self.bn.reset_episode()

    def step(self):
        obs_sage  = self._get_obs_sage()
        obs_brute = self._get_obs_brute()
        action    = self.bn.step(obs_sage, obs_brute)

        accel = float(action[0])
        steer = float(action[1])
        brake = float(action[2])

        self.speed += (accel - brake) * CAR_ACCEL
        self.speed  = max(0, min(CAR_MAX_SPEED, self.speed))
        self.speed *= (1 - CAR_FRICTION)
        self.angle += (steer - 0.5) * CAR_TURN_SPEED
        rad = math.radians(self.angle)
        self.pos.x += math.cos(rad) * self.speed
        self.pos.y += math.sin(rad) * self.speed
        self.pos.x  = max(0, min(FIELD_SIZE - 1, self.pos.x))
        self.pos.y  = max(0, min(FIELD_SIZE - 1, self.pos.y))

        self.energy -= ENERGY_DECAY
        self.energy -= self.speed * 0.001

        # キノコとの当たり判定
        for m in self.field.mushrooms[:]:
            if (self.pos - m.pos).length() < FOOD_RADIUS:
                self._eat_mushroom(m)
                self.field.mushrooms.remove(m)

        reward = REWARD_SURVIVE
        if self.speed > 0.1:
            reward += REWARD_MOVE * self.speed

        goal_dist = (self.pos - pygame.Vector2(*self.field.goal_pos)).length()
        if goal_dist < GOAL_RADIUS:
            self.energy = min(MAX_ENERGY, self.energy + ENERGY_GOAL)
            reward += REWARD_GOAL_SB
            self.intake_history = []
            self.toxic_count = 0

        self.total_reward += reward
        done = self.energy <= 0
        if done:
            self.alive = False
        return {'done': done, 'reward': reward}

    def _eat_mushroom(self, m):
        if m.is_rotten:
            self.energy += ENERGY_ROTTEN
            self.energy  = max(0, min(MAX_ENERGY, self.energy))
            return
        sk = (m.biome, m.grade, m.variant)
        self.intake_history.append(sk)
        if len(self.intake_history) > HISTORY_LEN:
            self.intake_history.pop(0)
        count = 0
        for h in reversed(self.intake_history):
            if h == sk: count += 1
            else: break
        self.toxic_count = count
        if count >= TOXIC_COUNT:
            self.energy += ENERGY_TOXIC
            self.energy  = max(0, min(MAX_ENERGY, self.energy))
            return
        normals = [h for h in self.intake_history if h[1] == 'normal']
        biomes  = set(h[0] for h in normals)
        if len(biomes) >= 2:
            self.energy += ENERGY_NORMAL * 1.5
        else:
            self.energy += MUSHROOM_SPECIES.get(sk, ENERGY_NORMAL)
        self.energy = max(0, min(MAX_ENERGY, self.energy))

    def _get_obs_sage(self):
        enc = self._get_mushroom_enc()
        sage_enc = enc.copy()
        sage_enc[0:3] = 0.0
        sage_enc[5]   = 0.0
        goal = pygame.Vector2(*self.field.goal_pos) - self.pos
        dist = goal.length()
        ang  = math.atan2(goal.y, goal.x) - math.radians(self.angle)
        ang  = math.atan2(math.sin(ang), math.cos(ang)) / math.pi
        return np.array([
            *sage_enc,
            float(ang),
            min(1.0, dist / FIELD_SIZE),
            self.energy / MAX_ENERGY,
            0.0, 0.0,
        ], dtype=np.float32)

    def _get_obs_brute(self):
        enc = self._get_mushroom_enc()
        brute_enc = enc.copy()
        brute_enc[3] = 0.0
        brute_enc[4] = 0.0
        rays = self._cast_rays()
        return np.array([
            *brute_enc,
            *rays,
            0.0, 0.0,
        ], dtype=np.float32)

    def _get_mushroom_enc(self):
        enc = np.zeros(MUSHROOM_ENC_DIM, dtype=np.float32)
        if not self.field.mushrooms:
            return enc
        mpos = np.array([[m.pos.x, m.pos.y]
                         for m in self.field.mushrooms],
                        dtype=np.float32)
        rad = math.radians(self.angle)
        fx, fy = math.cos(rad), math.sin(rad)
        dx = mpos[:, 0] - self.pos.x
        dy = mpos[:, 1] - self.pos.y
        dist = np.sqrt(dx*dx + dy*dy)
        dot  = fx * dx + fy * dy
        mask = (dist < FOCUS_RANGE) & (dot > 0)
        if not mask.any():
            return enc
        dists_masked = np.where(mask, dist, np.inf)
        best_i = int(np.argmin(dists_masked))
        best_m = self.field.mushrooms[best_i]
        bmap = {'W': 0, 'G': 1, 'M': 2}
        enc[bmap[best_m.biome]] = 1.0
        enc[3] = 1.0 if best_m.grade == 'premium' else 0.0
        enc[4] = 1.0 if best_m.variant == 2 else 0.0
        enc[5] = 1.0 if best_m.is_rotten else 0.0
        return enc

    def _cast_rays(self):
        n_rays = BRUTE_OBS_DIM - MUSHROOM_ENC_DIM - 2  # = 11 - 6 - 2 = 3
        if not self.field.mushrooms:
            return [0.0] * n_rays
        mpos = np.array([[m.pos.x, m.pos.y]
                         for m in self.field.mushrooms],
                        dtype=np.float32)
        dx = mpos[:, 0] - self.pos.x
        dy = mpos[:, 1] - self.pos.y
        dist = np.sqrt(dx*dx + dy*dy)
        rays = []
        for i in range(n_rays):
            a = self.angle + (i - n_rays // 2) * (
                VISION_ANGLE_DEG / max(1, n_rays - 1))
            rad = math.radians(a)
            fx, fy = math.cos(rad), math.sin(rad)
            dot  = fx * dx + fy * dy
            mask = (dist < VISION_RANGE) & (dot > 0)
            if mask.any():
                best = float(np.max(
                    np.where(mask, 1.0 - dist / VISION_RANGE, 0.0)))
            else:
                best = 0.0
            rays.append(best)
        return rays


class GeneticAlgorithm:
    def __init__(self, pop_size=GA_POP_SIZE, seed=0):
        self.pop_size   = pop_size
        self.generation = 0
        self.rng        = np.random.default_rng(seed)
        self.population = [GAGenome(np.random.default_rng(seed + i))
                           for i in range(pop_size)]
        self.mut_rate   = MUTATE_RATE_INIT
        self.mut_std    = MUTATE_STD_INIT
        self.species    = []
        self.best_fitness_history = []
        self.avg_fitness_history  = []
        self.species_count_history = []
        self._mut_rate  = self.mut_rate
        self._mut_std   = self.mut_std

    def evolve(self):
        fits = np.array([g.fitness for g in self.population])
        self.best_fitness_history.append(float(fits.max()))
        self.avg_fitness_history.append(float(fits.mean()))
        self.species_count_history.append(len(self.species))
        prev_best = self.best_fitness_history[-2] if len(self.best_fitness_history) > 1 else -1e9
        if fits.max() > prev_best:
            self.mut_rate = max(0.05, self.mut_rate * 0.95)
        else:
            self.mut_rate = min(0.5,  self.mut_rate * 1.15)
        self._mut_rate = self.mut_rate
        idx = np.argsort(fits)[::-1]
        elites = [copy.deepcopy(self.population[i]) for i in idx[:ELITE_SIZE]]
        new_pop = list(elites)
        while len(new_pop) < self.pop_size:
            p1, p2 = self.rng.choice(idx[:max(ELITE_SIZE*3, 10)], 2, replace=False)
            child  = self._crossover(self.population[p1], self.population[p2])
            self._mutate(child)
            new_pop.append(child)
        self.population = new_pop
        self.generation += 1

    def _crossover(self, g1, g2):
        child = GAGenome()
        f1, f2 = g1.flat(), g2.flat()
        mask = self.rng.random(len(f1)) < 0.5
        cf   = np.where(mask, f1, f2)
        child.load_flat(cf)
        return child

    def _mutate(self, g):
        f    = g.flat()
        mask = self.rng.random(len(f)) < self.mut_rate
        f[mask] += self.rng.normal(0, self.mut_std, mask.sum())
        g.load_flat(f)

    def get_best(self):
        return max(self.population, key=lambda g: g.fitness)

    def get_stats(self):
        fits = [g.fitness for g in self.population]
        return {
            'generation': self.generation,
            'best':       max(fits) if fits else 0.0,
            'avg':        sum(fits) / len(fits) if fits else 0.0,
            'worst':      min(fits) if fits else 0.0,
            'species':    len(self.species),
            'mut_rate':   self.mut_rate,
            'mut_std':    self.mut_std,
        }

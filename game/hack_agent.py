"""
hack_agent.py — ハックモード（H キー）エージェント

設計:
  - GAGenome を deep copy して使用（学習データは変更しない）
  - SAGE・ボトルネック通信は完全に通常通り動作
  - BRUTE の NN 計算も毎フレーム実行（パルス送信・記憶 GRU 更新継続）
  - BRUTE の行動出力（accel/steer/brake）だけプレイヤーが上書き
  - キーボード入力がない場合は BRUTE の出力をそのまま使用
"""
import math
import copy
import pygame
import numpy as np
from config import (
    CAR_ACCEL, CAR_MAX_SPEED, CAR_FRICTION, CAR_TURN_SPEED,
    ENERGY_DECAY, ENERGY_GOAL, FOOD_RADIUS, GOAL_RADIUS,
    FIELD_SIZE, INIT_ENERGY, MAX_ENERGY, REWARD_SURVIVE,
    REWARD_MOVE, REWARD_GOAL_SB,
)
from game.bottleneck import Bottleneck


class HackAgent:
    """
    ハックモード用エージェント。
    SageBruteAgent と同じ物理・NN 処理を持つが、
    BRUTE の行動出力をキーボードで上書きできる。
    """

    def __init__(self, genome, field, start_pos):
        # GAGenome を deep copy して使用（学習データは変更しない）
        self.genome = copy.deepcopy(genome)
        self.field  = field
        self.pos    = pygame.Vector2(*start_pos)
        self.angle  = 0.0
        self.speed  = 0.0
        self.energy = INIT_ENERGY
        self.alive  = True
        self.total_reward   = 0.0
        self.intake_history = []
        self.toxic_count    = 0
        self.food_collected = 0
        self.bn = Bottleneck(self.genome.sage, self.genome.brute)
        self.bn.reset_episode()

        # プレイヤー入力（毎フレーム pygame.key.get_pressed() で更新）
        self._player_accel = 0.0
        self._player_steer = 0.5
        self._player_brake = 0.0
        self._player_active = False   # True のとき BRUTE 出力を上書き

    # ------------------------------------------------------------------
    def update_keys(self):
        """キー状態を読み取る。毎フレーム step() の前に呼ぶ。"""
        keys = pygame.key.get_pressed()
        accel = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP])    else 0.0
        brake = 1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN])  else 0.0
        if   keys[pygame.K_a] or keys[pygame.K_LEFT]:
            steer = 0.0
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            steer = 1.0
        else:
            steer = 0.5

        self._player_active = (accel > 0 or brake > 0 or steer != 0.5)
        self._player_accel  = accel
        self._player_steer  = steer
        self._player_brake  = brake

    # ------------------------------------------------------------------
    def step(self):
        """
        1フレーム処理。
        SAGE・BRUTE の NN 計算は通常通り実行。
        行動出力はキー入力があれば上書きする。
        """
        from game.ga_agent import SageBruteAgent
        # obs 取得（SageBruteAgent と同じロジック）
        obs_sage  = self._get_obs_sage()
        obs_brute = self._get_obs_brute()

        # ボトルネック経由で両 NN を実行（パルス・記憶 GRU 更新）
        nn_action = self.bn.step(obs_sage, obs_brute)

        # 行動決定: キー入力があれば上書き、なければ BRUTE 出力
        if self._player_active:
            accel = self._player_accel
            steer = self._player_steer
            brake = self._player_brake
        else:
            accel = float(nn_action[0])
            steer = float(nn_action[1])
            brake = float(nn_action[2])

        # 物理更新
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

        # キノコ衝突
        for m in self.field.mushrooms[:]:
            if (self.pos - m.pos).length() < FOOD_RADIUS:
                self._eat_mushroom(m)
                self.field.mushrooms.remove(m)
                self.food_collected += 1

        # ゴール判定
        goal_dist = (self.pos - pygame.Vector2(*self.field.goal_pos)).length()
        self.dist_to_goal = goal_dist
        if goal_dist < GOAL_RADIUS:
            self.energy = min(MAX_ENERGY, self.energy + ENERGY_GOAL)
            self.intake_history = []
            self.toxic_count = 0

        reward = REWARD_SURVIVE
        if self.speed > 0.1:
            reward += REWARD_MOVE * self.speed
        self.total_reward += reward

        done = self.energy <= 0
        if done:
            self.alive = False
        return {'done': done, 'reward': reward}

    # ------------------------------------------------------------------
    def _eat_mushroom(self, m):
        """SageBruteAgent._eat_mushroom と同じロジック。"""
        from config import (ENERGY_ROTTEN, ENERGY_NORMAL, MUSHROOM_SPECIES,
                            TOXIC_COUNT, ENERGY_TOXIC, HISTORY_LEN)
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

    # ------------------------------------------------------------------
    def _get_obs_sage(self):
        """SAGE 用 obs（SageBruteAgent と同じ）。"""
        from game.ga_agent import SageBruteAgent
        # 一時的に SageBruteAgent のメソッドを借用
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
        """BRUTE 用 obs（SageBruteAgent と同じ）。"""
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
        from config import MUSHROOM_ENC_DIM, FOCUS_RANGE
        enc = np.zeros(MUSHROOM_ENC_DIM, dtype=np.float32)
        if not self.field.mushrooms:
            return enc
        mpos = np.array([[m.pos.x, m.pos.y] for m in self.field.mushrooms],
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
        best_i = int(np.argmin(np.where(mask, dist, np.inf)))
        best_m = self.field.mushrooms[best_i]
        bmap = {'W': 0, 'G': 1, 'M': 2}
        enc[bmap[best_m.biome]] = 1.0
        enc[3] = 1.0 if best_m.grade == 'premium' else 0.0
        enc[4] = 1.0 if best_m.variant == 2 else 0.0
        enc[5] = 1.0 if best_m.is_rotten else 0.0
        return enc

    def _cast_rays(self):
        from config import BRUTE_OBS_DIM, MUSHROOM_ENC_DIM, VISION_RANGE, VISION_ANGLE_DEG
        n_rays = BRUTE_OBS_DIM - MUSHROOM_ENC_DIM - 2
        if not self.field.mushrooms:
            return [0.0] * n_rays
        mpos = np.array([[m.pos.x, m.pos.y] for m in self.field.mushrooms],
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
                best = float(np.max(np.where(mask, 1.0 - dist / VISION_RANGE, 0.0)))
            else:
                best = 0.0
            rays.append(best)
        return rays

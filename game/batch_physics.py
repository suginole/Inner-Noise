"""
batch_physics.py — 全個体の物理計算をnumpy一括処理（モード3専用）
"""
import numpy as np
import pygame
from config import (
    CAR_ACCEL, CAR_MAX_SPEED, CAR_FRICTION, CAR_TURN_SPEED,
    ENERGY_DECAY, ENERGY_GOAL, FOOD_RADIUS, GOAL_RADIUS,
    FIELD_SIZE, REWARD_SURVIVE, REWARD_MOVE, REWARD_GOAL_SB,
    MAX_ENERGY,
)


class BatchPhysics:
    """
    全個体の位置・角度・速度をnumpy配列で一括管理する。
    モード3専用。モード2では使用しない。
    """

    def __init__(self, agents: list):
        self.agents = agents
        self.pos_x  = np.array([ag.pos.x        for ag in agents], dtype=np.float32)
        self.pos_y  = np.array([ag.pos.y        for ag in agents], dtype=np.float32)
        self.angle  = np.array([ag.angle        for ag in agents], dtype=np.float32)
        self.speed  = np.array([ag.speed        for ag in agents], dtype=np.float32)
        self.energy = np.array([ag.energy       for ag in agents], dtype=np.float32)
        self.alive  = np.array([ag.alive        for ag in agents], dtype=bool)
        self.reward = np.array([ag.total_reward for ag in agents], dtype=np.float32)

        # キノコ位置配列（衝突判定に使用）
        self._build_mushroom_arrays(agents[0].field)

    def _build_mushroom_arrays(self, field):
        """フィールドのキノコ位置をnumpy配列に変換する。"""
        if field.mushrooms:
            self.mush_x = np.array([m.pos.x for m in field.mushrooms],
                                    dtype=np.float32)
            self.mush_y = np.array([m.pos.y for m in field.mushrooms],
                                    dtype=np.float32)
        else:
            self.mush_x = np.array([], dtype=np.float32)
            self.mush_y = np.array([], dtype=np.float32)

    def apply_actions(self, actions_np: np.ndarray):
        """
        全個体の物理を一括更新する。
        actions_np: (pop_size, 3) numpy array [accel, steer, brake]
        """
        alive = self.alive
        accel = actions_np[:, 0]
        steer = actions_np[:, 1]
        brake = actions_np[:, 2]

        # 速度更新
        self.speed[alive] += (accel[alive] - brake[alive]) * CAR_ACCEL
        self.speed = np.clip(self.speed, 0, CAR_MAX_SPEED)
        self.speed[alive] *= (1 - CAR_FRICTION)

        # 角度更新
        self.angle[alive] += (steer[alive] - 0.5) * CAR_TURN_SPEED

        # 位置更新
        rad = np.radians(self.angle)
        self.pos_x[alive] += np.cos(rad[alive]) * self.speed[alive]
        self.pos_y[alive] += np.sin(rad[alive]) * self.speed[alive]
        self.pos_x = np.clip(self.pos_x, 0, FIELD_SIZE - 1)
        self.pos_y = np.clip(self.pos_y, 0, FIELD_SIZE - 1)

        # エネルギー減少
        self.energy[alive] -= ENERGY_DECAY
        self.energy[alive] -= self.speed[alive] * 0.001

        # 報酬
        self.reward[alive] += REWARD_SURVIVE
        self.reward[alive] += REWARD_MOVE * np.maximum(0, self.speed[alive])

        # 死亡判定
        newly_dead = alive & (self.energy <= 0)
        self.alive[newly_dead] = False

    def check_mushrooms(self, agents: list):
        """
        全個体とキノコの衝突判定をnumpy一括処理する。
        """
        if len(self.mush_x) == 0:
            return

        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) == 0:
            return

        # alive個体の位置
        ax = self.pos_x[alive_idx][:, np.newaxis]  # (alive, 1)
        ay = self.pos_y[alive_idx][:, np.newaxis]

        # キノコとの距離行列 (alive, mushrooms)
        dx = ax - self.mush_x[np.newaxis, :]
        dy = ay - self.mush_y[np.newaxis, :]
        dist = np.sqrt(dx * dx + dy * dy)

        # 衝突判定
        hit = dist < FOOD_RADIUS   # (alive, mushrooms)
        hit_agent_idx, hit_mush_idx = np.where(hit)

        eaten = set()
        for ai, mi in zip(hit_agent_idx, hit_mush_idx):
            if mi in eaten:
                continue
            eaten.add(mi)
            agent_i = alive_idx[ai]
            if mi < len(agents[agent_i].field.mushrooms):
                agents[agent_i]._eat_mushroom(
                    agents[agent_i].field.mushrooms[mi])

        # 食べたキノコを後ろから削除
        for mi in sorted(eaten, reverse=True):
            if mi < len(agents[0].field.mushrooms):
                agents[0].field.mushrooms.pop(mi)

        # キノコ配列を再構築
        if eaten:
            self._build_mushroom_arrays(agents[0].field)

    def check_goals(self, agents: list, goal_pos) -> int:
        """ゴール到達判定。到達数を返す。"""
        gx, gy = goal_pos
        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) == 0:
            return 0

        dx = self.pos_x[alive_idx] - gx
        dy = self.pos_y[alive_idx] - gy
        dist = np.sqrt(dx * dx + dy * dy)
        goal_hit = dist < GOAL_RADIUS
        count = int(goal_hit.sum())

        for ai in alive_idx[goal_hit]:
            self.energy[ai] = min(MAX_ENERGY, self.energy[ai] + ENERGY_GOAL)
            self.reward[ai] += REWARD_GOAL_SB
            agents[ai].intake_history = []
            agents[ai].toxic_count = 0

        return count

    def sync_to_agents(self, agents: list):
        """numpy配列の状態をエージェントオブジェクトに書き戻す。"""
        for i, ag in enumerate(agents):
            ag.pos.x        = float(self.pos_x[i])
            ag.pos.y        = float(self.pos_y[i])
            ag.angle        = float(self.angle[i])
            ag.speed        = float(self.speed[i])
            ag.energy       = float(self.energy[i])
            ag.alive        = bool(self.alive[i])
            ag.total_reward = float(self.reward[i])
            if not ag.alive:
                ag.genome.fitness = ag.total_reward

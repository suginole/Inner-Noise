"""
batch_physics.py — 全個体の物理計算をnumpy一括処理（モード3専用）

設計方針:
  - GPU モードでは全エージェントが同一の shared_field を参照する
  - check_mushrooms は Python ループを完全排除し numpy のみで処理する
  - エネルギー変化も numpy 配列で直接管理し _eat_mushroom は呼ばない
"""
import math
import numpy as np
import pygame
from config import (
    CAR_ACCEL, CAR_MAX_SPEED, CAR_FRICTION, CAR_TURN_SPEED,
    ENERGY_DECAY, ENERGY_GOAL, ENERGY_NORMAL, ENERGY_PREMIUM,
    ENERGY_ROTTEN, ENERGY_TOXIC, FOOD_RADIUS, GOAL_RADIUS,
    FIELD_SIZE, REWARD_SURVIVE, REWARD_MOVE, REWARD_GOAL_SB,
    MAX_ENERGY, TOXIC_COUNT, HISTORY_LEN,
    SAGE_OBS_DIM, BRUTE_OBS_DIM, MUSHROOM_ENC_DIM,
    FOCUS_RANGE, VISION_RANGE, VISION_ANGLE_DEG,
)


class BatchPhysics:
    """
    全個体の位置・角度・速度をnumpy配列で一括管理する。
    モード3専用。モード2では使用しない。

    重要: GPU モードでは全エージェントが同一の shared_field を参照すること。
    check_mushrooms は agents[0].field のキノコリストのみを使用する。
    """

    def __init__(self, agents: list):
        self.agents = agents
        n = len(agents)
        self.pos_x  = np.array([ag.pos.x        for ag in agents], dtype=np.float32)
        self.pos_y  = np.array([ag.pos.y        for ag in agents], dtype=np.float32)
        self.angle  = np.array([ag.angle        for ag in agents], dtype=np.float32)
        self.speed  = np.array([ag.speed        for ag in agents], dtype=np.float32)
        self.energy = np.array([ag.energy       for ag in agents], dtype=np.float32)
        self.alive  = np.array([ag.alive        for ag in agents], dtype=bool)
        self.reward = np.array([ag.total_reward for ag in agents], dtype=np.float32)

        # キノコ位置配列（衝突判定に使用）
        # NOTE: agents[0].field のキノコを共有フィールドとして使用する
        self._build_mushroom_arrays(agents[0].field)

    def _build_mushroom_arrays(self, field):
        """フィールドのキノコ位置・属性をnumpy配列に変換する。"""
        mushrooms = field.mushrooms
        if mushrooms:
            self.mush_x       = np.array([m.pos.x for m in mushrooms], dtype=np.float32)
            self.mush_y       = np.array([m.pos.y for m in mushrooms], dtype=np.float32)
            # エネルギー変化量を事前計算（is_rotten / grade で決定）
            self.mush_energy  = np.array([
                -40.0 if m.is_rotten else (20.0 if m.grade == 'premium' else 8.0)
                for m in mushrooms
            ], dtype=np.float32)
        else:
            self.mush_x      = np.array([], dtype=np.float32)
            self.mush_y      = np.array([], dtype=np.float32)
            self.mush_energy = np.array([], dtype=np.float32)

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
        Python ループなし。エネルギー変化も numpy で直接適用。
        agents[0].field のキノコリストを共有フィールドとして使用する。
        """
        if len(self.mush_x) == 0:
            return

        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) == 0:
            return

        # alive個体の位置 (alive, 1)
        ax = self.pos_x[alive_idx][:, np.newaxis]
        ay = self.pos_y[alive_idx][:, np.newaxis]

        # キノコとの距離行列 (alive, mushrooms)
        dx = ax - self.mush_x[np.newaxis, :]
        dy = ay - self.mush_y[np.newaxis, :]
        dist2 = dx * dx + dy * dy
        hit = dist2 < (FOOD_RADIUS * FOOD_RADIUS)   # (alive, mushrooms)

        if not hit.any():
            return

        # 各キノコを最初に当たったエージェントに割り当て（先着優先）
        # hit の列方向 (mushrooms) で最初の True の行 (agent) を取得
        mush_hit_any = hit.any(axis=0)                          # (mushrooms,)
        eaten_mush_idx = np.where(mush_hit_any)[0]              # 食べられるキノコのインデックス

        if len(eaten_mush_idx) == 0:
            return

        # 各食べられるキノコについて、最初に当たったエージェントを特定
        first_agent_row = np.argmax(hit[:, eaten_mush_idx], axis=0)  # (n_eaten,)
        first_agent_idx = alive_idx[first_agent_row]                  # 実際のエージェントインデックス

        # エネルギー変化を numpy で直接適用（Python ループなし）
        energy_gains = self.mush_energy[eaten_mush_idx]   # (n_eaten,)
        np.add.at(self.energy, first_agent_idx, energy_gains)
        self.energy = np.clip(self.energy, 0, MAX_ENERGY)

        # エージェントオブジェクトの intake_history も更新（toxic チェック用）
        field = agents[0].field
        for k, mi in enumerate(eaten_mush_idx):
            if mi >= len(field.mushrooms):
                continue
            m = field.mushrooms[mi]
            ai = int(first_agent_idx[k])
            if not m.is_rotten:
                sk = (m.biome, m.grade, m.variant)
                agents[ai].intake_history.append(sk)
                if len(agents[ai].intake_history) > HISTORY_LEN:
                    agents[ai].intake_history.pop(0)

        # 食べたキノコを後ろから削除（共有フィールドから）
        for mi in sorted(eaten_mush_idx.tolist(), reverse=True):
            if mi < len(field.mushrooms):
                field.mushrooms.pop(mi)

        # キノコ配列を再構築
        self._build_mushroom_arrays(field)

    def check_goals(self, agents: list, goal_pos) -> int:
        """ゴール到達判定。到達数を返す。"""
        gx, gy = goal_pos
        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) == 0:
            return 0

        dx = self.pos_x[alive_idx] - gx
        dy = self.pos_y[alive_idx] - gy
        dist2 = dx * dx + dy * dy
        goal_hit = dist2 < (GOAL_RADIUS * GOAL_RADIUS)
        count = int(goal_hit.sum())

        if count > 0:
            hit_agent_idx = alive_idx[goal_hit]
            self.energy[hit_agent_idx] = np.minimum(
                MAX_ENERGY, self.energy[hit_agent_idx] + ENERGY_GOAL)
            self.reward[hit_agent_idx] += REWARD_GOAL_SB
            for ai in hit_agent_idx:
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


# ================================================================
def collect_obs_batch(agents: list, field) -> tuple:
    """
    全エージェントの obs を一括収集する（numpy 一括・Python ループ最小化）。

    戻り値:
        obs_sage  (n, SAGE_OBS_DIM)  numpy float32
        obs_brute (n, BRUTE_OBS_DIM) numpy float32
    [9:11] は受信パルス用プレースホルダ（0.0）。bottleneck が後で上書きする。
    """
    n = len(agents)
    obs_sage  = np.zeros((n, SAGE_OBS_DIM),  dtype=np.float32)
    obs_brute = np.zeros((n, BRUTE_OBS_DIM), dtype=np.float32)

    alive = np.array([ag.alive for ag in agents], dtype=bool)
    if not alive.any() or not field.mushrooms:
        return obs_sage, obs_brute

    # 全エージェントの状態を一括取得
    pos_x  = np.array([ag.pos.x  for ag in agents], dtype=np.float32)
    pos_y  = np.array([ag.pos.y  for ag in agents], dtype=np.float32)
    angle  = np.array([ag.angle  for ag in agents], dtype=np.float32)
    energy = np.array([ag.energy for ag in agents], dtype=np.float32)

    # キノコ属性を一括取得
    mpos = np.array([[m.pos.x, m.pos.y]
                     for m in field.mushrooms], dtype=np.float32)
    mis_rotten  = np.array([m.is_rotten         for m in field.mushrooms], dtype=bool)
    mis_premium = np.array([m.grade == 'premium' for m in field.mushrooms], dtype=bool)
    mis_variant = np.array([m.variant == 2       for m in field.mushrooms], dtype=np.float32)
    biome_map   = {'W': 0, 'G': 1, 'M': 2}
    mis_biome   = np.array([biome_map[m.biome]   for m in field.mushrooms], dtype=np.int32)

    # ゴール位置
    gx, gy = field.goal_pos

    # 前方方向ベクトル（全エージェント一括）
    rad = np.radians(angle)
    fx  = np.cos(rad)   # (n,)
    fy  = np.sin(rad)

    # 視覚レイ数（BRUTE_OBS_DIM - MUSHROOM_ENC_DIM - 2 = 3）
    n_rays = BRUTE_OBS_DIM - MUSHROOM_ENC_DIM - 2

    alive_idx = np.where(alive)[0]

    for i in alive_idx:
        # キノコとの距離・dot（一括）
        dx   = mpos[:, 0] - pos_x[i]
        dy   = mpos[:, 1] - pos_y[i]
        dist = np.sqrt(dx * dx + dy * dy)
        dot  = fx[i] * dx + fy[i] * dy

        # 弁別視野（前方最近傍キノコ）
        mask = (dist < FOCUS_RANGE) & (dot > 0)
        enc  = np.zeros(MUSHROOM_ENC_DIM, dtype=np.float32)
        if mask.any():
            best_i = int(np.argmin(np.where(mask, dist, np.inf)))
            enc[mis_biome[best_i]] = 1.0
            enc[3] = float(mis_premium[best_i])
            enc[4] = float(mis_variant[best_i])
            enc[5] = float(mis_rotten[best_i])

        # SAGE enc マスク（バイオームと腐敗を隠す）
        sage_enc = enc.copy()
        sage_enc[0:3] = 0.0
        sage_enc[5]   = 0.0

        # BRUTE enc マスク（栄養価・バリアントを隠す）
        brute_enc = enc.copy()
        brute_enc[3] = 0.0
        brute_enc[4] = 0.0

        # ゴール情報（SAGE 用）
        goal_dx   = gx - pos_x[i]
        goal_dy   = gy - pos_y[i]
        goal_dist = math.hypot(goal_dx, goal_dy)
        goal_ang  = math.atan2(goal_dy, goal_dx) - angle[i]
        goal_ang  = math.atan2(math.sin(goal_ang),
                               math.cos(goal_ang)) / math.pi

        obs_sage[i, :6] = sage_enc
        obs_sage[i, 6]  = float(goal_ang)
        obs_sage[i, 7]  = min(1.0, goal_dist / FIELD_SIZE)
        obs_sage[i, 8]  = energy[i] / MAX_ENERGY
        # [9:11] = 0.0（受信パルス・bottleneck が後で上書き）

        # 視覚レイ（BRUTE 用・n_rays=3 本）
        rays = np.zeros(n_rays, dtype=np.float32)
        for ri in range(n_rays):
            a   = angle[i] + (ri - n_rays // 2) * (
                  VISION_ANGLE_DEG / max(1, n_rays - 1))
            rfx = math.cos(math.radians(a))
            rfy = math.sin(math.radians(a))
            rdot  = rfx * dx + rfy * dy
            rmask = (dist < VISION_RANGE) & (rdot > 0)
            if rmask.any():
                rays[ri] = float(np.max(
                    np.where(rmask, 1.0 - dist / VISION_RANGE, 0.0)))

        obs_brute[i, :6]       = brute_enc
        obs_brute[i, 6:6+n_rays] = rays
        # [6+n_rays:11] = 0.0（受信パルス・bottleneck が後で上書き）

    return obs_sage, obs_brute

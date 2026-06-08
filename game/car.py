"""
car.py — 車の物理シミュレーション
アクセル・ハンドル・ブレーキの3入力を受け取り、
位置・速度・エネルギーを更新する。
"""
import math
import pygame
from config import *


def _angle_diff(a: float, b: float) -> float:
    """2つの角度（ラジアン）の差を -π～π に正規化する。"""
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi


class Car:
    """
    運動系が操作する車。
    自身は空腹やダメージを「感知しない」（感覚系が監視する）。
    """

    def __init__(self, x: float, y: float, angle: float = 0.0):
        self.pos    = pygame.Vector2(x, y)
        self.angle  = angle          # 度（0=右向き）
        self.speed  = 0.0
        self.energy = ENERGY_MAX
        self.alive  = True
        self.food_collected = 0
        self.dist_to_goal   = 0.0
        self.prev_dist_goal = 0.0
        self._idle_frames   = 0

    # ----------------------------------------------------------------
    def step(self, accel: float, steer: float, brake: float,
             field) -> dict:
        """
        1フレーム分の物理更新。
        accel, steer, brake: 0〜1 の連続値
        field: Field オブジェクト
        Returns: reward_components dict
        """
        if not self.alive:
            return {"reward": 0.0, "done": True}

        reward = 0.0
        done   = False

        # ---- ステアリング ----
        steer_input = (steer - 0.5) * 2.0   # -1〜+1
        self.angle += steer_input * CAR_TURN_SPEED

        # ---- 加速 / ブレーキ ----
        rad = math.radians(self.angle)
        dx  = math.cos(rad)
        dy  = math.sin(rad)

        # 地形の勾配
        gx, gy = field.gradient_at(self.pos.x, self.pos.y)
        slope_along = dx * gx + dy * gy   # 進行方向の勾配成分

        # 加速
        self.speed += accel * CAR_ACCEL
        # ブレーキ
        self.speed -= brake * CAR_BRAKE
        # 摩擦
        self.speed -= self.speed * CAR_FRICTION
        # 登坂抵抗
        if slope_along > 0:
            self.speed -= slope_along * CAR_SLOPE_DRAG
        self.speed = max(-CAR_MAX_SPEED * 0.3, min(CAR_MAX_SPEED, self.speed))

        # ---- 位置更新 ----
        self.pos.x += dx * self.speed
        self.pos.y += dy * self.speed

        # ワールド境界クリップ
        self.pos.x = max(10, min(WORLD_W - 10, self.pos.x))
        self.pos.y = max(10, min(WORLD_H - 10, self.pos.y))

        # ---- 落下ダメージ ----
        slope_mag = math.sqrt(gx*gx + gy*gy)
        if slope_mag > SLOPE_DAMAGE_THRESH and self.speed < -0.5:
            self.energy -= CAR_FALL_DAMAGE
            reward += PENALTY_FALL

        # ---- エネルギー消費 ----
        decay = ENERGY_DECAY_BASE
        # 登坂消費: 勾配強度に比例して急激に増加
        if slope_along > 0.0:
            slope_mag = math.sqrt(gx * gx + gy * gy)
            decay += ENERGY_DECAY_CLIMB * slope_along * (1.0 + slope_mag * 4.0)
        if abs(self.speed) < IDLE_SPEED_THRESH:
            self._idle_frames += 1
            if self._idle_frames > 30:
                decay += ENERGY_DECAY_IDLE
        else:
            self._idle_frames = 0
        self.energy -= decay
        self.energy  = max(0.0, min(ENERGY_MAX, self.energy))

        # ---- 餌の回収 ----
        collected, is_premium = field.collect_food(self.pos.x, self.pos.y)
        if collected:
            gain = ENERGY_PER_FOOD_HI if is_premium else ENERGY_PER_FOOD
            rwd  = REWARD_FOOD_HI    if is_premium else REWARD_FOOD
            self.energy = min(ENERGY_MAX, self.energy + gain)
            self.food_collected += 1
            reward += rwd
            field.respawn_food()

        # ---- ゴール判定 ----
        gvec = pygame.Vector2(field.goal_pos)
        self.dist_to_goal = self.pos.distance_to(gvec)
        reward += (self.prev_dist_goal - self.dist_to_goal) * REWARD_GOAL_STEP
        self.prev_dist_goal = self.dist_to_goal

        if self.dist_to_goal < GOAL_RADIUS:
            reward += REWARD_GOAL
            done = True

        # ---- 餓死判定 ----
        if self.energy <= 0.0:
            reward += PENALTY_DEATH
            done = True
            self.alive = False

        return {"reward": reward, "done": done}

    # ----------------------------------------------------------------
    def get_observation(self, field) -> list[float]:
        """
        感覚系への観測ベクトル o_k ∈ R^(6 + VISION_RAYS)

        基本6次元:
          [energy, goal_angle, grad_x, grad_y, food_dx, food_dy]

        視野レイ VISION_RAYS次元:
          進行方向に対して ±45度の範囲を VISION_RAYS本のレイでスキャン。
          各レイにつき、VISION_RANGE以内に餌があれば (1 - dist/range)、なければ 0.0。
        """
        # ゴールへの相対角度
        gvec = pygame.Vector2(field.goal_pos) - self.pos
        goal_angle = math.atan2(gvec.y, gvec.x) - math.radians(self.angle)
        goal_angle = (goal_angle + math.pi) % (2 * math.pi) - math.pi

        # 勾配
        gx, gy = field.gradient_at(self.pos.x, self.pos.y)

        # 最近傍の餌ベクトル（全方向・画面内スケール）
        food_dx, food_dy = 0.0, 0.0
        if field.foods:
            nearest_pos = min(field.foods, key=lambda f: self.pos.distance_to(f[0]))[0]
            fvec = nearest_pos - self.pos
            dist = fvec.length()
            if dist > 0:
                food_dx = max(-1.0, min(1.0, fvec.x / (SCREEN_W * 0.5)))
                food_dy = max(-1.0, min(1.0, fvec.y / (SCREEN_H * 0.5)))

        base_obs = [
            float(self.energy),
            float(goal_angle / math.pi),
            float(gx * 10.0),
            float(gy * 10.0),
            float(food_dx),
            float(food_dy),
        ]

        # ---- 視野レイ（餌探索用） ----
        ray_obs = []
        facing_rad = math.radians(self.angle)
        half_fov   = math.radians(VISION_ANGLE_DEG)
        n_rays     = VISION_RAYS
        vision_range_sq = VISION_RANGE * VISION_RANGE  # 平方距離で事前フィルタ

        # 視野内の餌だけ事前フィルタ（平方距離で高速化）
        nearby_foods = []
        px, py = self.pos.x, self.pos.y
        for food_pos, _ in field.foods:   # (Vector2, is_premium)
            dx = food_pos.x - px
            dy = food_pos.y - py
            if dx * dx + dy * dy <= vision_range_sq:
                nearby_foods.append((food_pos, dx, dy, math.atan2(dy, dx)))

        ray_half_width = (half_fov / max(1, n_rays - 1) + 0.15) if n_rays > 1 else half_fov

        for i in range(n_rays):
            t = i / (n_rays - 1) if n_rays > 1 else 0.5
            ray_angle = facing_rad + half_fov * (2 * t - 1)

            best_signal = 0.0
            for food, dx, dy, food_angle in nearby_foods:
                diff = abs(_angle_diff(food_angle, ray_angle))
                if diff < ray_half_width:
                    dist = math.sqrt(dx * dx + dy * dy)
                    signal = 1.0 - dist / VISION_RANGE
                    if signal > best_signal:
                        best_signal = signal
            ray_obs.append(best_signal)

        return base_obs + ray_obs

    # ----------------------------------------------------------------
    def reset(self, x: float, y: float, angle: float = 0.0):
        self.pos    = pygame.Vector2(x, y)
        self.angle  = angle
        self.speed  = 0.0
        self.energy = ENERGY_MAX
        self.alive  = True
        self.food_collected = 0
        self.dist_to_goal   = 0.0
        self.prev_dist_goal = 0.0
        self._idle_frames   = 0

    # ----------------------------------------------------------------
    def draw(self, surface: pygame.Surface, cam_offset: pygame.Vector2,
             color=C_CAR):
        sx = int(self.pos.x - cam_offset.x)
        sy = int(self.pos.y - cam_offset.y)
        rad = math.radians(self.angle)
        # 車体（三角形）
        front = pygame.Vector2(math.cos(rad), math.sin(rad)) * 14
        left  = pygame.Vector2(math.cos(rad + 2.4), math.sin(rad + 2.4)) * 9
        right = pygame.Vector2(math.cos(rad - 2.4), math.sin(rad - 2.4)) * 9
        pts = [
            (sx + front.x, sy + front.y),
            (sx + left.x,  sy + left.y),
            (sx + right.x, sy + right.y),
        ]
        pygame.draw.polygon(surface, color, pts)
        pygame.draw.polygon(surface, C_WHITE, pts, 1)

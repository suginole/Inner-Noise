"""
player_agent.py — キーボード操作エージェント
W/S/A/D または矢印キーで車を操作する。
"""
import pygame
from game.agent import Agent
from game.car import Car
from game.field import Field


class PlayerAgent(Agent):
    """プレイヤーがキーボードで直接操作するエージェント。"""

    def __init__(self, car: Car, field: Field):
        super().__init__(car, field)

    def act(self) -> tuple[float, float, float]:
        keys = pygame.key.get_pressed()

        accel = 0.0
        steer = 0.5   # 中立
        brake = 0.0

        # 前進
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            accel = 1.0
        # 後退
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            brake = 1.0

        # ステアリング
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            steer = 0.0   # 左
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            steer = 1.0   # 右

        return accel, steer, brake

"""
agent.py — エージェント基底クラス
プレイヤー・GA・将来のRNNエージェントの共通インターフェース。
"""
from abc import ABC, abstractmethod
from game.car import Car
from game.field import Field


class Agent(ABC):
    """全エージェントの基底クラス。"""

    def __init__(self, car: Car, field: Field):
        self.car   = car
        self.field = field
        self.total_reward = 0.0
        self.frame_count  = 0

    @abstractmethod
    def act(self) -> tuple[float, float, float]:
        """
        行動を決定して (accel, steer, brake) を返す。
        各値は 0〜1 の連続値。
        """
        ...

    def step(self) -> dict:
        """
        1フレーム分の更新。
        Returns: reward_components dict
        """
        accel, steer, brake = self.act()
        result = self.car.step(accel, steer, brake, self.field)
        self.total_reward += result["reward"]
        self.frame_count  += 1
        return result

    def reset(self):
        self.total_reward = 0.0
        self.frame_count  = 0
        self.car.reset(*self.field.start_pos)

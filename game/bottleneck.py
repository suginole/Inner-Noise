"""
bottleneck.py — ボトルネック通信路
5Hz / 4bits の同期パルス通信路。
現在はスタブ（パススルー）実装。
将来: 感覚の海馬RNN → パルス量子化 → 運動の海馬RNN に差し替える。

【インターフェース仕様】
  push(obs: list[float]) -> None
      感覚系から観測ベクトルを受け取る（傾聴ターン中に毎フレーム呼ぶ）

  tick(dt: float) -> tuple[list[int] | None, str]
      フレームごとに呼ぶ。
      Returns:
        pulse: list[int] shape=(4,) or None（パルスが発火しないフレーム）
        mode:  "listen" | "speak"

  get_action() -> list[float]
      運動系が使う行動指令 [accel, steer, brake] を返す（発話ターン中）

  get_pulse_history() -> list[list[int]]
      直近20パルスの履歴（HUD表示用）
"""
from config import BN_HZ, BN_TURN_SEC, BN_PARAMS, BN_PULSES_PER_TURN, FPS
import math


class Bottleneck:
    """
    スタブ実装：観測ベクトルを直接行動にマッピングする。
    RNN実装に差し替える際はこのクラスを継承してオーバーライドする。
    """

    def __init__(self, weights=None):
        # weights: GA最適化対象の重みベクトル（スタブではシンプルな線形写像）
        import numpy as np
        self.weights = weights  # shape (3, 6) or None

        self._frame_count  = 0
        self._pulse_timer  = 0.0
        self._pulse_interval = 1.0 / BN_HZ   # 0.2秒
        self._turn_timer   = 0.0
        self._turn_duration = float(BN_TURN_SEC)
        self._mode         = "listen"   # "listen" | "speak"

        self._obs_buffer: list[list[float]] = []
        self._last_obs: list[float] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._last_action: list[float] = [0.0, 0.5, 0.0]
        self._pulse_history: list[list[int]] = []
        self._current_pulse: list[int] = [0, 0, 0, 0]

    # ----------------------------------------------------------------
    def push(self, obs: list[float]) -> None:
        """感覚系から観測ベクトルを受け取る。"""
        self._last_obs = obs
        self._obs_buffer.append(obs)

    # ----------------------------------------------------------------
    def tick(self, dt: float) -> tuple[list[int] | None, str]:
        """
        フレームごとに呼ぶ。
        Returns: (pulse or None, mode)
        """
        self._pulse_timer  += dt
        self._turn_timer   += dt

        # ターン切替（4秒固定）
        if self._turn_timer >= self._turn_duration:
            self._turn_timer -= self._turn_duration
            self._mode = "speak" if self._mode == "listen" else "listen"
            self._obs_buffer.clear()

        # パルス発火（200msごと）
        pulse = None
        if self._pulse_timer >= self._pulse_interval:
            self._pulse_timer -= self._pulse_interval
            pulse = self._fire_pulse()
            self._pulse_history.append(pulse[:])
            if len(self._pulse_history) > BN_PULSES_PER_TURN:
                self._pulse_history.pop(0)
            self._current_pulse = pulse

            # 発話ターンのパルスから行動を更新
            if self._mode == "speak":
                self._last_action = self._decode_action(pulse)

        return pulse, self._mode

    # ----------------------------------------------------------------
    def _fire_pulse(self) -> list[int]:
        """
        観測ベクトルを4 bitsパルスに変換する（スタブ：線形閾値）。
        将来: 感覚の海馬LSTM → Attention → 量子化 に差し替え。
        """
        obs = self._last_obs
        # スタブ：観測の各次元を閾値で2値化
        # [energy, goal_angle, grad_x, grad_y, food_dx, food_dy]
        p0 = 1 if obs[0] < 0.4 else 0          # エネルギー低下警告
        p1 = 1 if obs[1] > 0.3 else 0           # ゴール右方向
        p2 = 1 if obs[4] > 0.2 else 0           # 餌が右方向
        p3 = 1 if (obs[2]**2 + obs[3]**2) > 0.1 else 0  # 急勾配
        return [p0, p1, p2, p3]

    # ----------------------------------------------------------------
    def _decode_action(self, pulse: list[int]) -> list[float]:
        """
        4 bitsパルスを行動 [accel, steer, brake] に変換する（スタブ）。
        将来: 運動の海馬LSTM → 補間層 に差し替え。
        """
        if self.weights is not None:
            import numpy as np
            # 線形写像: weights (3,4) @ pulse (4,) → action (3,)
            w = self.weights
            raw = w @ pulse
            # sigmoid で 0〜1 に変換
            action = [1.0 / (1.0 + math.exp(-x)) for x in raw]
            return action

        # デフォルトスタブ：パルスパターンを単純マッピング
        p = pulse
        accel = 0.6 if p[0] == 0 else 0.3   # エネルギー低いと減速
        steer = 0.5 + (p[1] - p[2]) * 0.25  # ゴール方向 vs 餌方向
        brake = 0.3 if p[3] == 1 else 0.0   # 急勾配でブレーキ
        return [
            max(0.0, min(1.0, accel)),
            max(0.0, min(1.0, steer)),
            max(0.0, min(1.0, brake)),
        ]

    # ----------------------------------------------------------------
    def get_action(self) -> list[float]:
        return self._last_action

    def get_pulse_history(self) -> list[list[int]]:
        return self._pulse_history

    def get_mode(self) -> str:
        return self._mode

    def get_turn_progress(self) -> float:
        """現在のターンの進捗（0〜1）"""
        return self._turn_timer / self._turn_duration

    def get_current_pulse(self) -> list[int]:
        return self._current_pulse

    def reset(self):
        self._frame_count  = 0
        self._pulse_timer  = 0.0
        self._turn_timer   = 0.0
        self._mode         = "listen"
        self._obs_buffer.clear()
        self._last_obs     = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._last_action  = [0.0, 0.5, 0.0]
        self._pulse_history.clear()
        self._current_pulse = [0, 0, 0, 0]

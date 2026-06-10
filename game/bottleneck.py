"""
bottleneck.py — リアルタイム半双方向通信バス（Sage-Brute版）

リアルタイム同期方式:
  生成と消化を同周期（PULSE_GEN_INTERVAL=6フレーム）に同期。
  _outgoingバッファ廃止・生成と同時に転送。
  ターン境界は方向切替のみ（バッファ引き継ぎ）。

方向:
  S→B: SAGEが送信・BRUTEが受信
  B→S: BRUTEが送信・SAGEが受信
"""
import numpy as np
from config import (
    PULSE_GEN_INTERVAL, PULSE_CONSUME_RATE, TURN_FRAMES, PULSE_TOTAL,
    PHONEME_TABLE,
)


class Bottleneck:
    """リアルタイム半双方向通信バス。"""

    def __init__(self, sage, brute):
        self.sage  = sage
        self.brute = brute

        self._buf:       list[int] = []
        self._frame:     int = 0
        self._turn:      int = 0
        self.direction:  str = 'S→B'   # 'S→B' or 'B→S'

        self._last_action: np.ndarray = np.array([0.0, 0.5, 0.0])
        self._last_pulse:  int = 0

        # 表示・音声専用
        self._display_queue:   list[int] = []
        self._display_history: list[int] = []
        self._display_phoneme: str = ""
        self._display_frame:   int = 0

        self.audio_enabled: bool = False
        self.converter = None
        self._last_phoneme: str = ""

    def reset_episode(self):
        """エピソードリセット（spec互換エイリアス）。"""
        self.reset(prefill=True)

    def reset(self, prefill: bool = True):
        self._buf      = []
        self._frame    = 0
        self._turn     = 0
        self.direction = 'S→B'
        self._last_action   = np.array([0.0, 0.5, 0.0])
        self._last_pulse    = 0
        self._last_obs_sage  = None   # モニター表示用
        self._last_obs_brute = None   # モニター表示用
        self._display_queue   = []
        self._display_history = []
        self._display_phoneme = ""
        self._display_frame   = 0
        self.sage.reset_episode()
        self.brute.reset_episode()

        if prefill:
            # ダミー入力で1発生成してパイプラインを起動
            dummy_sage  = np.zeros(self.sage.W3.shape[1])
            dummy_brute = np.zeros(self.brute.W3.shape[1])
            if self.direction == 'S→B':
                p = self.sage.forward(dummy_sage, is_pulse_frame=True)
            else:
                _, p = self.brute.forward(dummy_brute, is_pulse_frame=True)
            self._buf = [p]
            self._display_queue = [p]

    def step(self, obs_sage: np.ndarray, obs_brute: np.ndarray) -> np.ndarray:
        """
        1フレーム処理。
        Returns: BRUTEの行動ベクトル [Accel, Steer, Brake]
        """
        # モニター表示用に保存
        self._last_obs_sage  = obs_sage
        self._last_obs_brute = obs_brute

        f = self._frame
        is_gen  = (f % PULSE_GEN_INTERVAL == 0)
        is_cons = (f % PULSE_CONSUME_RATE == 0)
        is_turn = (f > 0 and f % TURN_FRAMES == 0)

        # --- 1. パルス生成（送信側）→ 即座にバッファへ ---
        if is_gen:
            if self.direction == 'S→B':
                pulse = self.sage.forward(obs_sage, is_pulse_frame=True)
            else:
                _, pulse = self.brute.forward(obs_brute, is_pulse_frame=True)
            self._buf.append(pulse)
            self._last_pulse = pulse
            # 表示キューにも追加
            self._display_queue.append(pulse)
        else:
            # 非生成フレームでも両NNはobsを処理（GRU隠れ状態を更新）
            self.sage.forward(obs_sage, is_pulse_frame=False)
            _, _ = self.brute.forward(obs_brute, is_pulse_frame=False)

        # --- 2. 表示タイマー: 生成と同周期でスロット点灯・音声出力 ---
        if self._display_queue and self._display_frame % PULSE_GEN_INTERVAL == 0:
            dp = self._display_queue.pop(0)
            self._display_history.append(dp)
            if len(self._display_history) > PULSE_TOTAL:
                self._display_history.pop(0)
            self._display_phoneme = PHONEME_TABLE.get(dp & 0x3, "")
            self._last_phoneme    = self._display_phoneme
            if self.audio_enabled and self.converter is not None:
                try:
                    self.converter.play(dp, self.direction)
                except Exception:
                    pass
        self._display_frame += 1

        # --- 3. パルス消化（受信側）---
        if is_cons and self._buf:
            pulse = self._buf.pop(0)
            if self.direction == 'S→B':
                # SAGEからBRUTEへ: パルスをBRUTE obsに注入して行動取得
                obs_b = obs_brute.copy()
                obs_b[-2] = float((pulse >> 1) & 1)
                obs_b[-1] = float(pulse & 1)
                action, _ = self.brute.forward(obs_b, is_pulse_frame=True)
                self._last_action = action
            else:
                # BRUTEからSAGEへ: パルスをSAGE obsに注入
                obs_s = obs_sage.copy()
                obs_s[-2] = float((pulse >> 1) & 1)
                obs_s[-1] = float(pulse & 1)
                self.sage.forward(obs_s, is_pulse_frame=True)

        # --- 4. ターン切替（方向のみ・バッファ引き継ぎ）---
        if is_turn:
            self._turn += 1
            self.direction = 'B→S' if self.direction == 'S→B' else 'S→B'

        self._frame += 1
        return self._last_action

    # ---- 音声 ----
    def enable_audio(self):
        if self.audio_enabled:
            return
        try:
            from game.phoneme import PhonemeConverter
            if self.converter is None:
                self.converter = PhonemeConverter()
            self.audio_enabled = True
        except Exception:
            pass

    def disable_audio(self):
        self.audio_enabled = False

    def toggle_audio(self) -> bool:
        if self.audio_enabled:
            self.disable_audio()
        else:
            self.enable_audio()
        return self.audio_enabled

    # ---- getters（renderer互換） ----
    def get_mode(self) -> str:
        return 'listen' if self.direction == 'S→B' else 'speak'

    def get_turn_progress(self) -> float:
        return (self._frame % TURN_FRAMES) / TURN_FRAMES

    def get_current_pulse(self) -> list[int]:
        p = self._last_pulse
        return [(p >> 1) & 1, p & 1]

    def get_pulse_history(self) -> list[list[int]]:
        return [[(p >> 1) & 1, p & 1] for p in self._display_history]

    def get_display_history(self) -> list[list[int]]:
        return self.get_pulse_history()

    def get_display_phoneme(self) -> str:
        return self._display_phoneme

    def get_display_progress(self) -> float:
        total = PULSE_TOTAL
        return len(self._display_history) / total if total > 0 else 0.0

    def get_last_phoneme(self) -> str:
        return self._last_phoneme

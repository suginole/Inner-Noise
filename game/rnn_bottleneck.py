"""
rnn_bottleneck.py — RNNボトルネック本実装

アーキテクチャ:
  SensoryNN: obs(12) → 感覚皮質FF(12→16,tanh) → 感覚GRU(16→16) → パルス符号化FF(16→2,tanh) → Step → 2bits
  MotorNN:   2bits → パルス埋め込みFF(2→16) → 運動GRU(16→16) → 運動皮質FF(16→12,tanh) → 出力FF(12→3,sigmoid)

パイプライン型半双方向通信:
  - フレームカウンタのみで管理（time.time()不使用）
  - ターン1: S→M送信（12フレームごとに2bitsパルス生成・送信）
  - ターン2以降: 24フレームごとにパルスバッファから1発消化（半速）
  - 定常状態: 前ターンの後半10パルスを消化中 + 今ターンの前半10パルスが届いている

GAが進化させるもの:
  - 感覚皮質FF重み・バイアス
  - 感覚GRU重み
  - 感覚GRU初期隠れ状態 γ_s（16次元）
  - パルス埋め込みFF重み・バイアス
  - 運動GRU重み
  - 運動GRU初期隠れ状態 γ_m（16次元）
  - 運動皮質FF重み・バイアス
  - 出力FF重み・バイアス
  合計 ≈ 3,637次元

オンライン更新（GAが進化させない）:
  - 感覚GRU隠れ状態（エピソード内のみ、世代間非継承）
  - 運動GRU隠れ状態（同上）
"""
from __future__ import annotations
import numpy as np
from config import (
    SENSORY_INPUT_DIM, SENSORY_CORTEX_DIM, SENSORY_GRU_DIM,
    MOTOR_EMBED_DIM, MOTOR_GRU_DIM, MOTOR_CORTEX_DIM, MOTOR_OUTPUT_DIM,
    TURN_FRAMES, PULSE_TOTAL, PULSE_CONSUME_RATE, PIPELINE_OFFSET,
    BN_PARAMS,
)


# ================================================================
# GRUセル（手動実装）
# ================================================================
def gru_step(x: np.ndarray, h: np.ndarray,
             Wz: np.ndarray, Wr: np.ndarray, Wh: np.ndarray,
             Uz: np.ndarray, Ur: np.ndarray, Uh: np.ndarray,
             bz: np.ndarray, br: np.ndarray, bh: np.ndarray) -> np.ndarray:
    """
    GRUセルの1ステップ更新。

    z = sigmoid(Wz@x + Uz@h + bz)   # update gate
    r = sigmoid(Wr@x + Ur@h + br)   # reset gate
    h̃ = tanh(Wh@x + Uh@(r*h) + bh) # candidate
    h' = (1-z)*h + z*h̃
    """
    def sigmoid(v): return 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))

    z  = sigmoid(Wz @ x + Uz @ h + bz)
    r  = sigmoid(Wr @ x + Ur @ h + br)
    h_ = np.tanh(Wh @ x + Uh @ (r * h) + bh)
    return (1 - z) * h + z * h_


# ================================================================
# SensoryNN
# ================================================================
class SensoryNN:
    """
    感覚NN。obs(12) → 2bitsパルス。

    GAが進化させる重み:
      W_cortex, b_cortex  (12×16, 16)
      GRU重み 9行列       (各16×16 or 16×16)
      W_encode, b_encode  (16×2, 2)
      gamma_s             (16,)  ← 初期隠れ状態
    """

    def __init__(self, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()
        s = 0.3

        # 感覚皮質FF: 12→16
        self.W_cortex = rng.normal(0, s, (SENSORY_CORTEX_DIM, SENSORY_INPUT_DIM))
        self.b_cortex = np.zeros(SENSORY_CORTEX_DIM)

        # 感覚GRU重み (input_dim=16, hidden_dim=16)
        id_, hd = SENSORY_CORTEX_DIM, SENSORY_GRU_DIM
        self.Wz = rng.normal(0, s, (hd, id_)); self.Uz = rng.normal(0, s, (hd, hd)); self.bz = np.zeros(hd)
        self.Wr = rng.normal(0, s, (hd, id_)); self.Ur = rng.normal(0, s, (hd, hd)); self.br = np.zeros(hd)
        self.Wh = rng.normal(0, s, (hd, id_)); self.Uh = rng.normal(0, s, (hd, hd)); self.bh = np.zeros(hd)

        # パルス符号化FF: 16→2
        self.W_encode = rng.normal(0, s, (BN_PARAMS, SENSORY_GRU_DIM))
        self.b_encode = np.zeros(BN_PARAMS)

        # 初期隠れ状態γ（GAが進化させる）
        self.gamma_s = np.zeros(SENSORY_GRU_DIM)

        # オンライン状態（世代間非継承）
        self._h: np.ndarray = self.gamma_s.copy()

        # アクティベーション記録（可視化用）
        self.last_cortex_act: list[float] = [0.0] * SENSORY_CORTEX_DIM
        self.last_gru_act:    list[float] = [0.0] * SENSORY_GRU_DIM
        self.last_pulse:      list[int]   = [0, 0]

    def reset(self):
        """エピソード開始時にγでリセット。"""
        self._h = self.gamma_s.copy()

    def forward(self, obs: list[float]) -> list[int]:
        """obs → 2bitsパルス（[p0, p1]）を返す。"""
        x = np.array(obs, dtype=np.float32)

        # 感覚皮質FF
        cortex = np.tanh(self.W_cortex @ x + self.b_cortex)
        self.last_cortex_act = cortex.tolist()

        # 感覚GRU
        self._h = gru_step(cortex, self._h,
                           self.Wz, self.Wr, self.Wh,
                           self.Uz, self.Ur, self.Uh,
                           self.bz, self.br, self.bh)
        self.last_gru_act = self._h.tolist()

        # パルス符号化FF + Step関数
        enc = np.tanh(self.W_encode @ self._h + self.b_encode)
        bits = (enc >= 0).astype(int)
        pulse = [int(bits[0]), int(bits[1])]
        self.last_pulse = pulse
        return pulse

    def encode_pulse(self, obs: list[float]) -> int:
        """obs → 2bits整数を返す（bits2 = p0<<1 | p1）。"""
        p = self.forward(obs)
        return (p[0] << 1) | p[1]

    # ----------------------------------------------------------------
    # GA用フラット化・復元
    # ----------------------------------------------------------------
    def flat(self) -> np.ndarray:
        return np.concatenate([
            self.W_cortex.ravel(), self.b_cortex,
            self.Wz.ravel(), self.Uz.ravel(), self.bz,
            self.Wr.ravel(), self.Ur.ravel(), self.br,
            self.Wh.ravel(), self.Uh.ravel(), self.bh,
            self.W_encode.ravel(), self.b_encode,
            self.gamma_s,
        ])

    @staticmethod
    def flat_size() -> int:
        id_, hd = SENSORY_CORTEX_DIM, SENSORY_GRU_DIM
        return (
            SENSORY_INPUT_DIM * hd + hd +          # W_cortex, b_cortex
            3 * (id_ * hd + hd * hd + hd) +        # GRU 3ゲート
            hd * BN_PARAMS + BN_PARAMS +            # W_encode, b_encode
            hd                                      # gamma_s
        )

    def load_flat(self, v: np.ndarray):
        i = 0
        def take(n):
            nonlocal i; r = v[i:i+n]; i += n; return r
        id_, hd = SENSORY_CORTEX_DIM, SENSORY_GRU_DIM
        in_dim = SENSORY_INPUT_DIM
        self.W_cortex = take(in_dim * hd).reshape(hd, in_dim)
        self.b_cortex = take(hd)
        self.Wz = take(id_ * hd).reshape(hd, id_); self.Uz = take(hd * hd).reshape(hd, hd); self.bz = take(hd)
        self.Wr = take(id_ * hd).reshape(hd, id_); self.Ur = take(hd * hd).reshape(hd, hd); self.br = take(hd)
        self.Wh = take(id_ * hd).reshape(hd, id_); self.Uh = take(hd * hd).reshape(hd, hd); self.bh = take(hd)
        self.W_encode = take(hd * BN_PARAMS).reshape(BN_PARAMS, hd); self.b_encode = take(BN_PARAMS)
        self.gamma_s  = take(hd)


# ================================================================
# MotorNN
# ================================================================
class MotorNN:
    """
    運動NN。2bitsパルス → [Accel, Steer, Brake]。

    GAが進化させる重み:
      W_embed, b_embed    (2×16, 16)
      GRU重み 9行列
      W_cortex, b_cortex  (16×12, 12)
      W_out, b_out        (12×3, 3)
      gamma_m             (16,)
    """

    def __init__(self, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()
        s = 0.3

        # パルス埋め込みFF: 2→16
        self.W_embed = rng.normal(0, s, (MOTOR_EMBED_DIM, BN_PARAMS))
        self.b_embed = np.zeros(MOTOR_EMBED_DIM)

        # 運動GRU重み
        id_, hd = MOTOR_EMBED_DIM, MOTOR_GRU_DIM
        self.Wz = rng.normal(0, s, (hd, id_)); self.Uz = rng.normal(0, s, (hd, hd)); self.bz = np.zeros(hd)
        self.Wr = rng.normal(0, s, (hd, id_)); self.Ur = rng.normal(0, s, (hd, hd)); self.br = np.zeros(hd)
        self.Wh = rng.normal(0, s, (hd, id_)); self.Uh = rng.normal(0, s, (hd, hd)); self.bh = np.zeros(hd)

        # 運動皮質FF: 16→12
        self.W_cortex = rng.normal(0, s, (MOTOR_CORTEX_DIM, MOTOR_GRU_DIM))
        self.b_cortex = np.zeros(MOTOR_CORTEX_DIM)

        # 出力FF: 12→3
        self.W_out = rng.normal(0, s, (MOTOR_OUTPUT_DIM, MOTOR_CORTEX_DIM))
        self.b_out = np.zeros(MOTOR_OUTPUT_DIM)

        # 初期隠れ状態γ（GAが進化させる）
        self.gamma_m = np.zeros(MOTOR_GRU_DIM)

        # オンライン状態
        self._h: np.ndarray = self.gamma_m.copy()

        # アクティベーション記録（可視化用）
        self.last_gru_act:    list[float] = [0.0] * MOTOR_GRU_DIM
        self.last_output_act: list[float] = [0.5, 0.5, 0.0]

    def reset(self):
        self._h = self.gamma_m.copy()

    def forward(self, pulse: list[int]) -> list[float]:
        """2bitsパルス → [Accel, Steer, Brake]（各0〜1）。"""
        x = np.array(pulse, dtype=np.float32)

        # パルス埋め込みFF
        embed = np.tanh(self.W_embed @ x + self.b_embed)

        # 運動GRU
        self._h = gru_step(embed, self._h,
                           self.Wz, self.Wr, self.Wh,
                           self.Uz, self.Ur, self.Uh,
                           self.bz, self.br, self.bh)
        self.last_gru_act = self._h.tolist()

        # 運動皮質FF
        cortex = np.tanh(self.W_cortex @ self._h + self.b_cortex)

        # 出力FF (sigmoid)
        def sigmoid(v): return 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))
        out = sigmoid(self.W_out @ cortex + self.b_out)
        self.last_output_act = out.tolist()
        return out.tolist()

    # ----------------------------------------------------------------
    def flat(self) -> np.ndarray:
        return np.concatenate([
            self.W_embed.ravel(), self.b_embed,
            self.Wz.ravel(), self.Uz.ravel(), self.bz,
            self.Wr.ravel(), self.Ur.ravel(), self.br,
            self.Wh.ravel(), self.Uh.ravel(), self.bh,
            self.W_cortex.ravel(), self.b_cortex,
            self.W_out.ravel(), self.b_out,
            self.gamma_m,
        ])

    @staticmethod
    def flat_size() -> int:
        id_, hd = MOTOR_EMBED_DIM, MOTOR_GRU_DIM
        return (
            BN_PARAMS * id_ + id_ +                 # W_embed, b_embed
            3 * (id_ * hd + hd * hd + hd) +         # GRU 3ゲート
            hd * MOTOR_CORTEX_DIM + MOTOR_CORTEX_DIM +  # W_cortex, b_cortex
            MOTOR_CORTEX_DIM * MOTOR_OUTPUT_DIM + MOTOR_OUTPUT_DIM +  # W_out, b_out
            hd                                       # gamma_m
        )

    def load_flat(self, v: np.ndarray):
        i = 0
        def take(n):
            nonlocal i; r = v[i:i+n]; i += n; return r
        id_, hd = MOTOR_EMBED_DIM, MOTOR_GRU_DIM
        self.W_embed = take(BN_PARAMS * id_).reshape(id_, BN_PARAMS); self.b_embed = take(id_)
        self.Wz = take(id_ * hd).reshape(hd, id_); self.Uz = take(hd * hd).reshape(hd, hd); self.bz = take(hd)
        self.Wr = take(id_ * hd).reshape(hd, id_); self.Ur = take(hd * hd).reshape(hd, hd); self.br = take(hd)
        self.Wh = take(id_ * hd).reshape(hd, id_); self.Uh = take(hd * hd).reshape(hd, hd); self.bh = take(hd)
        self.W_cortex = take(hd * MOTOR_CORTEX_DIM).reshape(MOTOR_CORTEX_DIM, hd); self.b_cortex = take(MOTOR_CORTEX_DIM)
        self.W_out    = take(MOTOR_CORTEX_DIM * MOTOR_OUTPUT_DIM).reshape(MOTOR_OUTPUT_DIM, MOTOR_CORTEX_DIM)
        self.b_out    = take(MOTOR_OUTPUT_DIM)
        self.gamma_m  = take(hd)


# ================================================================
# RNNBottleneck（パイプライン型半双方向通信）
# ================================================================
class RNNBottleneck:
    """
    パイプライン型半双方向通信ボトルネック。

    フレームカウンタのみで管理（time.time()不使用）。

    ターン1（0〜TURN_FRAMES）:
      S側: 12フレームごとにパルス生成・送信（計20発）
      M側: γによる自律制御のみ（パルス未受信）

    ターン2以降（定常パイプライン）:
      24フレームごとにパルスバッファから1発消化（半速）
      消化と並行して次ターン用パルスを裏で生成・蓄積
      ターン終了時に蓄積パルスを一斉送信

    音声出力フック:
      パルス送信時に on_pulse_emit() を呼ぶ（Bottleneckと同じI/F）
    """

    PULSE_GEN_INTERVAL = TURN_FRAMES // PULSE_TOTAL   # = 12フレームごとに1パルス生成

    def __init__(self, sensory: SensoryNN, motor: MotorNN):
        self.sensory = sensory
        self.motor   = motor

        # パイプライン状態
        self._frame:         int = 0
        self._turn:          int = 0
        self._pulse_buffer:  list[list[int]] = []   # 受信済みパルスのキュー
        self._outgoing:      list[list[int]] = []   # 次ターン用生成中パルス
        self._consume_count: int = 0   # 今ターンの消化済みパルス数

        # 現在のパルス・履歴（HUD表示用）
        self._current_pulse: list[int] = [0, 0]
        self._pulse_history: list[list[int]] = []
        self._mode:          str = "listen"   # "listen" | "speak"

        # 音声フック（Bottleneckと同じI/F）
        self.audio_enabled: bool = False
        self.converter = None
        self._last_phoneme: str = ""

        # 最後の行動
        self._last_action: list[float] = [0.0, 0.5, 0.0]

        # アクティベーション記録（可視化用）
        self.last_sensory_gru: list[float] = [0.0] * SENSORY_GRU_DIM
        self.last_motor_gru:   list[float] = [0.0] * MOTOR_GRU_DIM
        self.last_output:      list[float] = [0.5, 0.5, 0.0]

    def reset(self):
        """エピソード開始時にリセット。"""
        self._frame         = 0
        self._turn          = 0
        self._pulse_buffer  = []
        self._outgoing      = []
        self._consume_count = 0
        self._current_pulse = [0, 0]
        self._pulse_history = []
        self._mode          = "listen"
        self._last_action   = [0.0, 0.5, 0.0]
        self.sensory.reset()
        self.motor.reset()

    # ----------------------------------------------------------------
    def step(self, obs: list[float]) -> list[float]:
        """
        1フレーム分の更新。
        obs: 感覚観測ベクトル（12次元）
        Returns: [Accel, Steer, Brake]
        """
        f = self._frame

        # ---- S側: パルス生成（12フレームごと）----
        if f % self.PULSE_GEN_INTERVAL == 0:
            pulse = self.sensory.forward(obs)
            self._outgoing.append(pulse)
            self.last_sensory_gru = self.sensory.last_gru_act

        # ---- ターン終了: 蓄積パルスを一斉送信 ----
        if f > 0 and f % TURN_FRAMES == 0:
            self._pulse_buffer.extend(self._outgoing)
            self._outgoing = []
            self._turn += 1
            self._consume_count = 0
            self._mode = "speak" if self._mode == "listen" else "listen"

        # ---- M側: パルスバッファから半速消化（24フレームごと）----
        if f % PULSE_CONSUME_RATE == 0 and self._pulse_buffer:
            pulse = self._pulse_buffer.pop(0)
            self._current_pulse = pulse
            self._pulse_history.append(pulse[:])
            if len(self._pulse_history) > PULSE_TOTAL:
                self._pulse_history.pop(0)
            self._consume_count += 1

            # 音声フック
            bits2 = (pulse[0] << 1) | pulse[1]
            self.on_pulse_emit(bits2)

            # 運動GRUを更新
            action = self.motor.forward(pulse)
            self._last_action = action
            self.last_motor_gru = self.motor.last_gru_act
            self.last_output    = self.motor.last_output_act

        # ---- ターン1（パルス未受信）: γによる自律制御 ----
        elif self._turn == 0:
            # γ（初期隠れ状態）のみで行動を生成
            action = self.motor.forward([0, 0])
            self._last_action = action
            self.last_motor_gru = self.motor.last_gru_act
            self.last_output    = self.motor.last_output_act

        self._frame += 1
        return self._last_action

    # ----------------------------------------------------------------
    def on_pulse_emit(self, bits2: int) -> None:
        """パルス消化時（受信側）に呼ばれる音声フック。
        送信側では呼ばない（2重鳴防止）。
        傘聴ターン（Mが受信）: S→M方向の高ピッチ（♀）
        発話ターン（Sが受信）: M→S方向の低ピッチ（♂）
        """
        from config import PHONEME_TABLE
        self._last_phoneme = PHONEME_TABLE.get(bits2 & 0x3, "")
        if self.audio_enabled and self.converter is not None:
            # 消化側のターンに応じた方向で再生
            # listenターン中はステージングパートのパルスをMが受信 → S→M方向
            direction = 'S→M' if self._mode == 'listen' else 'M→S'
            try:
                self.converter.play(bits2, direction=direction)
            except Exception:
                pass

    def enable_audio(self) -> None:
        if self.audio_enabled:
            return
        try:
            from game.phoneme import PhonemeConverter, PhonemeDecoder
            if self.converter is None:
                self.converter = PhonemeConverter()
            self.audio_enabled = True
        except Exception:
            pass

    def disable_audio(self) -> None:
        self.audio_enabled = False

    def toggle_audio(self) -> bool:
        if self.audio_enabled:
            self.disable_audio()
        else:
            self.enable_audio()
        return self.audio_enabled

    # ---- HUD用プロパティ ----
    def get_current_pulse(self) -> list[int]:
        return self._current_pulse

    def get_pulse_history(self) -> list[list[int]]:
        return self._pulse_history

    def get_mode(self) -> str:
        return self._mode

    def get_turn_progress(self) -> float:
        return (self._frame % TURN_FRAMES) / TURN_FRAMES

    def get_last_phoneme(self) -> str:
        return self._last_phoneme

    def get_consume_progress(self) -> tuple[int, int]:
        """(消化済みパルス数, バッファ残数)"""
        return self._consume_count, len(self._pulse_buffer)

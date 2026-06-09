"""
rnn_bottleneck.py — RNNボトルネック本実装（継承・非継承分割設計）

設計方針:
  「発話方法を毎世代忘れる」問題を解決するため、
  GRU隠れ状態を継承領域（GA進化対象）と非継承領域（エピソード内のみ）に分割。

GRU隠れ状態の分割（感覚・運動共通）:
  上位8次元: GA継承領域 ← 発話の癖・文法（世代間で引き継ぐ）
  下位8次元: 非継承領域 ← エピソード固有の経験・文脈（毎回ゼロリセット）

  エピソード開始時:
    h = concat([gamma[:8], zeros(8)])

感覚NN:
  obs(12) → 入力FF(12→16,tanh) → 記憶GRU(16→16) → 統合FF(16→16,tanh)
           → パルス符号化FF(16→2,tanh) → Step → 2bits

運動NN:
  2bits → 埋め込みFF(2→16) → 記憶GRU(16→16) → 統合FF(16→16,tanh)
        → 運動皮質FF(16→12,tanh) → 出力FF(12→3,sigmoid)

GAが進化させるもの:
  感覚: 入力FF重み, GRU重み(9行列), 統合FF重み, パルス符号化FF重み, γ_s(8次元)
  運動: 埋め込みFF重み, GRU重み(9行列), 統合FF重み, 運動皮質FF重み, 出力FF重み, γ_m(8次元)
  合計 ≈ 4,245次元

オンライン更新（GAが進化させない）:
  GRU隠れ状態の下位8次元（エピソード内のみ）
"""
from __future__ import annotations
import numpy as np
from config import (
    SENSORY_INPUT_DIM, SENSORY_FF_DIM, SENSORY_GRU_DIM, SENSORY_INTEG_DIM,
    MOTOR_EMBED_DIM, MOTOR_GRU_DIM, MOTOR_INTEG_DIM, MOTOR_CORTEX_DIM, MOTOR_OUTPUT_DIM,
    TURN_FRAMES, PULSE_TOTAL, PULSE_CONSUME_RATE, PIPELINE_OFFSET,
    BN_PARAMS, GRU_INHERIT_DIM, GRU_EPISODE_DIM,
)


# ================================================================
# GRUセル（手動実装）
# ================================================================
def gru_step(x: np.ndarray, h: np.ndarray,
             Wz: np.ndarray, Wr: np.ndarray, Wh: np.ndarray,
             Uz: np.ndarray, Ur: np.ndarray, Uh: np.ndarray,
             bz: np.ndarray, br: np.ndarray, bh: np.ndarray) -> np.ndarray:
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

    obs(12) → 入力FF(12→16,tanh) → 記憶GRU(16→16)
            → 統合FF(16→16,tanh) → パルス符号化FF(16→2,tanh) → Step → 2bits

    GAが進化させる重み:
      W_input, b_input  (12×16, 16)
      GRU重み 9行列
      W_integ, b_integ  (16×16, 16)
      W_encode, b_encode (16×2, 2)
      gamma_s           (8,)  ← GRU継承領域の初期値
    """

    def __init__(self, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()
        s = 0.3

        # 入力FF: 12→16
        self.W_input  = rng.normal(0, s, (SENSORY_FF_DIM, SENSORY_INPUT_DIM))
        self.b_input  = np.zeros(SENSORY_FF_DIM)

        # 記憶GRU重み (input_dim=16, hidden_dim=16)
        id_, hd = SENSORY_FF_DIM, SENSORY_GRU_DIM
        self.Wz = rng.normal(0, s, (hd, id_)); self.Uz = rng.normal(0, s, (hd, hd)); self.bz = np.zeros(hd)
        self.Wr = rng.normal(0, s, (hd, id_)); self.Ur = rng.normal(0, s, (hd, hd)); self.br = np.zeros(hd)
        self.Wh = rng.normal(0, s, (hd, id_)); self.Uh = rng.normal(0, s, (hd, hd)); self.bh = np.zeros(hd)

        # 統合FF: 16→16
        self.W_integ  = rng.normal(0, s, (SENSORY_INTEG_DIM, SENSORY_GRU_DIM))
        self.b_integ  = np.zeros(SENSORY_INTEG_DIM)

        # パルス符号化FF: 16→2
        self.W_encode = rng.normal(0, s, (BN_PARAMS, SENSORY_INTEG_DIM))
        self.b_encode = np.zeros(BN_PARAMS)

        # 初期隠れ状態γ（継承領域8次元のみ・GAが進化させる）
        self.gamma_s = np.zeros(GRU_INHERIT_DIM)

        # オンライン状態（エピソード内のみ）
        self._h: np.ndarray = self._init_h()

        # アクティベーション記録（可視化用）
        self.last_input_act:  list[float] = [0.0] * SENSORY_FF_DIM
        self.last_cortex_act: list[float] = [0.0] * SENSORY_FF_DIM  # 互換性用
        self.last_gru_act:    list[float] = [0.0] * SENSORY_GRU_DIM
        self.last_integ_act:  list[float] = [0.0] * SENSORY_INTEG_DIM
        self.last_pulse:      list[int]   = [0, 0]

    def _init_h(self) -> np.ndarray:
        """エピソード開始時の隠れ状態を生成する。
        上位8次元: γ（継承）、下位8次元: ゼロ（非継承）
        """
        return np.concatenate([self.gamma_s, np.zeros(GRU_EPISODE_DIM)])

    def reset(self):
        self._h = self._init_h()

    def forward(self, obs: list[float]) -> list[int]:
        x = np.array(obs, dtype=np.float32)

        # 入力FF
        inp = np.tanh(self.W_input @ x + self.b_input)
        self.last_input_act  = inp.tolist()
        self.last_cortex_act = inp.tolist()  # 互換性

        # 記憶GRU
        self._h = gru_step(inp, self._h,
                           self.Wz, self.Wr, self.Wh,
                           self.Uz, self.Ur, self.Uh,
                           self.bz, self.br, self.bh)
        self.last_gru_act = self._h.tolist()

        # 統合FF
        integ = np.tanh(self.W_integ @ self._h + self.b_integ)
        self.last_integ_act = integ.tolist()

        # パルス符号化FF + Step
        enc = np.tanh(self.W_encode @ integ + self.b_encode)
        bits = (enc >= 0).astype(int)
        pulse = [int(bits[0]), int(bits[1])]
        self.last_pulse = pulse
        return pulse

    def encode_pulse(self, obs: list[float]) -> int:
        p = self.forward(obs)
        return (p[0] << 1) | p[1]

    # ----------------------------------------------------------------
    def flat(self) -> np.ndarray:
        return np.concatenate([
            self.W_input.ravel(), self.b_input,
            self.Wz.ravel(), self.Uz.ravel(), self.bz,
            self.Wr.ravel(), self.Ur.ravel(), self.br,
            self.Wh.ravel(), self.Uh.ravel(), self.bh,
            self.W_integ.ravel(), self.b_integ,
            self.W_encode.ravel(), self.b_encode,
            self.gamma_s,
        ])

    @staticmethod
    def flat_size() -> int:
        id_, hd = SENSORY_FF_DIM, SENSORY_GRU_DIM
        return (
            SENSORY_INPUT_DIM * id_ + id_ +          # W_input, b_input
            3 * (id_ * hd + hd * hd + hd) +          # GRU 3ゲート
            hd * SENSORY_INTEG_DIM + SENSORY_INTEG_DIM +  # W_integ, b_integ
            SENSORY_INTEG_DIM * BN_PARAMS + BN_PARAMS +   # W_encode, b_encode
            GRU_INHERIT_DIM                           # gamma_s
        )

    def load_flat(self, v: np.ndarray):
        i = 0
        def take(n):
            nonlocal i; r = v[i:i+n]; i += n; return r
        id_, hd = SENSORY_FF_DIM, SENSORY_GRU_DIM
        self.W_input  = take(SENSORY_INPUT_DIM * id_).reshape(id_, SENSORY_INPUT_DIM)
        self.b_input  = take(id_)
        self.Wz = take(id_ * hd).reshape(hd, id_); self.Uz = take(hd * hd).reshape(hd, hd); self.bz = take(hd)
        self.Wr = take(id_ * hd).reshape(hd, id_); self.Ur = take(hd * hd).reshape(hd, hd); self.br = take(hd)
        self.Wh = take(id_ * hd).reshape(hd, id_); self.Uh = take(hd * hd).reshape(hd, hd); self.bh = take(hd)
        self.W_integ  = take(hd * SENSORY_INTEG_DIM).reshape(SENSORY_INTEG_DIM, hd)
        self.b_integ  = take(SENSORY_INTEG_DIM)
        self.W_encode = take(SENSORY_INTEG_DIM * BN_PARAMS).reshape(BN_PARAMS, SENSORY_INTEG_DIM)
        self.b_encode = take(BN_PARAMS)
        self.gamma_s  = take(GRU_INHERIT_DIM)


# ================================================================
# MotorNN
# ================================================================
class MotorNN:
    """
    運動NN。2bitsパルス → [Accel, Steer, Brake]。

    2bits → 埋め込みFF(2→16) → 記憶GRU(16→16) → 統合FF(16→16,tanh)
           → 運動皮質FF(16→12,tanh) → 出力FF(12→3,sigmoid)

    GAが進化させる重み:
      W_embed, b_embed
      GRU重み 9行列
      W_integ, b_integ  (16×16, 16)  ← 新規追加
      W_cortex, b_cortex (16×12, 12)
      W_out, b_out      (12×3, 3)
      gamma_m           (8,)
    """

    def __init__(self, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()
        s = 0.3

        # 埋め込みFF: 2→16
        self.W_embed = rng.normal(0, s, (MOTOR_EMBED_DIM, BN_PARAMS))
        self.b_embed = np.zeros(MOTOR_EMBED_DIM)

        # 記憶GRU重み
        id_, hd = MOTOR_EMBED_DIM, MOTOR_GRU_DIM
        self.Wz = rng.normal(0, s, (hd, id_)); self.Uz = rng.normal(0, s, (hd, hd)); self.bz = np.zeros(hd)
        self.Wr = rng.normal(0, s, (hd, id_)); self.Ur = rng.normal(0, s, (hd, hd)); self.br = np.zeros(hd)
        self.Wh = rng.normal(0, s, (hd, id_)); self.Uh = rng.normal(0, s, (hd, hd)); self.bh = np.zeros(hd)

        # 統合FF: 16→16
        self.W_integ  = rng.normal(0, s, (MOTOR_INTEG_DIM, MOTOR_GRU_DIM))
        self.b_integ  = np.zeros(MOTOR_INTEG_DIM)

        # 運動皮質FF: 16→12
        self.W_cortex = rng.normal(0, s, (MOTOR_CORTEX_DIM, MOTOR_INTEG_DIM))
        self.b_cortex = np.zeros(MOTOR_CORTEX_DIM)

        # 出力FF: 12→3
        self.W_out = rng.normal(0, s, (MOTOR_OUTPUT_DIM, MOTOR_CORTEX_DIM))
        self.b_out = np.zeros(MOTOR_OUTPUT_DIM)

        # 初期隠れ状態γ（継承領域8次元のみ）
        self.gamma_m = np.zeros(GRU_INHERIT_DIM)

        # オンライン状態
        self._h: np.ndarray = self._init_h()

        # アクティベーション記録（可視化用）
        self.last_embed_act:  list[float] = [0.0] * MOTOR_EMBED_DIM
        self.last_gru_act:    list[float] = [0.0] * MOTOR_GRU_DIM
        self.last_integ_act:  list[float] = [0.0] * MOTOR_INTEG_DIM
        self.last_cortex_act: list[float] = [0.0] * MOTOR_CORTEX_DIM
        self.last_output_act: list[float] = [0.5, 0.5, 0.0]

    def _init_h(self) -> np.ndarray:
        return np.concatenate([self.gamma_m, np.zeros(GRU_EPISODE_DIM)])

    def reset(self):
        self._h = self._init_h()

    def forward(self, pulse: list[int]) -> list[float]:
        x = np.array(pulse, dtype=np.float32)

        # 埋め込みFF
        embed = np.tanh(self.W_embed @ x + self.b_embed)
        self.last_embed_act = embed.tolist()

        # 記憶GRU
        self._h = gru_step(embed, self._h,
                           self.Wz, self.Wr, self.Wh,
                           self.Uz, self.Ur, self.Uh,
                           self.bz, self.br, self.bh)
        self.last_gru_act = self._h.tolist()

        # 統合FF
        integ = np.tanh(self.W_integ @ self._h + self.b_integ)
        self.last_integ_act = integ.tolist()

        # 運動皮質FF
        cortex = np.tanh(self.W_cortex @ integ + self.b_cortex)
        self.last_cortex_act = cortex.tolist()

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
            self.W_integ.ravel(), self.b_integ,
            self.W_cortex.ravel(), self.b_cortex,
            self.W_out.ravel(), self.b_out,
            self.gamma_m,
        ])

    @staticmethod
    def flat_size() -> int:
        id_, hd = MOTOR_EMBED_DIM, MOTOR_GRU_DIM
        return (
            BN_PARAMS * id_ + id_ +                          # W_embed, b_embed
            3 * (id_ * hd + hd * hd + hd) +                  # GRU 3ゲート
            hd * MOTOR_INTEG_DIM + MOTOR_INTEG_DIM +          # W_integ, b_integ
            MOTOR_INTEG_DIM * MOTOR_CORTEX_DIM + MOTOR_CORTEX_DIM +  # W_cortex, b_cortex
            MOTOR_CORTEX_DIM * MOTOR_OUTPUT_DIM + MOTOR_OUTPUT_DIM + # W_out, b_out
            GRU_INHERIT_DIM                                   # gamma_m
        )

    def load_flat(self, v: np.ndarray):
        i = 0
        def take(n):
            nonlocal i; r = v[i:i+n]; i += n; return r
        id_, hd = MOTOR_EMBED_DIM, MOTOR_GRU_DIM
        self.W_embed  = take(BN_PARAMS * id_).reshape(id_, BN_PARAMS); self.b_embed = take(id_)
        self.Wz = take(id_ * hd).reshape(hd, id_); self.Uz = take(hd * hd).reshape(hd, hd); self.bz = take(hd)
        self.Wr = take(id_ * hd).reshape(hd, id_); self.Ur = take(hd * hd).reshape(hd, hd); self.br = take(hd)
        self.Wh = take(id_ * hd).reshape(hd, id_); self.Uh = take(hd * hd).reshape(hd, hd); self.bh = take(hd)
        self.W_integ  = take(hd * MOTOR_INTEG_DIM).reshape(MOTOR_INTEG_DIM, hd); self.b_integ = take(MOTOR_INTEG_DIM)
        self.W_cortex = take(MOTOR_INTEG_DIM * MOTOR_CORTEX_DIM).reshape(MOTOR_CORTEX_DIM, MOTOR_INTEG_DIM)
        self.b_cortex = take(MOTOR_CORTEX_DIM)
        self.W_out    = take(MOTOR_CORTEX_DIM * MOTOR_OUTPUT_DIM).reshape(MOTOR_OUTPUT_DIM, MOTOR_CORTEX_DIM)
        self.b_out    = take(MOTOR_OUTPUT_DIM)
        self.gamma_m  = take(GRU_INHERIT_DIM)


# ================================================================
# RNNBottleneck（パイプライン型半双方向通信）
# ================================================================
class RNNBottleneck:
    """パイプライン型半双方向通信ボトルネック。（変更なし）"""

    PULSE_GEN_INTERVAL = TURN_FRAMES // PULSE_TOTAL

    # 表示専用タイマーの間隔: 5Hz = 12フレームに1スロット点灯
    DISPLAY_INTERVAL = TURN_FRAMES // PULSE_TOTAL  # = 12

    def __init__(self, sensory: SensoryNN, motor: MotorNN):
        self.sensory = sensory
        self.motor   = motor

        self._frame:         int = 0
        self._turn:          int = 0
        self._pulse_buffer:  list[list[int]] = []
        self._outgoing:      list[list[int]] = []
        self._consume_count: int = 0

        self._current_pulse: list[int] = [0, 0]
        self._pulse_history: list[list[int]] = []
        self._mode:          str = "listen"

        self.audio_enabled: bool = False
        self.converter = None
        self._last_phoneme: str = ""

        self._last_action: list[float] = [0.0, 0.5, 0.0]

        self.last_sensory_gru: list[float] = [0.0] * SENSORY_GRU_DIM
        self.last_motor_gru:   list[float] = [0.0] * MOTOR_GRU_DIM
        self.last_output:      list[float] = [0.5, 0.5, 0.0]

        # ---- 表示・音声専用（実際の消化パイプラインとは独立） ----
        # ターン境界でそのターンの全パルスをコピーし、5Hzで1個ずつ点灯・発音する
        self._display_queue:   list[list[int]] = []   # 今ターンの未表示パルス
        self._display_history: list[list[int]] = []   # 表示済みスロット（最大20個）
        self._display_phoneme: str = ""               # 現在表示中の音素文字
        self._display_frame:   int = 0                # 表示タイマー用フレームカウンタ

    def reset(self, prefill: bool = True):
        """エピソードをリセットする。

        prefill=True（デフォルト）の場合、感覚層をダミー入力で
        1ターン分（PULSE_TOTAL回）先行起動し、生成したパルスを
        運動NNのパルスバッファに事前充填する。
        これにより「最初の4秒だけγ頃り」問題を解消し、
        全ターンで同じパイプライン動作に統一する。
        """
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
        # 表示専用リセット
        self._display_queue   = []
        self._display_history = []
        self._display_phoneme = ""
        self._display_frame   = 0

        if prefill:
            # 感覚層をダミー入力でPULSE_TOTAL回先行起動し、
            # 生成したパルスを運動NNのパルスバッファに事前充填する。
            # ダミー入力: 全要素を中立値(0.5)に設定した初期状態を表現
            from config import SENSORY_INPUT_DIM, PULSE_TOTAL
            dummy_obs = [0.5] * SENSORY_INPUT_DIM
            pre_pulses = []
            for _ in range(PULSE_TOTAL):
                pulse = self.sensory.forward(dummy_obs)
                pre_pulses.append(pulse[:])
            # 運動NNのパルスバッファに事前充填
            self._pulse_buffer = pre_pulses
            # 感覚層の隠れ状態は先行起動後の状態を維持（リセット不要）
            # ターンカウンタは0のまま（正規パイプラインのターン、1から始まる）

    def step(self, obs: list[float]) -> list[float]:
        f = self._frame

        if f % self.PULSE_GEN_INTERVAL == 0:
            pulse = self.sensory.forward(obs)
            self._outgoing.append(pulse)
            self.last_sensory_gru = self.sensory.last_gru_act

        if f > 0 and f % TURN_FRAMES == 0:
            self._pulse_buffer.extend(self._outgoing)
            # 表示専用: そのターンの全パルスを表示キューにコピーし、スロットをリセット
            self._display_queue   = [p[:] for p in self._outgoing]
            self._display_history = []
            self._display_frame   = 0
            self._outgoing = []
            self._turn += 1
            self._consume_count = 0
            self._mode = "speak" if self._mode == "listen" else "listen"

        # 表示専用タイマー: 5Hz（12フレームに1個）でキューからスロットを点灯・発音
        if self._display_queue and self._display_frame % self.DISPLAY_INTERVAL == 0:
            disp_pulse = self._display_queue.pop(0)
            self._display_history.append(disp_pulse)
            if len(self._display_history) > PULSE_TOTAL:
                self._display_history.pop(0)
            bits2 = (disp_pulse[0] << 1) | disp_pulse[1]
            from config import PHONEME_TABLE
            self._display_phoneme = PHONEME_TABLE.get(bits2 & 0x3, "")
            # 音声出力（表示専用タイミングで発音）
            if self.audio_enabled and self.converter is not None:
                direction = 'S→M' if self._mode == 'listen' else 'M→S'
                try:
                    self.converter.play(bits2, direction=direction)
                except Exception:
                    pass
        self._display_frame += 1

        if f % PULSE_CONSUME_RATE == 0 and self._pulse_buffer:
            pulse = self._pulse_buffer.pop(0)
            self._current_pulse = pulse
            self._pulse_history.append(pulse[:])
            if len(self._pulse_history) > PULSE_TOTAL:
                self._pulse_history.pop(0)
            self._consume_count += 1

            bits2 = (pulse[0] << 1) | pulse[1]
            # 音声は表示専用タイマー側で出力するため、ここでは_last_phonemeの更新のみ
            from config import PHONEME_TABLE
            self._last_phoneme = PHONEME_TABLE.get(bits2 & 0x3, "")

            action = self.motor.forward(pulse)
            self._last_action = action
            self.last_motor_gru = self.motor.last_gru_act
            self.last_output    = self.motor.last_output_act
        # 注意: プリフィルにより_pulse_bufferは常に初期充填済みのため、
        # 旧実装の「ターン1のみデフォルト行動」フォールバックは不要

        self._frame += 1
        return self._last_action

    def on_pulse_emit(self, bits2: int) -> None:
        """パルス消化時（受信側）に呼ばれるフック。
        音声出力は表示専用タイマー側に移動したため、ここでは_last_phonemeの更新のみ。
        """
        from config import PHONEME_TABLE
        self._last_phoneme = PHONEME_TABLE.get(bits2 & 0x3, "")

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
        return self._consume_count, len(self._pulse_buffer)

    # ---- 表示専用getter ----
    def get_display_history(self) -> list[list[int]]:
        """表示専用スロット履歴（5Hzタイミングで埋まる、1ターンでリセット）。"""
        return self._display_history

    def get_display_phoneme(self) -> str:
        """現在表示中の音素文字（5Hzタイミングで更新）。"""
        return self._display_phoneme

    def get_display_progress(self) -> float:
        """表示タイマーの進捗（0.0～1.0）。スロットの埋まり具合。"""
        total = PULSE_TOTAL
        filled = len(self._display_history)
        return filled / total if total > 0 else 0.0

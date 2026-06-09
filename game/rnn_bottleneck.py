"""
rnn_bottleneck.py — RNNボトルネック本実装（バッファGRU挿入型・対称構造 v2）

設計方針:
  センサリーNN・モーターNNともに同じ対称構造。
  第三層FF → バッファGRU（横入り）→ 第二層（記憶GRU + 通常FF）→ 第一層FF → OUT

層構成（センサリー/モーター共通）:
  第三層FF(→24) → バッファGRU(12→5, パルス受信時のみ更新)
               → 記憶GRU(17→12) + 通常FF(12→16) → 第一層FF(28→24) → 出力

バッファGRUの特性:
  - 入力: 第三層バッファノード[12:24] (12次元)
  - 出力: buf_out (5次元)
  - パルス受信フレームのみ隠れ状態を更新（非受信フレームは凍結）
  - 記憶GRUへの入力として注入（通常ノード12 + buf_out5 = 17次元）

記憶GRUの隠れ状態分割:
  上位6次元: GA継承領域（γから復元）
  下位6次元: 非継承領域（ゼロ初期化）

センサリーNN パラメータ: 2,622次元
モーターNN パラメータ:  2,407次元
合計:                   5,029次元
"""
from __future__ import annotations
import numpy as np
from config import (
    SENSORY_INPUT_DIM, BN_PARAMS,
    L3_OUT_DIM, L3_NORMAL_DIM, L3_BUFFER_DIM,
    BUF_GRU_DIM, MEM_GRU_DIM, GRU_INHERIT_DIM, GRU_EPISODE_DIM,
    BYPASS_FF_DIM, L1_IN_DIM, L1_OUT_DIM,
    SENSORY_ENCODE_DIM, MOTOR_OUTPUT_DIM,
    TURN_FRAMES, PULSE_TOTAL, PULSE_CONSUME_RATE, PULSE_GEN_INTERVAL, PIPELINE_OFFSET,
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
    感覚NN。obs(12) → 2bitsパルス。v2仕様。

    obs(12)
      → 第三層FF(12→24, tanh)
          通常ノード[0:12] → 記憶GRU入力 + 通常FF入力
          バッファノード[12:24] → バッファGRU入力（パルス受信時のみ更新）
      → バッファGRU(12→5) ← パルス受信フレームのみ更新
      → 記憶GRU(17→12) ← 入力=concat(通常ノード[0:12], buf_out[0:5])
      → 通常FF(12→16, tanh) ← 入力=通常ノード[0:12]のみ
      → 第一層FF(28→24, tanh) ← 入力=concat(記憶GRU出力12, 通常FF出力16)
      → パルス符号化FF(24→2, tanh) → Step → 2bits

    GAが進化させる重み: 2,622次元
    """

    def __init__(self, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()
        s = 0.3

        # 第三層FF: 12→24
        self.W3 = rng.normal(0, s, (L3_OUT_DIM, SENSORY_INPUT_DIM))
        self.b3 = np.zeros(L3_OUT_DIM)

        # バッファGRU: 12→5
        id_b, hd_b = L3_BUFFER_DIM, BUF_GRU_DIM
        self.Wz_b = rng.normal(0, s, (hd_b, id_b)); self.Uz_b = rng.normal(0, s, (hd_b, hd_b)); self.bz_b = np.zeros(hd_b)
        self.Wr_b = rng.normal(0, s, (hd_b, id_b)); self.Ur_b = rng.normal(0, s, (hd_b, hd_b)); self.br_b = np.zeros(hd_b)
        self.Wh_b = rng.normal(0, s, (hd_b, id_b)); self.Uh_b = rng.normal(0, s, (hd_b, hd_b)); self.bh_b = np.zeros(hd_b)

        # 記憶GRU: 17→12 (入力=通常ノード12 + buf_out5)
        id_m, hd_m = L3_NORMAL_DIM + BUF_GRU_DIM, MEM_GRU_DIM
        self.Wz_m = rng.normal(0, s, (hd_m, id_m)); self.Uz_m = rng.normal(0, s, (hd_m, hd_m)); self.bz_m = np.zeros(hd_m)
        self.Wr_m = rng.normal(0, s, (hd_m, id_m)); self.Ur_m = rng.normal(0, s, (hd_m, hd_m)); self.br_m = np.zeros(hd_m)
        self.Wh_m = rng.normal(0, s, (hd_m, id_m)); self.Uh_m = rng.normal(0, s, (hd_m, hd_m)); self.bh_m = np.zeros(hd_m)

        # γ: 記憶GRU上位6次元の初期値
        self.gamma = np.zeros(GRU_INHERIT_DIM)

        # 通常FF: 12→16
        self.W_bypass = rng.normal(0, s, (BYPASS_FF_DIM, L3_NORMAL_DIM))
        self.b_bypass = np.zeros(BYPASS_FF_DIM)

        # 第一層FF（統合）: 28→24
        self.W1 = rng.normal(0, s, (L1_OUT_DIM, L1_IN_DIM))
        self.b1 = np.zeros(L1_OUT_DIM)

        # パルス符号化FF: 24→2
        self.W_encode = rng.normal(0, s, (SENSORY_ENCODE_DIM, L1_OUT_DIM))
        self.b_encode = np.zeros(SENSORY_ENCODE_DIM)

        # オンライン状態
        self._h_mem: np.ndarray = self._init_h_mem()
        self._h_buf: np.ndarray = np.zeros(BUF_GRU_DIM)
        self._buf_out: np.ndarray = np.zeros(BUF_GRU_DIM)

        # アクティベーション記録（可視化用）
        self.last_l3_act:      list[float] = [0.0] * L3_OUT_DIM
        self.last_buf_act:     list[float] = [0.0] * BUF_GRU_DIM
        self.last_buf_active:  bool = False
        self.last_gru_act:     list[float] = [0.0] * MEM_GRU_DIM
        self.last_bypass_act:  list[float] = [0.0] * BYPASS_FF_DIM
        self.last_integ_act:   list[float] = [0.0] * L1_OUT_DIM
        self.last_pulse:       list[int]   = [0, 0]
        # 互換性用エイリアス
        self.last_input_act  = self.last_l3_act
        self.last_cortex_act = self.last_l3_act

    def _init_h_mem(self) -> np.ndarray:
        h = np.zeros(MEM_GRU_DIM)
        h[:GRU_INHERIT_DIM] = self.gamma
        return h

    def reset(self):
        self._h_mem   = self._init_h_mem()
        self._h_buf   = np.zeros(BUF_GRU_DIM)
        self._buf_out = np.zeros(BUF_GRU_DIM)

    def forward(self, obs: list[float], is_pulse_frame: bool = False) -> list[int]:
        x = np.array(obs, dtype=np.float32)

        # 第三層FF
        x3 = np.tanh(self.W3 @ x + self.b3)
        self.last_l3_act = x3.tolist()
        x3_normal = x3[:L3_NORMAL_DIM]    # [0:12]
        x3_buffer = x3[L3_NORMAL_DIM:]    # [12:24]

        # バッファGRU（パルス受信フレームのみ更新）
        if is_pulse_frame:
            self._h_buf = gru_step(
                x3_buffer, self._h_buf,
                self.Wz_b, self.Wr_b, self.Wh_b,
                self.Uz_b, self.Ur_b, self.Uh_b,
                self.bz_b, self.br_b, self.bh_b,
            )
            self._buf_out = self._h_buf.copy()
        self.last_buf_act    = self._buf_out.tolist()
        self.last_buf_active = is_pulse_frame

        # 記憶GRU: 入力=concat(通常ノード[0:12], buf_out[0:5])
        x_mem_in = np.concatenate([x3_normal, self._buf_out])
        self._h_mem = gru_step(
            x_mem_in, self._h_mem,
            self.Wz_m, self.Wr_m, self.Wh_m,
            self.Uz_m, self.Ur_m, self.Uh_m,
            self.bz_m, self.br_m, self.bh_m,
        )
        self.last_gru_act = self._h_mem.tolist()

        # 通常FF
        bypass = np.tanh(self.W_bypass @ x3_normal + self.b_bypass)
        self.last_bypass_act = bypass.tolist()

        # 第一層FF（統合）
        x1_in = np.concatenate([self._h_mem, bypass])
        x1 = np.tanh(self.W1 @ x1_in + self.b1)
        self.last_integ_act = x1.tolist()

        # パルス符号化FF + Step
        enc = np.tanh(self.W_encode @ x1 + self.b_encode)
        bits = (enc >= 0).astype(int)
        pulse = [int(bits[0]), int(bits[1])]
        self.last_pulse = pulse
        return pulse

    def encode_pulse(self, obs: list[float], is_pulse_frame: bool = False) -> int:
        p = self.forward(obs, is_pulse_frame)
        return (p[0] << 1) | p[1]

    # ----------------------------------------------------------------
    def flat(self) -> np.ndarray:
        return np.concatenate([
            self.W3.ravel(), self.b3,
            self.Wz_b.ravel(), self.Uz_b.ravel(), self.bz_b,
            self.Wr_b.ravel(), self.Ur_b.ravel(), self.br_b,
            self.Wh_b.ravel(), self.Uh_b.ravel(), self.bh_b,
            self.Wz_m.ravel(), self.Uz_m.ravel(), self.bz_m,
            self.Wr_m.ravel(), self.Ur_m.ravel(), self.br_m,
            self.Wh_m.ravel(), self.Uh_m.ravel(), self.bh_m,
            self.gamma,
            self.W_bypass.ravel(), self.b_bypass,
            self.W1.ravel(), self.b1,
            self.W_encode.ravel(), self.b_encode,
        ])

    @staticmethod
    def flat_size() -> int:
        id_b, hd_b = L3_BUFFER_DIM, BUF_GRU_DIM
        id_m, hd_m = L3_NORMAL_DIM + BUF_GRU_DIM, MEM_GRU_DIM
        return (
            SENSORY_INPUT_DIM * L3_OUT_DIM + L3_OUT_DIM +          # W3, b3
            3 * (id_b * hd_b + hd_b * hd_b + hd_b) +              # バッファGRU 3ゲート
            3 * (id_m * hd_m + hd_m * hd_m + hd_m) +              # 記憶GRU 3ゲート
            GRU_INHERIT_DIM +                                        # gamma
            L3_NORMAL_DIM * BYPASS_FF_DIM + BYPASS_FF_DIM +        # W_bypass, b_bypass
            L1_IN_DIM * L1_OUT_DIM + L1_OUT_DIM +                  # W1, b1
            L1_OUT_DIM * SENSORY_ENCODE_DIM + SENSORY_ENCODE_DIM   # W_encode, b_encode
        )

    def load_flat(self, v: np.ndarray):
        i = 0
        def take(n):
            nonlocal i; r = v[i:i+n]; i += n; return r
        id_b, hd_b = L3_BUFFER_DIM, BUF_GRU_DIM
        id_m, hd_m = L3_NORMAL_DIM + BUF_GRU_DIM, MEM_GRU_DIM

        self.W3 = take(SENSORY_INPUT_DIM * L3_OUT_DIM).reshape(L3_OUT_DIM, SENSORY_INPUT_DIM)
        self.b3 = take(L3_OUT_DIM)

        self.Wz_b = take(id_b * hd_b).reshape(hd_b, id_b); self.Uz_b = take(hd_b * hd_b).reshape(hd_b, hd_b); self.bz_b = take(hd_b)
        self.Wr_b = take(id_b * hd_b).reshape(hd_b, id_b); self.Ur_b = take(hd_b * hd_b).reshape(hd_b, hd_b); self.br_b = take(hd_b)
        self.Wh_b = take(id_b * hd_b).reshape(hd_b, id_b); self.Uh_b = take(hd_b * hd_b).reshape(hd_b, hd_b); self.bh_b = take(hd_b)

        self.Wz_m = take(id_m * hd_m).reshape(hd_m, id_m); self.Uz_m = take(hd_m * hd_m).reshape(hd_m, hd_m); self.bz_m = take(hd_m)
        self.Wr_m = take(id_m * hd_m).reshape(hd_m, id_m); self.Ur_m = take(hd_m * hd_m).reshape(hd_m, hd_m); self.br_m = take(hd_m)
        self.Wh_m = take(id_m * hd_m).reshape(hd_m, id_m); self.Uh_m = take(hd_m * hd_m).reshape(hd_m, hd_m); self.bh_m = take(hd_m)

        self.gamma = take(GRU_INHERIT_DIM)

        self.W_bypass = take(L3_NORMAL_DIM * BYPASS_FF_DIM).reshape(BYPASS_FF_DIM, L3_NORMAL_DIM)
        self.b_bypass = take(BYPASS_FF_DIM)

        self.W1 = take(L1_IN_DIM * L1_OUT_DIM).reshape(L1_OUT_DIM, L1_IN_DIM)
        self.b1 = take(L1_OUT_DIM)

        self.W_encode = take(L1_OUT_DIM * SENSORY_ENCODE_DIM).reshape(SENSORY_ENCODE_DIM, L1_OUT_DIM)
        self.b_encode = take(SENSORY_ENCODE_DIM)


# ================================================================
# MotorNN
# ================================================================
class MotorNN:
    """
    運動NN。2bitsパルス → [Accel, Steer, Brake]。v2仕様。
    センサリーNNと対称構造。

    pulse(2)
      → 第三層FF(2→24, tanh)
          通常ノード[0:12] → 記憶GRU入力 + 通常FF入力
          バッファノード[12:24] → バッファGRU入力（パルス受信時のみ更新）
      → バッファGRU(12→5) ← パルス受信フレームのみ更新
      → 記憶GRU(17→12) ← 入力=concat(通常ノード[0:12], buf_out[0:5])
      → 通常FF(12→16, tanh)
      → 第一層FF(28→24, tanh)
      → 出力FF(24→3, sigmoid) → Accel/Steer/Brake

    GAが進化させる重み: 2,407次元
    """

    def __init__(self, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()
        s = 0.3

        # 第三層FF: 2→24
        self.W3 = rng.normal(0, s, (L3_OUT_DIM, BN_PARAMS))
        self.b3 = np.zeros(L3_OUT_DIM)

        # バッファGRU: 12→5
        id_b, hd_b = L3_BUFFER_DIM, BUF_GRU_DIM
        self.Wz_b = rng.normal(0, s, (hd_b, id_b)); self.Uz_b = rng.normal(0, s, (hd_b, hd_b)); self.bz_b = np.zeros(hd_b)
        self.Wr_b = rng.normal(0, s, (hd_b, id_b)); self.Ur_b = rng.normal(0, s, (hd_b, hd_b)); self.br_b = np.zeros(hd_b)
        self.Wh_b = rng.normal(0, s, (hd_b, id_b)); self.Uh_b = rng.normal(0, s, (hd_b, hd_b)); self.bh_b = np.zeros(hd_b)

        # 記憶GRU: 17→12
        id_m, hd_m = L3_NORMAL_DIM + BUF_GRU_DIM, MEM_GRU_DIM
        self.Wz_m = rng.normal(0, s, (hd_m, id_m)); self.Uz_m = rng.normal(0, s, (hd_m, hd_m)); self.bz_m = np.zeros(hd_m)
        self.Wr_m = rng.normal(0, s, (hd_m, id_m)); self.Ur_m = rng.normal(0, s, (hd_m, hd_m)); self.br_m = np.zeros(hd_m)
        self.Wh_m = rng.normal(0, s, (hd_m, id_m)); self.Uh_m = rng.normal(0, s, (hd_m, hd_m)); self.bh_m = np.zeros(hd_m)

        # γ
        self.gamma = np.zeros(GRU_INHERIT_DIM)

        # 通常FF: 12→16
        self.W_bypass = rng.normal(0, s, (BYPASS_FF_DIM, L3_NORMAL_DIM))
        self.b_bypass = np.zeros(BYPASS_FF_DIM)

        # 第一層FF（統合）: 28→24
        self.W1 = rng.normal(0, s, (L1_OUT_DIM, L1_IN_DIM))
        self.b1 = np.zeros(L1_OUT_DIM)

        # 出力FF: 24→3
        self.W_out = rng.normal(0, s, (MOTOR_OUTPUT_DIM, L1_OUT_DIM))
        self.b_out = np.zeros(MOTOR_OUTPUT_DIM)

        # オンライン状態
        self._h_mem: np.ndarray = self._init_h_mem()
        self._h_buf: np.ndarray = np.zeros(BUF_GRU_DIM)
        self._buf_out: np.ndarray = np.zeros(BUF_GRU_DIM)

        # アクティベーション記録
        self.last_l3_act:      list[float] = [0.0] * L3_OUT_DIM
        self.last_buf_act:     list[float] = [0.0] * BUF_GRU_DIM
        self.last_buf_active:  bool = False
        self.last_gru_act:     list[float] = [0.0] * MEM_GRU_DIM
        self.last_bypass_act:  list[float] = [0.0] * BYPASS_FF_DIM
        self.last_integ_act:   list[float] = [0.0] * L1_OUT_DIM
        self.last_output_act:  list[float] = [0.5, 0.5, 0.0]
        # 互換性用エイリアス
        self.last_embed_act  = self.last_l3_act
        self.last_cortex_act = self.last_integ_act

    def _init_h_mem(self) -> np.ndarray:
        h = np.zeros(MEM_GRU_DIM)
        h[:GRU_INHERIT_DIM] = self.gamma
        return h

    def reset(self):
        self._h_mem   = self._init_h_mem()
        self._h_buf   = np.zeros(BUF_GRU_DIM)
        self._buf_out = np.zeros(BUF_GRU_DIM)

    def forward(self, pulse: list[int], is_pulse_frame: bool = False) -> list[float]:
        x = np.array(pulse, dtype=np.float32)

        # 第三層FF
        x3 = np.tanh(self.W3 @ x + self.b3)
        self.last_l3_act = x3.tolist()
        x3_normal = x3[:L3_NORMAL_DIM]
        x3_buffer = x3[L3_NORMAL_DIM:]

        # バッファGRU
        if is_pulse_frame:
            self._h_buf = gru_step(
                x3_buffer, self._h_buf,
                self.Wz_b, self.Wr_b, self.Wh_b,
                self.Uz_b, self.Ur_b, self.Uh_b,
                self.bz_b, self.br_b, self.bh_b,
            )
            self._buf_out = self._h_buf.copy()
        self.last_buf_act    = self._buf_out.tolist()
        self.last_buf_active = is_pulse_frame

        # 記憶GRU
        x_mem_in = np.concatenate([x3_normal, self._buf_out])
        self._h_mem = gru_step(
            x_mem_in, self._h_mem,
            self.Wz_m, self.Wr_m, self.Wh_m,
            self.Uz_m, self.Ur_m, self.Uh_m,
            self.bz_m, self.br_m, self.bh_m,
        )
        self.last_gru_act = self._h_mem.tolist()

        # 通常FF
        bypass = np.tanh(self.W_bypass @ x3_normal + self.b_bypass)
        self.last_bypass_act = bypass.tolist()

        # 第一層FF（統合）
        x1_in = np.concatenate([self._h_mem, bypass])
        x1 = np.tanh(self.W1 @ x1_in + self.b1)
        self.last_integ_act = x1.tolist()

        # 出力FF (sigmoid)
        def sigmoid(v): return 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))
        out = sigmoid(self.W_out @ x1 + self.b_out)
        self.last_output_act = out.tolist()
        return out.tolist()

    # ----------------------------------------------------------------
    def flat(self) -> np.ndarray:
        return np.concatenate([
            self.W3.ravel(), self.b3,
            self.Wz_b.ravel(), self.Uz_b.ravel(), self.bz_b,
            self.Wr_b.ravel(), self.Ur_b.ravel(), self.br_b,
            self.Wh_b.ravel(), self.Uh_b.ravel(), self.bh_b,
            self.Wz_m.ravel(), self.Uz_m.ravel(), self.bz_m,
            self.Wr_m.ravel(), self.Ur_m.ravel(), self.br_m,
            self.Wh_m.ravel(), self.Uh_m.ravel(), self.bh_m,
            self.gamma,
            self.W_bypass.ravel(), self.b_bypass,
            self.W1.ravel(), self.b1,
            self.W_out.ravel(), self.b_out,
        ])

    @staticmethod
    def flat_size() -> int:
        id_b, hd_b = L3_BUFFER_DIM, BUF_GRU_DIM
        id_m, hd_m = L3_NORMAL_DIM + BUF_GRU_DIM, MEM_GRU_DIM
        return (
            BN_PARAMS * L3_OUT_DIM + L3_OUT_DIM +                  # W3, b3
            3 * (id_b * hd_b + hd_b * hd_b + hd_b) +              # バッファGRU 3ゲート
            3 * (id_m * hd_m + hd_m * hd_m + hd_m) +              # 記憶GRU 3ゲート
            GRU_INHERIT_DIM +                                        # gamma
            L3_NORMAL_DIM * BYPASS_FF_DIM + BYPASS_FF_DIM +        # W_bypass, b_bypass
            L1_IN_DIM * L1_OUT_DIM + L1_OUT_DIM +                  # W1, b1
            L1_OUT_DIM * MOTOR_OUTPUT_DIM + MOTOR_OUTPUT_DIM       # W_out, b_out
        )

    def load_flat(self, v: np.ndarray):
        i = 0
        def take(n):
            nonlocal i; r = v[i:i+n]; i += n; return r
        id_b, hd_b = L3_BUFFER_DIM, BUF_GRU_DIM
        id_m, hd_m = L3_NORMAL_DIM + BUF_GRU_DIM, MEM_GRU_DIM

        self.W3 = take(BN_PARAMS * L3_OUT_DIM).reshape(L3_OUT_DIM, BN_PARAMS)
        self.b3 = take(L3_OUT_DIM)

        self.Wz_b = take(id_b * hd_b).reshape(hd_b, id_b); self.Uz_b = take(hd_b * hd_b).reshape(hd_b, hd_b); self.bz_b = take(hd_b)
        self.Wr_b = take(id_b * hd_b).reshape(hd_b, id_b); self.Ur_b = take(hd_b * hd_b).reshape(hd_b, hd_b); self.br_b = take(hd_b)
        self.Wh_b = take(id_b * hd_b).reshape(hd_b, id_b); self.Uh_b = take(hd_b * hd_b).reshape(hd_b, hd_b); self.bh_b = take(hd_b)

        self.Wz_m = take(id_m * hd_m).reshape(hd_m, id_m); self.Uz_m = take(hd_m * hd_m).reshape(hd_m, hd_m); self.bz_m = take(hd_m)
        self.Wr_m = take(id_m * hd_m).reshape(hd_m, id_m); self.Ur_m = take(hd_m * hd_m).reshape(hd_m, hd_m); self.br_m = take(hd_m)
        self.Wh_m = take(id_m * hd_m).reshape(hd_m, id_m); self.Uh_m = take(hd_m * hd_m).reshape(hd_m, hd_m); self.bh_m = take(hd_m)

        self.gamma = take(GRU_INHERIT_DIM)

        self.W_bypass = take(L3_NORMAL_DIM * BYPASS_FF_DIM).reshape(BYPASS_FF_DIM, L3_NORMAL_DIM)
        self.b_bypass = take(BYPASS_FF_DIM)

        self.W1 = take(L1_IN_DIM * L1_OUT_DIM).reshape(L1_OUT_DIM, L1_IN_DIM)
        self.b1 = take(L1_OUT_DIM)

        self.W_out = take(L1_OUT_DIM * MOTOR_OUTPUT_DIM).reshape(MOTOR_OUTPUT_DIM, L1_OUT_DIM)
        self.b_out = take(MOTOR_OUTPUT_DIM)


# ================================================================
# RNNBottleneck（パイプライン型半双方向通信）
# ================================================================
class RNNBottleneck:
    """リアルタイム半双方向通信ボトルネック。

    生成と消化を同周期・同位相に同期させることで、
    感覚の鮮度を遅延なく運動側に届ける。
    遅延: 最大PULSE_GEN_INTERVALフレーム（6フレーム）固定。
    ターン境界は方向切替のみ。
    """

    PULSE_GEN_INTERVAL = PULSE_GEN_INTERVAL   # configから取得 = 6
    DISPLAY_INTERVAL   = PULSE_GEN_INTERVAL   # 表示も生成と同周期 = 6（10Hz）

    def __init__(self, sensory: SensoryNN, motor: MotorNN):
        self.sensory = sensory
        self.motor   = motor

        self._frame:         int = 0
        self._turn:          int = 0
        self._pulse_buffer:  list[list[int]] = []
        self._consume_count: int = 0

        self._current_pulse: list[int] = [0, 0]
        self._pulse_history: list[list[int]] = []
        self._mode:          str = "listen"

        self.audio_enabled: bool = False
        self.converter = None
        self._last_phoneme: str = ""

        self._last_action: list[float] = [0.0, 0.5, 0.0]

        self.last_sensory_gru: list[float] = [0.0] * MEM_GRU_DIM
        self.last_motor_gru:   list[float] = [0.0] * MEM_GRU_DIM
        self.last_output:      list[float] = [0.5, 0.5, 0.0]

        # 表示・音声専用
        self._display_queue:   list[list[int]] = []
        self._display_history: list[list[int]] = []
        self._display_phoneme: str = ""
        self._display_frame:   int = 0

    def reset(self, prefill: bool = True):
        """エピソードをリセットする。
        prefill=Trueの場合、ダミー入力で1発生成してパイプラインを起動する。
        """
        self._frame         = 0
        self._turn          = 0
        self._pulse_buffer  = []
        self._consume_count = 0
        self._current_pulse = [0, 0]
        self._pulse_history = []
        self._mode          = "listen"
        self._last_action   = [0.0, 0.5, 0.0]
        self.sensory.reset()
        self.motor.reset()
        self._display_queue   = []
        self._display_history = []
        self._display_phoneme = ""
        self._display_frame   = 0

        if prefill:
            # ダミー入力で1発生成してパイプラインを起動する。
            # リアルタイム方式では1発で十分（消化は次の生成フレームまで待つ）。
            dummy_obs = [0.5] * SENSORY_INPUT_DIM
            pulse = self.sensory.forward(dummy_obs, is_pulse_frame=True)
            self._pulse_buffer  = [pulse[:]]
            self._display_queue = [pulse[:]]

    def step(self, obs: list[float]) -> list[float]:
        """リアルタイム同期方式の1ステップ処理。

        処理順序:
          1. 生成（6フレームごと）: obsを感覚NNに通し、即座に_pulse_bufferへ
          2. 表示タイマー（6フレームごと）: スロット点灯・音声出力
          3. 消化（6フレームごと）: バッファから1発取り出し運動NNへ
          4. ターン境界（120フレーム）: 方向切替のみ（バッファ継続）
        """
        f = self._frame
        from config import PHONEME_TABLE

        # --- 1. 生成: 6フレームごとに即座に_pulse_bufferへ追加 ---
        is_gen_frame = (f % self.PULSE_GEN_INTERVAL == 0)
        if is_gen_frame:
            pulse = self.sensory.forward(obs, is_pulse_frame=True)
            self._pulse_buffer.append(pulse[:])
            self.last_sensory_gru = self.sensory.last_gru_act
        else:
            # 非生成フレームでも感覚NNは常にobsを処理する（GRU隔れ状態を更新）
            self.sensory.forward(obs, is_pulse_frame=False)
            self.last_sensory_gru = self.sensory.last_gru_act

        # --- 2. 表示タイマー: 生成と同周期でスロット点灯・音声出力 ---
        if self._display_queue and self._display_frame % self.DISPLAY_INTERVAL == 0:
            disp_pulse = self._display_queue.pop(0)
            self._display_history.append(disp_pulse)
            if len(self._display_history) > PULSE_TOTAL:
                self._display_history.pop(0)
            bits2 = (disp_pulse[0] << 1) | disp_pulse[1]
            self._display_phoneme = PHONEME_TABLE.get(bits2 & 0x3, "")
            if self.audio_enabled and self.converter is not None:
                direction = 'S→M' if self._mode == 'listen' else 'M→S'
                try:
                    self.converter.play(bits2, direction=direction)
                except Exception:
                    pass
        # 表示キューに生成パルスを追加（生成と同フレームで表示キューにも追加）
        if is_gen_frame:
            self._display_queue.append(self._pulse_buffer[-1][:])
        self._display_frame += 1

        # --- 3. 消化: PULSE_CONSUME_RATE（6フレーム）ごとにバッファから1発取り出し ---
        is_consume_frame = (f % PULSE_CONSUME_RATE == 0 and bool(self._pulse_buffer))
        if is_consume_frame:
            pulse = self._pulse_buffer.pop(0)
            self._current_pulse = pulse
            self._pulse_history.append(pulse[:])
            if len(self._pulse_history) > PULSE_TOTAL:
                self._pulse_history.pop(0)
            self._consume_count += 1

            bits2 = (pulse[0] << 1) | pulse[1]
            self._last_phoneme = PHONEME_TABLE.get(bits2 & 0x3, "")

            action = self.motor.forward(pulse, is_pulse_frame=True)
            self._last_action = action
            self.last_motor_gru = self.motor.last_gru_act
            self.last_output    = self.motor.last_output_act
        else:
            # 非消化フレームでも運動NNは常に実行（GRU隔れ状態を更新）
            if self._current_pulse:
                self.motor.forward(self._current_pulse, is_pulse_frame=False)
                self.last_motor_gru = self.motor.last_gru_act
                self.last_output    = self.motor.last_output_act

        # --- 4. ターン境界: 方向切替のみ（バッファは継続） ---
        if f > 0 and f % TURN_FRAMES == 0:
            self._turn += 1
            self._consume_count = 0
            self._mode = "speak" if self._mode == "listen" else "listen"
            # _pulse_bufferはリセットしない（パイプライン継続）

        self._frame += 1
        return self._last_action

    def on_pulse_emit(self, bits2: int) -> None:
        from config import PHONEME_TABLE
        self._last_phoneme = PHONEME_TABLE.get(bits2 & 0x3, "")

    def enable_audio(self) -> None:
        if self.audio_enabled:
            return
        try:
            from game.phoneme import PhonemeConverter
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

    def get_display_history(self) -> list[list[int]]:
        return self._display_history

    def get_display_phoneme(self) -> str:
        return self._display_phoneme

    def get_display_progress(self) -> float:
        total = PULSE_TOTAL
        filled = len(self._display_history)
        return filled / total if total > 0 else 0.0

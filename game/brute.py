"""
brute.py — BRUTE（野人）ニューラルネットワーク

BRUTEはキノコのバイオーム・腐敗を弁別視野で感知し、
行動を出力しながらSAGEへパルスを送信する運動系 NN。

obs構成（11次元）:
  [0:6]  弁別視野マスク済み6次元
         [0:3]=バイオーム / [3]=0固定 / [4]=0固定 / [5]=腐敗
  [6:11] 視覚レイ5本（±45度・正規化 0～1）
  [9:11] 受信パルス（2bits: 0.0 or 1.0）

パラメータ数: 2,673次元（入力次元変わらず）
"""
from __future__ import annotations
import numpy as np
from config import (
    BRUTE_OBS_DIM, BRUTE_L3_OUT, BRUTE_L3_NORMAL, BRUTE_L3_BUF,
    BRUTE_BUF_DIM, BRUTE_MEM_DIM, BRUTE_MEM_INHERIT,
    BRUTE_BYPASS_OUT, BRUTE_L1_IN, BRUTE_L1_OUT,
    BRUTE_ACTION_DIM, BRUTE_ENCODE_DIM,
)


class BruteNN:
    """BRUTE（野人）NN。obs(11) → (action[3], pulse_int)。"""

    def __init__(self, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()
        s = 0.3

        # 第三層FF: 11→24
        self.W3 = rng.normal(0, s, (BRUTE_L3_OUT, BRUTE_OBS_DIM))
        self.b3 = np.zeros(BRUTE_L3_OUT)

        # バッファGRU: 12→5
        id_b, hd_b = BRUTE_L3_BUF, BRUTE_BUF_DIM
        self.Wz_b = rng.normal(0, s, (hd_b, id_b)); self.Uz_b = rng.normal(0, s, (hd_b, hd_b)); self.bz_b = np.zeros(hd_b)
        self.Wr_b = rng.normal(0, s, (hd_b, id_b)); self.Ur_b = rng.normal(0, s, (hd_b, hd_b)); self.br_b = np.zeros(hd_b)
        self.Wh_b = rng.normal(0, s, (hd_b, id_b)); self.Uh_b = rng.normal(0, s, (hd_b, hd_b)); self.bh_b = np.zeros(hd_b)

        # 記憶GRU: 17→12
        id_m, hd_m = BRUTE_L3_NORMAL + BRUTE_BUF_DIM, BRUTE_MEM_DIM
        self.Wz_m = rng.normal(0, s, (hd_m, id_m)); self.Uz_m = rng.normal(0, s, (hd_m, hd_m)); self.bz_m = np.zeros(hd_m)
        self.Wr_m = rng.normal(0, s, (hd_m, id_m)); self.Ur_m = rng.normal(0, s, (hd_m, hd_m)); self.br_m = np.zeros(hd_m)
        self.Wh_m = rng.normal(0, s, (hd_m, id_m)); self.Uh_m = rng.normal(0, s, (hd_m, hd_m)); self.bh_m = np.zeros(hd_m)

        # γ
        self.gamma = np.zeros(BRUTE_MEM_INHERIT)

        # 通常FF: 12→16
        self.W_bypass = rng.normal(0, s, (BRUTE_BYPASS_OUT, BRUTE_L3_NORMAL))
        self.b_bypass = np.zeros(BRUTE_BYPASS_OUT)

        # 第一層FF（統合）: 28→24
        self.W1 = rng.normal(0, s, (BRUTE_L1_OUT, BRUTE_L1_IN))
        self.b1 = np.zeros(BRUTE_L1_OUT)

        # 行動出力FF: 24→3 (sigmoid)
        self.W_act = rng.normal(0, s, (BRUTE_ACTION_DIM, BRUTE_L1_OUT))
        self.b_act = np.zeros(BRUTE_ACTION_DIM)

        # パルス符号化FF: 24→2
        self.W_enc = rng.normal(0, s, (BRUTE_ENCODE_DIM, BRUTE_L1_OUT))
        self.b_enc = np.zeros(BRUTE_ENCODE_DIM)

        # オンライン状態
        self.h_mem:   np.ndarray = self._init_h_mem()
        self.h_buf:   np.ndarray = np.zeros(BRUTE_BUF_DIM)
        self.buf_out: np.ndarray = np.zeros(BRUTE_BUF_DIM)

        # アクティベーション記録（可視化用）
        self.last_l3_act:     list[float] = [0.0] * BRUTE_L3_OUT
        self.last_buf_act:    list[float] = [0.0] * BRUTE_BUF_DIM
        self.last_buf_active: bool = False
        self.last_gru_act:    list[float] = [0.0] * BRUTE_MEM_DIM
        self.last_bypass_act: list[float] = [0.0] * BRUTE_BYPASS_OUT
        self.last_integ_act:  list[float] = [0.0] * BRUTE_L1_OUT
        self.last_output_act: list[float] = [0.5, 0.5, 0.0]
        self.last_pulse:      int = 0

    def _init_h_mem(self) -> np.ndarray:
        h = np.zeros(BRUTE_MEM_DIM)
        h[:BRUTE_MEM_INHERIT] = self.gamma
        return h

    def reset_episode(self):
        self.h_mem   = self._init_h_mem()
        self.h_buf   = np.zeros(BRUTE_BUF_DIM)
        self.buf_out = np.zeros(BRUTE_BUF_DIM)

    @staticmethod
    def _gru(x: np.ndarray, h: np.ndarray, W: dict) -> np.ndarray:
        def sig(v): return 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))
        z = sig(W['Wz'] @ x + W['Uz'] @ h + W['bz'])
        r = sig(W['Wr'] @ x + W['Ur'] @ h + W['br'])
        n = np.tanh(W['Wh'] @ x + W['Uh'] @ (r * h) + W['bh'])
        return (1 - z) * h + z * n

    def _W_buf(self) -> dict:
        return {'Wz': self.Wz_b, 'Uz': self.Uz_b, 'bz': self.bz_b,
                'Wr': self.Wr_b, 'Ur': self.Ur_b, 'br': self.br_b,
                'Wh': self.Wh_b, 'Uh': self.Uh_b, 'bh': self.bh_b}

    def _W_mem(self) -> dict:
        return {'Wz': self.Wz_m, 'Uz': self.Uz_m, 'bz': self.bz_m,
                'Wr': self.Wr_m, 'Ur': self.Ur_m, 'br': self.br_m,
                'Wh': self.Wh_m, 'Uh': self.Uh_m, 'bh': self.bh_m}

    def forward(self, obs: np.ndarray, is_pulse_frame: bool = False) -> tuple[np.ndarray, int]:
        """obs(11) → (action[3], pulse_int)"""
        # 第三層FF
        x3 = np.tanh(self.W3 @ obs + self.b3)
        self.last_l3_act = x3.tolist()
        x3_normal = x3[:BRUTE_L3_NORMAL]
        x3_buf    = x3[BRUTE_L3_NORMAL:]

        # バッファGRU
        if is_pulse_frame:
            self.h_buf   = self._gru(x3_buf, self.h_buf, self._W_buf())
            self.buf_out = self.h_buf.copy()
        self.last_buf_act    = self.buf_out.tolist()
        self.last_buf_active = is_pulse_frame

        # 記憶GRU
        x_mem = np.concatenate([x3_normal, self.buf_out])
        self.h_mem = self._gru(x_mem, self.h_mem, self._W_mem())
        self.last_gru_act = self.h_mem.tolist()

        # 通常FF
        x_bp = np.tanh(self.W_bypass @ x3_normal + self.b_bypass)
        self.last_bypass_act = x_bp.tolist()

        # 第一層FF（統合）
        x1 = np.tanh(self.W1 @ np.concatenate([self.h_mem, x_bp]) + self.b1)
        self.last_integ_act = x1.tolist()

        # 行動出力FF (sigmoid)
        def sig(v): return 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))
        action = sig(self.W_act @ x1 + self.b_act)
        self.last_output_act = action.tolist()

        # パルス符号化FF + Step
        raw  = np.tanh(self.W_enc @ x1 + self.b_enc)
        bits = (raw >= 0).astype(int)
        pulse = int((bits[0] << 1) | bits[1])
        self.last_pulse = pulse

        return action, pulse

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
            self.W_act.ravel(), self.b_act,
            self.W_enc.ravel(), self.b_enc,
        ])

    @staticmethod
    def param_count() -> int:
        id_b, hd_b = BRUTE_L3_BUF, BRUTE_BUF_DIM
        id_m, hd_m = BRUTE_L3_NORMAL + BRUTE_BUF_DIM, BRUTE_MEM_DIM
        return (
            BRUTE_OBS_DIM * BRUTE_L3_OUT + BRUTE_L3_OUT +
            3 * (id_b * hd_b + hd_b * hd_b + hd_b) +
            3 * (id_m * hd_m + hd_m * hd_m + hd_m) +
            BRUTE_MEM_INHERIT +
            BRUTE_L3_NORMAL * BRUTE_BYPASS_OUT + BRUTE_BYPASS_OUT +
            BRUTE_L1_IN * BRUTE_L1_OUT + BRUTE_L1_OUT +
            BRUTE_L1_OUT * BRUTE_ACTION_DIM + BRUTE_ACTION_DIM +
            BRUTE_L1_OUT * BRUTE_ENCODE_DIM + BRUTE_ENCODE_DIM
        )

    def load_flat(self, v: np.ndarray):
        i = 0
        def take(n):
            nonlocal i; r = v[i:i+n]; i += n; return r
        id_b, hd_b = BRUTE_L3_BUF, BRUTE_BUF_DIM
        id_m, hd_m = BRUTE_L3_NORMAL + BRUTE_BUF_DIM, BRUTE_MEM_DIM

        self.W3 = take(BRUTE_OBS_DIM * BRUTE_L3_OUT).reshape(BRUTE_L3_OUT, BRUTE_OBS_DIM)
        self.b3 = take(BRUTE_L3_OUT)

        self.Wz_b = take(id_b * hd_b).reshape(hd_b, id_b); self.Uz_b = take(hd_b * hd_b).reshape(hd_b, hd_b); self.bz_b = take(hd_b)
        self.Wr_b = take(id_b * hd_b).reshape(hd_b, id_b); self.Ur_b = take(hd_b * hd_b).reshape(hd_b, hd_b); self.br_b = take(hd_b)
        self.Wh_b = take(id_b * hd_b).reshape(hd_b, id_b); self.Uh_b = take(hd_b * hd_b).reshape(hd_b, hd_b); self.bh_b = take(hd_b)

        self.Wz_m = take(id_m * hd_m).reshape(hd_m, id_m); self.Uz_m = take(hd_m * hd_m).reshape(hd_m, hd_m); self.bz_m = take(hd_m)
        self.Wr_m = take(id_m * hd_m).reshape(hd_m, id_m); self.Ur_m = take(hd_m * hd_m).reshape(hd_m, hd_m); self.br_m = take(hd_m)
        self.Wh_m = take(id_m * hd_m).reshape(hd_m, id_m); self.Uh_m = take(hd_m * hd_m).reshape(hd_m, hd_m); self.bh_m = take(hd_m)

        self.gamma = take(BRUTE_MEM_INHERIT)

        self.W_bypass = take(BRUTE_L3_NORMAL * BRUTE_BYPASS_OUT).reshape(BRUTE_BYPASS_OUT, BRUTE_L3_NORMAL)
        self.b_bypass = take(BRUTE_BYPASS_OUT)

        self.W1 = take(BRUTE_L1_IN * BRUTE_L1_OUT).reshape(BRUTE_L1_OUT, BRUTE_L1_IN)
        self.b1 = take(BRUTE_L1_OUT)

        self.W_act = take(BRUTE_L1_OUT * BRUTE_ACTION_DIM).reshape(BRUTE_ACTION_DIM, BRUTE_L1_OUT)
        self.b_act = take(BRUTE_ACTION_DIM)

        self.W_enc = take(BRUTE_L1_OUT * BRUTE_ENCODE_DIM).reshape(BRUTE_ENCODE_DIM, BRUTE_L1_OUT)
        self.b_enc = take(BRUTE_ENCODE_DIM)

        self.h_mem = self._init_h_mem()
        self.h_buf = np.zeros(BRUTE_BUF_DIM)
        self.buf_out = np.zeros(BRUTE_BUF_DIM)

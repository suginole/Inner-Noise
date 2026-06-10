"""
brute.py — BRUTE（野人）ニューラルネットワーク

obs構成（11次元）:
  [0:6]  弁別視野マスク済み
          [0:3]=バイオームone-hot
          [3]=0固定, [4]=0固定
          [5]=腐敗（0.0 or 1.0）
  [6:11] 視覚レイ5本（0〜1）
  ※[9:11]に受信パルスをbottleneckが上書き
"""
import numpy as np
from config import *


class BruteNN:
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        s = MUTATE_STD_INIT

        # 第三層FF (11→24)
        self.W3 = rng.normal(0, s, (BRUTE_L3_OUT, BRUTE_OBS_DIM))
        self.b3 = np.zeros(BRUTE_L3_OUT)

        # バッファGRU (12→5)
        self.Wz_b = rng.normal(0, s, (BRUTE_BUF_DIM, BRUTE_L3_BUF))
        self.Uz_b = rng.normal(0, s, (BRUTE_BUF_DIM, BRUTE_BUF_DIM))
        self.bz_b = np.zeros(BRUTE_BUF_DIM)
        self.Wr_b = rng.normal(0, s, (BRUTE_BUF_DIM, BRUTE_L3_BUF))
        self.Ur_b = rng.normal(0, s, (BRUTE_BUF_DIM, BRUTE_BUF_DIM))
        self.br_b = np.zeros(BRUTE_BUF_DIM)
        self.Wh_b = rng.normal(0, s, (BRUTE_BUF_DIM, BRUTE_L3_BUF))
        self.Uh_b = rng.normal(0, s, (BRUTE_BUF_DIM, BRUTE_BUF_DIM))
        self.bh_b = np.zeros(BRUTE_BUF_DIM)

        # 記憶GRU (17→12)
        mem_in = BRUTE_L3_NORMAL + BRUTE_BUF_DIM
        self.Wz_m = rng.normal(0, s, (BRUTE_MEM_DIM, mem_in))
        self.Uz_m = rng.normal(0, s, (BRUTE_MEM_DIM, BRUTE_MEM_DIM))
        self.bz_m = np.zeros(BRUTE_MEM_DIM)
        self.Wr_m = rng.normal(0, s, (BRUTE_MEM_DIM, mem_in))
        self.Ur_m = rng.normal(0, s, (BRUTE_MEM_DIM, BRUTE_MEM_DIM))
        self.br_m = np.zeros(BRUTE_MEM_DIM)
        self.Wh_m = rng.normal(0, s, (BRUTE_MEM_DIM, mem_in))
        self.Uh_m = rng.normal(0, s, (BRUTE_MEM_DIM, BRUTE_MEM_DIM))
        self.bh_m = np.zeros(BRUTE_MEM_DIM)

        # γ（継承領域初期値）
        self.gamma = np.zeros(BRUTE_MEM_INHERIT)

        # 通常FF (12→16)
        self.W_bypass = rng.normal(0, s, (BRUTE_BYPASS_OUT, BRUTE_L3_NORMAL))
        self.b_bypass = np.zeros(BRUTE_BYPASS_OUT)

        # 第一層FF (28→24)
        self.W1 = rng.normal(0, s, (BRUTE_L1_OUT, BRUTE_L1_IN))
        self.b1 = np.zeros(BRUTE_L1_OUT)

        # 行動出力FF (24→3)
        self.W_act = rng.normal(0, s, (BRUTE_ACTION_DIM, BRUTE_L1_OUT))
        self.b_act = np.zeros(BRUTE_ACTION_DIM)

        # 符号化FF (24→2)
        self.W_enc = rng.normal(0, s, (BRUTE_ENCODE_DIM, BRUTE_L1_OUT))
        self.b_enc = np.zeros(BRUTE_ENCODE_DIM)

        self.reset_episode()

    @staticmethod
    def _gru(x, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh):
        z = 1 / (1 + np.exp(-(Wz @ x + Uz @ h + bz)))
        r = 1 / (1 + np.exp(-(Wr @ x + Ur @ h + br)))
        n = np.tanh(Wh @ x + Uh @ (r * h) + bh)
        return (1 - z) * h + z * n

    def forward(self, obs: np.ndarray, is_pulse_frame: bool):
        x3 = np.tanh(self.W3 @ obs + self.b3)
        x3_normal = x3[:BRUTE_L3_NORMAL]
        x3_buf    = x3[BRUTE_L3_NORMAL:]

        if is_pulse_frame:
            self.h_buf   = self._gru(x3_buf, self.h_buf,
                self.Wz_b, self.Uz_b, self.bz_b,
                self.Wr_b, self.Ur_b, self.br_b,
                self.Wh_b, self.Uh_b, self.bh_b)
            self.buf_out = self.h_buf.copy()

        x_mem = np.concatenate([x3_normal, self.buf_out])
        self.h_mem = self._gru(x_mem, self.h_mem,
            self.Wz_m, self.Uz_m, self.bz_m,
            self.Wr_m, self.Ur_m, self.br_m,
            self.Wh_m, self.Uh_m, self.bh_m)

        x_bp = np.tanh(self.W_bypass @ x3_normal + self.b_bypass)
        x1   = np.tanh(self.W1 @ np.concatenate([self.h_mem, x_bp]) + self.b1)

        action = 1 / (1 + np.exp(-(self.W_act @ x1 + self.b_act)))
        raw    = np.tanh(self.W_enc @ x1 + self.b_enc)
        bits   = (raw >= 0).astype(int)
        pulse  = (bits[0] << 1) | bits[1]

        self.last_output_act = action.tolist()
        # モニター用保存
        self.last_l3_act     = x3.tolist()
        self.last_buf_act    = self.buf_out.tolist()
        self.last_buf_active = is_pulse_frame
        self.last_gru_act    = self.h_mem.tolist()
        self.last_pulse      = pulse

        return action, pulse

    def reset_episode(self):
        self.h_mem   = np.zeros(BRUTE_MEM_DIM)
        self.h_mem[:BRUTE_MEM_INHERIT] = self.gamma
        self.h_buf   = np.zeros(BRUTE_BUF_DIM)
        self.buf_out = np.zeros(BRUTE_BUF_DIM)
        self.last_pulse      = 0
        self.last_l3_act     = [0.0] * BRUTE_L3_OUT
        self.last_buf_act    = [0.0] * BRUTE_BUF_DIM
        self.last_buf_active = False
        self.last_gru_act    = [0.0] * BRUTE_MEM_DIM
        self.last_output_act = [0.5, 0.5, 0.0]

    def param_count(self) -> int:
        return len(self.flat())

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

    def load_flat(self, arr: np.ndarray):
        idx = 0
        def take(shape):
            nonlocal idx
            n = int(np.prod(shape))
            v = arr[idx:idx+n].reshape(shape)
            idx += n
            return v
        self.W3       = take((BRUTE_L3_OUT, BRUTE_OBS_DIM))
        self.b3       = take((BRUTE_L3_OUT,))
        self.Wz_b     = take((BRUTE_BUF_DIM, BRUTE_L3_BUF))
        self.Uz_b     = take((BRUTE_BUF_DIM, BRUTE_BUF_DIM))
        self.bz_b     = take((BRUTE_BUF_DIM,))
        self.Wr_b     = take((BRUTE_BUF_DIM, BRUTE_L3_BUF))
        self.Ur_b     = take((BRUTE_BUF_DIM, BRUTE_BUF_DIM))
        self.br_b     = take((BRUTE_BUF_DIM,))
        self.Wh_b     = take((BRUTE_BUF_DIM, BRUTE_L3_BUF))
        self.Uh_b     = take((BRUTE_BUF_DIM, BRUTE_BUF_DIM))
        self.bh_b     = take((BRUTE_BUF_DIM,))
        mem_in = BRUTE_L3_NORMAL + BRUTE_BUF_DIM
        self.Wz_m     = take((BRUTE_MEM_DIM, mem_in))
        self.Uz_m     = take((BRUTE_MEM_DIM, BRUTE_MEM_DIM))
        self.bz_m     = take((BRUTE_MEM_DIM,))
        self.Wr_m     = take((BRUTE_MEM_DIM, mem_in))
        self.Ur_m     = take((BRUTE_MEM_DIM, BRUTE_MEM_DIM))
        self.br_m     = take((BRUTE_MEM_DIM,))
        self.Wh_m     = take((BRUTE_MEM_DIM, mem_in))
        self.Uh_m     = take((BRUTE_MEM_DIM, BRUTE_MEM_DIM))
        self.bh_m     = take((BRUTE_MEM_DIM,))
        self.gamma    = take((BRUTE_MEM_INHERIT,))
        self.W_bypass = take((BRUTE_BYPASS_OUT, BRUTE_L3_NORMAL))
        self.b_bypass = take((BRUTE_BYPASS_OUT,))
        self.W1       = take((BRUTE_L1_OUT, BRUTE_L1_IN))
        self.b1       = take((BRUTE_L1_OUT,))
        self.W_act    = take((BRUTE_ACTION_DIM, BRUTE_L1_OUT))
        self.b_act    = take((BRUTE_ACTION_DIM,))
        self.W_enc    = take((BRUTE_ENCODE_DIM, BRUTE_L1_OUT))
        self.b_enc    = take((BRUTE_ENCODE_DIM,))
        self.reset_episode()

"""
sage.py — SAGE（賢者）ニューラルネットワーク

obs構成（11次元）:
  [0:6]  弁別視野マスク済み
          [0:3]=0固定（バイオーム見えない）
          [3]=栄養価, [4]=バリアント
          [5]=0固定（腐敗見えない）
  [6]    ゴール角度（-1〜1）
  [7]    ゴール距離（0〜1）
  [8]    エネルギー（0〜1）
  [9:11] 受信パルス（bottleneckが上書き）
"""
import numpy as np
from config import *


class SageNN:
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        s = MUTATE_STD_INIT

        # 第三層FF (11→24)
        self.W3 = rng.normal(0, s, (SAGE_L3_OUT, SAGE_OBS_DIM))
        self.b3 = np.zeros(SAGE_L3_OUT)

        # バッファGRU (12→5)
        self.Wz_b = rng.normal(0, s, (SAGE_BUF_DIM, SAGE_L3_BUF))
        self.Uz_b = rng.normal(0, s, (SAGE_BUF_DIM, SAGE_BUF_DIM))
        self.bz_b = np.zeros(SAGE_BUF_DIM)
        self.Wr_b = rng.normal(0, s, (SAGE_BUF_DIM, SAGE_L3_BUF))
        self.Ur_b = rng.normal(0, s, (SAGE_BUF_DIM, SAGE_BUF_DIM))
        self.br_b = np.zeros(SAGE_BUF_DIM)
        self.Wh_b = rng.normal(0, s, (SAGE_BUF_DIM, SAGE_L3_BUF))
        self.Uh_b = rng.normal(0, s, (SAGE_BUF_DIM, SAGE_BUF_DIM))
        self.bh_b = np.zeros(SAGE_BUF_DIM)

        # 記憶GRU (17→12)
        mem_in = SAGE_L3_NORMAL + SAGE_BUF_DIM  # 12+5=17
        self.Wz_m = rng.normal(0, s, (SAGE_MEM_DIM, mem_in))
        self.Uz_m = rng.normal(0, s, (SAGE_MEM_DIM, SAGE_MEM_DIM))
        self.bz_m = np.zeros(SAGE_MEM_DIM)
        self.Wr_m = rng.normal(0, s, (SAGE_MEM_DIM, mem_in))
        self.Ur_m = rng.normal(0, s, (SAGE_MEM_DIM, SAGE_MEM_DIM))
        self.br_m = np.zeros(SAGE_MEM_DIM)
        self.Wh_m = rng.normal(0, s, (SAGE_MEM_DIM, mem_in))
        self.Uh_m = rng.normal(0, s, (SAGE_MEM_DIM, SAGE_MEM_DIM))
        self.bh_m = np.zeros(SAGE_MEM_DIM)

        # γ（継承領域初期値）
        self.gamma = np.zeros(SAGE_MEM_INHERIT)

        # 通常FF (12→16)
        self.W_bypass = rng.normal(0, s, (SAGE_BYPASS_OUT, SAGE_L3_NORMAL))
        self.b_bypass = np.zeros(SAGE_BYPASS_OUT)

        # 第一層FF (28→24)
        self.W1 = rng.normal(0, s, (SAGE_L1_OUT, SAGE_L1_IN))
        self.b1 = np.zeros(SAGE_L1_OUT)

        # 符号化FF (24→2)
        self.W_enc = rng.normal(0, s, (SAGE_ENCODE_DIM, SAGE_L1_OUT))
        self.b_enc = np.zeros(SAGE_ENCODE_DIM)

        self.reset_episode()

    @staticmethod
    def _gru(x, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh):
        z = 1 / (1 + np.exp(-(Wz @ x + Uz @ h + bz)))
        r = 1 / (1 + np.exp(-(Wr @ x + Ur @ h + br)))
        n = np.tanh(Wh @ x + Uh @ (r * h) + bh)
        return (1 - z) * h + z * n

    def forward(self, obs: np.ndarray, is_pulse_frame: bool) -> int:
        x3 = np.tanh(self.W3 @ obs + self.b3)
        x3_normal = x3[:SAGE_L3_NORMAL]
        x3_buf    = x3[SAGE_L3_NORMAL:]

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

        raw  = np.tanh(self.W_enc @ x1 + self.b_enc)
        bits = (raw >= 0).astype(int)
        self.last_pulse = (bits[0] << 1) | bits[1]

        # モニター用保存
        self.last_l3_act     = x3.tolist()
        self.last_buf_act    = self.buf_out.tolist()
        self.last_buf_active = is_pulse_frame
        self.last_gru_act    = self.h_mem.tolist()

        return self.last_pulse

    def reset_episode(self):
        self.h_mem   = np.zeros(SAGE_MEM_DIM)
        self.h_mem[:SAGE_MEM_INHERIT] = self.gamma
        self.h_buf   = np.zeros(SAGE_BUF_DIM)
        self.buf_out = np.zeros(SAGE_BUF_DIM)
        self.last_pulse      = 0
        self.last_l3_act     = [0.0] * SAGE_L3_OUT
        self.last_buf_act    = [0.0] * SAGE_BUF_DIM
        self.last_buf_active = False
        self.last_gru_act    = [0.0] * SAGE_MEM_DIM

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
        self.W3       = take((SAGE_L3_OUT, SAGE_OBS_DIM))
        self.b3       = take((SAGE_L3_OUT,))
        self.Wz_b     = take((SAGE_BUF_DIM, SAGE_L3_BUF))
        self.Uz_b     = take((SAGE_BUF_DIM, SAGE_BUF_DIM))
        self.bz_b     = take((SAGE_BUF_DIM,))
        self.Wr_b     = take((SAGE_BUF_DIM, SAGE_L3_BUF))
        self.Ur_b     = take((SAGE_BUF_DIM, SAGE_BUF_DIM))
        self.br_b     = take((SAGE_BUF_DIM,))
        self.Wh_b     = take((SAGE_BUF_DIM, SAGE_L3_BUF))
        self.Uh_b     = take((SAGE_BUF_DIM, SAGE_BUF_DIM))
        self.bh_b     = take((SAGE_BUF_DIM,))
        mem_in = SAGE_L3_NORMAL + SAGE_BUF_DIM
        self.Wz_m     = take((SAGE_MEM_DIM, mem_in))
        self.Uz_m     = take((SAGE_MEM_DIM, SAGE_MEM_DIM))
        self.bz_m     = take((SAGE_MEM_DIM,))
        self.Wr_m     = take((SAGE_MEM_DIM, mem_in))
        self.Ur_m     = take((SAGE_MEM_DIM, SAGE_MEM_DIM))
        self.br_m     = take((SAGE_MEM_DIM,))
        self.Wh_m     = take((SAGE_MEM_DIM, mem_in))
        self.Uh_m     = take((SAGE_MEM_DIM, SAGE_MEM_DIM))
        self.bh_m     = take((SAGE_MEM_DIM,))
        self.gamma    = take((SAGE_MEM_INHERIT,))
        self.W_bypass = take((SAGE_BYPASS_OUT, SAGE_L3_NORMAL))
        self.b_bypass = take((SAGE_BYPASS_OUT,))
        self.W1       = take((SAGE_L1_OUT, SAGE_L1_IN))
        self.b1       = take((SAGE_L1_OUT,))
        self.W_enc    = take((SAGE_ENCODE_DIM, SAGE_L1_OUT))
        self.b_enc    = take((SAGE_ENCODE_DIM,))
        self.reset_episode()

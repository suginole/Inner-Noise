"""
batch_gpu.py — GPU並列バッチNN計算（モード3専用）

常に固定pop_sizeのバッチで処理する。
死亡個体はalive_maskで管理し、テンソルサイズは変えない。
"""
import numpy as np
import torch
from config import (
    SAGE_OBS_DIM, SAGE_L3_NORMAL, SAGE_L3_BUF, SAGE_BUF_DIM,
    SAGE_MEM_DIM, SAGE_MEM_INHERIT, SAGE_BYPASS_OUT, SAGE_L1_IN,
    SAGE_L1_OUT, SAGE_ENCODE_DIM,
    BRUTE_OBS_DIM, BRUTE_L3_NORMAL, BRUTE_L3_BUF, BRUTE_BUF_DIM,
    BRUTE_MEM_DIM, BRUTE_MEM_INHERIT, BRUTE_BYPASS_OUT, BRUTE_L1_IN,
    BRUTE_L1_OUT, BRUTE_ACTION_DIM, BRUTE_ENCODE_DIM,
    PULSE_GEN_INTERVAL, PULSE_CONSUME_RATE, TURN_FRAMES,
)


def _bmm(W, x):
    """(pop, out, in) @ (pop, in, 1) → (pop, out)"""
    return torch.bmm(W, x.unsqueeze(-1)).squeeze(-1)


def batch_gru_step(x, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh):
    """
    全個体のGRUを1ステップ一括処理。
    x:  (pop, input_dim)
    h:  (pop, hidden_dim)
    Wz: (pop, hidden_dim, input_dim)
    Uz: (pop, hidden_dim, hidden_dim)
    bz: (pop, hidden_dim)
    """
    z = torch.sigmoid(_bmm(Wz, x) + _bmm(Uz, h) + bz)
    r = torch.sigmoid(_bmm(Wr, x) + _bmm(Ur, h) + br)
    n = torch.tanh(_bmm(Wh, x) + _bmm(Uh, r * h) + bh)
    return (1 - z) * h + z * n


class BatchWeightManager:
    """
    全個体の重みをGPUバッチテンソルとして管理する。
    pop_sizeは固定。死亡個体もalive_maskで管理する。
    """

    def __init__(self, population: list, device: str):
        self.device   = device
        self.pop_size = len(population)
        self._load_weights(population)
        self.reset_episode()

    # ------------------------------------------------------------------
    def _stack(self, population, nn_name: str, attr: str) -> torch.Tensor:
        """population[i].sage.W3 などをスタックしてGPUテンソルに変換する。"""
        arrays = [getattr(getattr(g, nn_name), attr) for g in population]
        return torch.tensor(np.stack(arrays), dtype=torch.float32,
                            device=self.device)

    def _load_weights(self, population):
        """SAGEとBRUTEの重みを全個体分バッチテンソルに変換する。"""
        S = 'sage'
        B = 'brute'
        COMMON_KEYS = [
            'W3', 'b3',
            'Wz_b', 'Uz_b', 'bz_b',
            'Wr_b', 'Ur_b', 'br_b',
            'Wh_b', 'Uh_b', 'bh_b',
            'Wz_m', 'Uz_m', 'bz_m',
            'Wr_m', 'Ur_m', 'br_m',
            'Wh_m', 'Uh_m', 'bh_m',
            'W_bypass', 'b_bypass',
            'W1', 'b1',
            'W_enc', 'b_enc',
        ]
        self.ws = {k: self._stack(population, S, k) for k in COMMON_KEYS}
        self.wb = {k: self._stack(population, B, k) for k in COMMON_KEYS}
        self.wb['W_act'] = self._stack(population, B, 'W_act')
        self.wb['b_act'] = self._stack(population, B, 'b_act')
        self.gamma_sage  = self._stack(population, S, 'gamma')
        self.gamma_brute = self._stack(population, B, 'gamma')

    def reset_episode(self):
        """エピソード開始時に隠れ状態をγから初期化する。"""
        p = self.pop_size
        d = self.device

        # SAGE隠れ状態
        self.h_mem_sage   = torch.zeros(p, SAGE_MEM_DIM,  device=d)
        self.h_buf_sage   = torch.zeros(p, SAGE_BUF_DIM,  device=d)
        self.buf_out_sage = torch.zeros(p, SAGE_BUF_DIM,  device=d)
        self.h_mem_sage[:, :SAGE_MEM_INHERIT] = self.gamma_sage

        # BRUTE隠れ状態
        self.h_mem_brute   = torch.zeros(p, BRUTE_MEM_DIM, device=d)
        self.h_buf_brute   = torch.zeros(p, BRUTE_BUF_DIM, device=d)
        self.buf_out_brute = torch.zeros(p, BRUTE_BUF_DIM, device=d)
        self.h_mem_brute[:, :BRUTE_MEM_INHERIT] = self.gamma_brute

        # パルスバッファ・フレームカウンタ・行動
        self.pulse_buf   = [[] for _ in range(p)]
        self.bn_frame    = 0
        self.last_action = torch.zeros(p, BRUTE_ACTION_DIM, device=d)
        self.last_action[:, 1] = 0.5   # Steer中立
        self.alive_mask  = torch.ones(p, dtype=torch.bool, device=d)

    def update_weights(self, population):
        """GA進化後に重みテンソルを新世代で更新する。"""
        self._load_weights(population)
        self.reset_episode()

    # ------------------------------------------------------------------
    def forward_sage(self,
                     obs_np: np.ndarray,
                     is_pulse_frame: bool) -> torch.Tensor:
        """
        SAGEフォワード。
        obs_np: (pop_size, SAGE_OBS_DIM) numpy array
        戻り値: pulse_int (pop_size,) int tensor on device
        """
        w = self.ws
        x = torch.tensor(obs_np, dtype=torch.float32,
                         device=self.device)   # (pop, 11)

        x3        = torch.tanh(_bmm(w['W3'], x) + w['b3'])
        x3_normal = x3[:, :SAGE_L3_NORMAL]
        x3_buf    = x3[:, SAGE_L3_NORMAL:]

        if is_pulse_frame:
            self.h_buf_sage = batch_gru_step(
                x3_buf, self.h_buf_sage,
                w['Wz_b'], w['Uz_b'], w['bz_b'],
                w['Wr_b'], w['Ur_b'], w['br_b'],
                w['Wh_b'], w['Uh_b'], w['bh_b'])
            self.buf_out_sage = self.h_buf_sage.clone()

        x_mem = torch.cat([x3_normal, self.buf_out_sage], dim=-1)
        self.h_mem_sage = batch_gru_step(
            x_mem, self.h_mem_sage,
            w['Wz_m'], w['Uz_m'], w['bz_m'],
            w['Wr_m'], w['Ur_m'], w['br_m'],
            w['Wh_m'], w['Uh_m'], w['bh_m'])

        x_bp = torch.tanh(_bmm(w['W_bypass'], x3_normal) + w['b_bypass'])
        x1   = torch.tanh(_bmm(w['W1'],
                               torch.cat([self.h_mem_sage, x_bp], dim=-1))
                          + w['b1'])

        raw  = torch.tanh(_bmm(w['W_enc'], x1) + w['b_enc'])
        bits = (raw >= 0).int()
        return (bits[:, 0] << 1) | bits[:, 1]   # (pop,) int

    def forward_brute(self,
                      obs_np: np.ndarray,
                      is_pulse_frame: bool
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        BRUTEフォワード。
        obs_np: (pop_size, BRUTE_OBS_DIM) numpy array
        戻り値: (action (pop,3), pulse_int (pop,)) ともにdevice上
        """
        w = self.wb
        x = torch.tensor(obs_np, dtype=torch.float32,
                         device=self.device)

        x3        = torch.tanh(_bmm(w['W3'], x) + w['b3'])
        x3_normal = x3[:, :BRUTE_L3_NORMAL]
        x3_buf    = x3[:, BRUTE_L3_NORMAL:]

        if is_pulse_frame:
            self.h_buf_brute = batch_gru_step(
                x3_buf, self.h_buf_brute,
                w['Wz_b'], w['Uz_b'], w['bz_b'],
                w['Wr_b'], w['Ur_b'], w['br_b'],
                w['Wh_b'], w['Uh_b'], w['bh_b'])
            self.buf_out_brute = self.h_buf_brute.clone()

        x_mem = torch.cat([x3_normal, self.buf_out_brute], dim=-1)
        self.h_mem_brute = batch_gru_step(
            x_mem, self.h_mem_brute,
            w['Wz_m'], w['Uz_m'], w['bz_m'],
            w['Wr_m'], w['Ur_m'], w['br_m'],
            w['Wh_m'], w['Uh_m'], w['bh_m'])

        x_bp = torch.tanh(_bmm(w['W_bypass'], x3_normal) + w['b_bypass'])
        x1   = torch.tanh(_bmm(w['W1'],
                               torch.cat([self.h_mem_brute, x_bp], dim=-1))
                          + w['b1'])

        action = torch.sigmoid(_bmm(w['W_act'], x1) + w['b_act'])
        raw    = torch.tanh(_bmm(w['W_enc'], x1) + w['b_enc'])
        bits   = (raw >= 0).int()
        pulse  = (bits[:, 0] << 1) | bits[:, 1]
        return action, pulse   # (pop,3), (pop,)

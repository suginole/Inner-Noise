"""
renderer.py — 描画システム（Sage-Brute版 v2）

3パネルモニター:
  左: SAGE（青系・5列: obs/L3FF/BUF/GRU/P）
  中: BOTTLENECK（方向別パルス配色）
  右: BRUTE（赤系・5列: obs/L3FF/BUF/GRU/OUT）

摂取履歴パネル（中央下部）
"""
import math
import numpy as np
import pygame
from config import *

TOP_K_EDGES = 8


class Renderer:
    _JP_FONTS = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/truetype/ipafont/ipag.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    _font_cache: dict = {}

    @staticmethod
    def _load_font(size: int) -> pygame.font.Font:
        import os
        if size in Renderer._font_cache:
            p = Renderer._font_cache[size]
            return pygame.font.Font(p, size) if p else pygame.font.SysFont(None, size)
        for path in Renderer._JP_FONTS:
            if os.path.exists(path):
                Renderer._font_cache[size] = path
                return pygame.font.Font(path, size)
        Renderer._font_cache[size] = None
        return pygame.font.SysFont(None, size)

    def __init__(self, screen: pygame.Surface):
        self.screen  = screen
        self.font_s  = self._load_font(12)
        self.font_m  = self._load_font(15)
        self.font_l  = self._load_font(20)
        self._static_buf:   pygame.Surface | None = None
        self._minimap_surf: pygame.Surface | None = None

    # ----------------------------------------------------------------
    def calc_camera(self, pos: pygame.Vector2) -> pygame.Vector2:
        cx = max(0, min(WORLD_W - SCREEN_W, pos.x - SCREEN_W // 2))
        cy = max(0, min(WORLD_H - SCREEN_H, pos.y - SCREEN_H // 2))
        return pygame.Vector2(cx, cy)

    def draw_field(self, field, cam: pygame.Vector2):
        surf = field.get_surface()
        src  = pygame.Rect(int(cam.x), int(cam.y), SCREEN_W, SCREEN_H)
        self.screen.blit(surf, (0, 0), src)
        # キノコ描画
        for m in field.mushrooms:
            sx = int(m.pos.x - cam.x)
            sy = int(m.pos.y - cam.y)
            if -20 < sx < SCREEN_W + 20 and -20 < sy < SCREEN_H + 20:
                if m.is_rotten:
                    color = C_FOOD_ROT
                elif m.grade == 'premium':
                    color = C_FOOD_HI
                else:
                    color = C_FOOD
                pygame.draw.circle(self.screen, color, (sx, sy), 6)
                bc = BIOME_COLORS.get(m.biome, C_GRAY)
                pygame.draw.circle(self.screen, bc, (sx, sy), 6, 1)
        # ゴール
        gx = int(field.goal_pos[0] - cam.x)
        gy = int(field.goal_pos[1] - cam.y)
        pygame.draw.circle(self.screen, C_GOAL, (gx, gy), GOAL_RADIUS, 2)
        gt = self.font_s.render("GOAL", True, C_GOAL)
        self.screen.blit(gt, (gx - gt.get_width()//2, gy - 20))

    def draw_agent(self, agent, cam: pygame.Vector2, color=C_CAR, is_best=False):
        sx = int(agent.pos.x - cam.x)
        sy = int(agent.pos.y - cam.y)
        r  = 8 if is_best else 5
        pygame.draw.circle(self.screen, color, (sx, sy), r)
        ar = math.radians(agent.angle)
        ex = sx + int(math.cos(ar) * r * 2)
        ey = sy + int(math.sin(ar) * r * 2)
        pygame.draw.line(self.screen, C_WHITE, (sx, sy), (ex, ey), 1)

    def draw_minimap(self, field, focus_pos, goal_pos, x=20, y=20, size=120):
        bg = pygame.Surface((size, size), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 180))
        self.screen.blit(bg, (SCREEN_W - size - x, y))
        pygame.draw.rect(self.screen, C_GRAY, (SCREEN_W - size - x, y, size, size), 1)
        sx = int(focus_pos.x / WORLD_W * size) + SCREEN_W - size - x
        sy = int(focus_pos.y / WORLD_H * size) + y
        pygame.draw.circle(self.screen, C_CAR, (sx, sy), 3)
        gx = int(goal_pos[0] / WORLD_W * size) + SCREEN_W - size - x
        gy = int(goal_pos[1] / WORLD_H * size) + y
        pygame.draw.circle(self.screen, C_GOAL, (gx, gy), 4)

    # ----------------------------------------------------------------
    # エッジ描画ヘルパー
    # ----------------------------------------------------------------
    def _draw_edges(self, acts_a, sy_a, cx_a, r_a,
                    acts_b, sy_b, cx_b, r_b,
                    W, node_gap_a=12, node_gap_b=12):
        """上位TOP_K_EDGESのエッジを描画する。正=緑・負=赤。"""
        if W is None:
            return
        try:
            W_np = np.array(W) if not hasattr(W, 'shape') else W
            n_a = min(len(acts_a), W_np.shape[1] if W_np.ndim > 1 else 1)
            n_b = min(len(acts_b), W_np.shape[0])
            pairs = []
            for i in range(n_a):
                for j in range(n_b):
                    try:
                        w = float(W_np[j, i]) if W_np.ndim > 1 else float(W_np[j])
                        pairs.append((abs(w), w, i, j))
                    except Exception:
                        pass
            pairs.sort(reverse=True)
            for _, w, i, j in pairs[:TOP_K_EDGES]:
                iy = sy_a + i * node_gap_a
                jy = sy_b + j * node_gap_b
                alpha = min(200, int(abs(w) * 80))
                c = (20, alpha, 20) if w > 0 else (alpha, 20, 20)
                pygame.draw.line(self.screen, c, (cx_a + r_a, iy), (cx_b - r_b, jy), 1)
        except Exception:
            pass

    # ----------------------------------------------------------------
    # 3パネルモニター
    # ----------------------------------------------------------------
    def draw_rnn_monitor_panels(self, genome, bottleneck,
                                 x=0, y=0, panel_w=280, panel_h=220):
        gap = 8
        cx_bn    = x + panel_w + gap
        cx_brute = x + (panel_w + gap) * 2

        # 追従エージェントのobs（bottleneckから取得できれば使用）
        obs_sage  = getattr(bottleneck, '_last_obs_sage',  None)
        obs_brute = getattr(bottleneck, '_last_obs_brute', None)

        self._draw_sage_panel(genome.sage, x, y, panel_w, panel_h, obs_sage)
        self._draw_bottleneck_panel(bottleneck, cx_bn, y, panel_w, panel_h)
        self._draw_brute_panel(genome.brute, cx_brute, y, panel_w, panel_h, obs_brute)
        self._draw_flow_arrows(x, y, panel_w, panel_h, gap, bottleneck)

    # ----------------------------------------------------------------
    def _draw_sage_panel(self, sage, x, y, w, h, obs=None):
        """左パネル: SAGE（青系）5列: obs/L3/BUF/GRU/P"""
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((5, 10, 30, 220))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (60, 100, 200), (x, y, w, h), 1, border_radius=4)
        title = self.font_s.render("SAGE", True, (100, 160, 255))
        self.screen.blit(title, (x + 6, y + 4))

        l3_acts  = getattr(sage, 'last_l3_act',     [0.0] * SAGE_L3_OUT)
        buf_acts = getattr(sage, 'last_buf_act',    [0.0] * SAGE_BUF_DIM)
        buf_act  = getattr(sage, 'last_buf_active', False)
        gru_acts = getattr(sage, 'last_gru_act',    [0.0] * SAGE_MEM_DIM)
        pulse    = getattr(sage, 'last_pulse', 0)
        pulse_bits = [(pulse >> 1) & 1, pulse & 1]

        node_r = 3; node_gap = 10; row_top = y + 18
        # 5列の水平位置
        col_obs   = x + 14
        col_l3    = x + int(w * 0.28)
        col_buf   = x + int(w * 0.46)
        col_gru   = x + int(w * 0.64)
        col_pulse = x + w - 14

        def bip(v):
            t = min(1.0, abs(float(v)))
            return (int(40+215*t), 40, 40) if float(v) >= 0 else (40, 40, int(40+215*t))

        # ---- 列1: obs INPUT (17次元) ----
        if obs is not None:
            obs_arr = list(obs)
        else:
            obs_arr = [0.0] * SAGE_OBS_DIM
        obs_labels = [
            "Biome-W", "Biome-G", "Biome-M",  # [0:3] マスク（常に0）
            "Grade",                            # [3]   栄養価
            "Variant",                          # [4]   バリアント
            "Rot",                              # [5]   マスク（常に0）
            "GoalAng",                          # [6]   ゴール角度
            "GoalDist",                         # [7]   ゴール距離
            "Energy",                           # [8]   エネルギー
            "Pulse-0",                          # [9]   受信パルスbit0
            "Pulse-1",                          # [10]  受信パルスbit1
        ]
        sage_mask_idx = {0, 1, 2, 5}   # 常に0固定のインデックス
        n_obs = len(obs_arr)
        sy_obs = row_top + max(0, (h - 28 - n_obs * node_gap) // 2)
        for i, v in enumerate(obs_arr):
            ny = sy_obs + i * node_gap
            if i < 3:    # バイオーム（マスク）
                c = (20, 20, 20); border = (40, 40, 60)
            elif i == 3:  # 栄養価: 0=暗・1=明緑
                t = max(0.0, min(1.0, float(v)))
                c = (int(20+200*t), int(20+200*t), 20); border = (60, 120, 60)
            elif i == 4:  # バリアント
                t = max(0.0, min(1.0, float(v)))
                c = (int(20+200*t), int(20+200*t), 20); border = (60, 120, 60)
            elif i == 5:  # 腐敗（マスク）
                c = (20, 20, 20); border = (40, 40, 60)
            elif i == 6:  # ゴール角度
                t = min(1.0, abs(float(v)))
                c = (int(40+180*t), 40, 40) if float(v) >= 0 else (40, 40, int(40+180*t))
                border = (100, 100, 100)
            elif i == 7:  # ゴール距離
                t = max(0.0, min(1.0, float(v)))
                c = (int(20+200*t), int(20+200*t), int(20+200*t)); border = (100, 100, 100)
            elif i == 8:  # エネルギー
                t = max(0.0, min(1.0, float(v)))
                c = (int(220*(1-t)), int(220*t), 20); border = (100, 100, 100)
            else:          # 受信パルス
                c = (80, 160, 255) if float(v) > 0.5 else (40, 40, 60)
                border = (80, 120, 200)
            pygame.draw.circle(self.screen, c, (col_obs, ny), node_r)
            pygame.draw.circle(self.screen, border, (col_obs, ny), node_r, 1)
            if i < len(obs_labels):
                lc = (40, 40, 60) if i in sage_mask_idx else (60, 90, 140)
                lt = self.font_s.render(obs_labels[i], True, lc)
                self.screen.blit(lt, (col_obs - node_r - lt.get_width() - 1, ny - 5))

        # ---- 列2: 第三層FF (24次元・通常/バッファ区切り) ----
        n = len(l3_acts); sy_l3 = row_top + max(0, (h-28-n*node_gap)//2)
        for i, v in enumerate(l3_acts):
            ny = sy_l3 + i * node_gap
            border = (80, 130, 220) if i < SAGE_L3_NORMAL else (60, 60, 120)
            pygame.draw.circle(self.screen, bip(v), (col_l3, ny), node_r)
            pygame.draw.circle(self.screen, border, (col_l3, ny), node_r, 1)
        if n >= SAGE_L3_NORMAL:
            sep = sy_l3 + SAGE_L3_NORMAL * node_gap - node_gap // 2
            pygame.draw.line(self.screen, (60,80,140), (col_l3-8, sep), (col_l3+8, sep), 1)

        # ---- 列3: バッファGRU buf_out (5次元・紫系) ----
        buf_c = (180, 80, 220) if buf_act else (80, 30, 100)
        nb = len(buf_acts); sy_buf = row_top + max(0, (h-28-nb*node_gap)//2)
        for i, v in enumerate(buf_acts):
            ny = sy_buf + i * node_gap
            pygame.draw.circle(self.screen, bip(v), (col_buf, ny), node_r)
            pygame.draw.circle(self.screen, buf_c, (col_buf, ny), node_r, 1)
        if not buf_act:
            lt = self.font_s.render("■", True, (60, 20, 80))
            self.screen.blit(lt, (col_buf - lt.get_width()//2, sy_buf - 12))

        # ---- 列4: 記憶GRU (12次元・継承/非継承) ----
        ng = len(gru_acts); sy_gru = row_top + max(0, (h-28-ng*node_gap)//2)
        for i, v in enumerate(gru_acts):
            ny = sy_gru + i * node_gap
            border = (80, 120, 200) if i < SAGE_MEM_INHERIT else (30, 50, 80)
            pygame.draw.circle(self.screen, bip(v), (col_gru, ny), node_r)
            pygame.draw.circle(self.screen, border, (col_gru, ny), node_r, 1)
        if ng > SAGE_MEM_INHERIT:
            sep = sy_gru + SAGE_MEM_INHERIT * node_gap - node_gap // 2
            pygame.draw.line(self.screen, (60,80,120), (col_gru-8, sep), (col_gru+8, sep), 1)

        # ---- 列5: パルス符号化 (2次元) ----
        sy_p = row_top + max(0, (h-28-2*26)//2)
        for i, bit in enumerate(pulse_bits):
            ny = sy_p + i * 26
            c = C_PULSE_ON if bit else C_PULSE_OFF
            pygame.draw.rect(self.screen, c, (col_pulse-9, ny, 16, 18))
            pygame.draw.rect(self.screen, C_GRAY, (col_pulse-9, ny, 16, 18), 1)
            bt = self.font_s.render(str(bit), True, C_WHITE if bit else C_GRAY)
            self.screen.blit(bt, (col_pulse-9+8-bt.get_width()//2, ny+3))

        # ---- エッジ線 ----
        W3 = getattr(sage, 'W3', None)
        W_enc = getattr(sage, 'W_enc', None)
        if W3 is not None:
            self._draw_edges(obs_arr, sy_obs, col_obs, node_r,
                             l3_acts, sy_l3, col_l3, node_r, W3, node_gap_a=node_gap)
        if W_enc is not None:
            self._draw_edges(gru_acts, sy_gru, col_gru, node_r,
                             pulse_bits, sy_p, col_pulse, node_r, W_enc,
                             node_gap_a=node_gap, node_gap_b=26)

        # ---- ラベル ----
        for lbl, lx, lc in [("IN", col_obs, (70,110,200)), ("L3", col_l3, (90,130,220)),
                              ("BUF", col_buf, (180,80,220)), ("GRU", col_gru, (100,140,220)),
                              ("P", col_pulse, (200,200,80))]:
            lt = self.font_s.render(lbl, True, lc)
            self.screen.blit(lt, (lx - lt.get_width()//2, y+h-14))

    # ----------------------------------------------------------------
    def _draw_brute_panel(self, brute, x, y, w, h, obs=None):
        """右パネル: BRUTE（赤系）5列: obs/L3/BUF/GRU/OUT"""
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((30, 5, 5, 220))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (200, 60, 60), (x, y, w, h), 1, border_radius=4)
        title = self.font_s.render("BRUTE", True, (255, 100, 100))
        self.screen.blit(title, (x + 6, y + 4))

        l3_acts  = getattr(brute, 'last_l3_act',     [0.0] * BRUTE_L3_OUT)
        buf_acts = getattr(brute, 'last_buf_act',    [0.0] * BRUTE_BUF_DIM)
        buf_act  = getattr(brute, 'last_buf_active', False)
        gru_acts = getattr(brute, 'last_gru_act',    [0.0] * BRUTE_MEM_DIM)
        out_acts = getattr(brute, 'last_output_act', [0.5, 0.5, 0.0])
        out_labels = ["Acc", "Str", "Brk"]

        node_r = 3; node_gap = 10; row_top = y + 18
        col_obs   = x + 14
        col_l3    = x + int(w * 0.28)
        col_buf   = x + int(w * 0.46)
        col_gru   = x + int(w * 0.64)
        col_out   = x + int(w * 0.80)
        bar_w     = w - (col_out - x) - 6

        def bip(v):
            t = min(1.0, abs(float(v)))
            return (int(40+215*t), 40, 40) if float(v) >= 0 else (40, 40, int(40+215*t))

        # ---- 列1: obs INPUT (11次元) ----
        if obs is not None:
            obs_arr = list(obs)
        else:
            obs_arr = [0.0] * BRUTE_OBS_DIM
        obs_labels = [
            "Biome-W", "Biome-G", "Biome-M",  # [0:3] バイオーム
            "Grade",                            # [3]   マスク（常に0）
            "Variant",                          # [4]   マスク（常に0）
            "Rot",                              # [5]   腐敗
            "Ray-0", "Ray-1", "Ray-2",         # [6:9] 視覚レイ
            "Ray-3", "Ray-4",                  # [9:11]視覚レイ続き
        ]
        brute_mask_idx = {3, 4}   # 常に0固定のインデックス
        n_obs = len(obs_arr)
        sy_obs = row_top + max(0, (h - 28 - n_obs * node_gap) // 2)
        for i, v in enumerate(obs_arr):
            ny = sy_obs + i * node_gap
            if i < 3:    # バイオームone-hot: バイオーム色
                biome_keys = ['W', 'G', 'M']
                bc = BIOME_COLORS.get(biome_keys[i], C_GRAY)
                t = max(0.0, min(1.0, float(v)))
                c = tuple(int(cc * t + 20 * (1-t)) for cc in bc)
                border = bc
            elif i == 3:  # 栄養価（マスク）
                c = (20, 20, 20); border = (50, 35, 35)
            elif i == 4:  # バリアント（マスク）
                c = (20, 20, 20); border = (50, 35, 35)
            elif i == 5:  # 腐敗: ON=赤・OFF=暗
                c = (220, 60, 60) if float(v) > 0.5 else (40, 40, 60)
                border = (180, 60, 60)
            elif i < 9:   # 視覚レイ[0:3]: 0=暗・1=明緑
                t = max(0.0, min(1.0, float(v)))
                c = (int(20+200*t), int(20+200*t), 20); border = (60, 120, 60)
            else:          # 視覚レイ[3:5]
                t = max(0.0, min(1.0, float(v)))
                c = (int(20+200*t), int(20+200*t), 20); border = (60, 120, 60)
            pygame.draw.circle(self.screen, c, (col_obs, ny), node_r)
            pygame.draw.circle(self.screen, border, (col_obs, ny), node_r, 1)
            if i < len(obs_labels):
                lc = (50, 35, 35) if i in brute_mask_idx else (120, 60, 60)
                lt = self.font_s.render(obs_labels[i], True, lc)
                self.screen.blit(lt, (col_obs - node_r - lt.get_width() - 1, ny - 5))

        # ---- 列2: 第三層FF (24次元) ----
        n = len(l3_acts); sy_l3 = row_top + max(0, (h-28-n*node_gap)//2)
        for i, v in enumerate(l3_acts):
            ny = sy_l3 + i * node_gap
            border = (220, 80, 80) if i < BRUTE_L3_NORMAL else (100, 30, 30)
            pygame.draw.circle(self.screen, bip(v), (col_l3, ny), node_r)
            pygame.draw.circle(self.screen, border, (col_l3, ny), node_r, 1)
        if n >= BRUTE_L3_NORMAL:
            sep = sy_l3 + BRUTE_L3_NORMAL * node_gap - node_gap // 2
            pygame.draw.line(self.screen, (120,40,40), (col_l3-8, sep), (col_l3+8, sep), 1)

        # ---- 列3: バッファGRU (5次元・紫系) ----
        buf_c = (180, 80, 220) if buf_act else (80, 30, 100)
        nb = len(buf_acts); sy_buf = row_top + max(0, (h-28-nb*node_gap)//2)
        for i, v in enumerate(buf_acts):
            ny = sy_buf + i * node_gap
            pygame.draw.circle(self.screen, bip(v), (col_buf, ny), node_r)
            pygame.draw.circle(self.screen, buf_c, (col_buf, ny), node_r, 1)
        if not buf_act:
            lt = self.font_s.render("■", True, (60, 20, 80))
            self.screen.blit(lt, (col_buf - lt.get_width()//2, sy_buf - 12))

        # ---- 列4: 記憶GRU (12次元) ----
        ng = len(gru_acts); sy_gru = row_top + max(0, (h-28-ng*node_gap)//2)
        for i, v in enumerate(gru_acts):
            ny = sy_gru + i * node_gap
            border = (200, 80, 80) if i < BRUTE_MEM_INHERIT else (60, 20, 20)
            pygame.draw.circle(self.screen, bip(v), (col_gru, ny), node_r)
            pygame.draw.circle(self.screen, border, (col_gru, ny), node_r, 1)
        if ng > BRUTE_MEM_INHERIT:
            sep = sy_gru + BRUTE_MEM_INHERIT * node_gap - node_gap // 2
            pygame.draw.line(self.screen, (100,40,40), (col_gru-8, sep), (col_gru+8, sep), 1)

        # ---- 列5: OUTPUT (バー+数値) ----
        n_out = len(out_acts); sy_out = row_top + max(0, (h-28-n_out*36)//2)
        for i, (v, lbl) in enumerate(zip(out_acts, out_labels)):
            oy = sy_out + i * 36
            lt = self.font_s.render(lbl, True, (200, 100, 100))
            self.screen.blit(lt, (col_out - lt.get_width() - 2, oy + 2))
            pygame.draw.rect(self.screen, (50,20,20), (col_out, oy, bar_w, 14))
            pygame.draw.rect(self.screen, (220,80,80),
                             (col_out, oy, int(bar_w * max(0, min(1, v))), 14))
            vt = self.font_s.render(f"{v:.2f}", True, (200,150,150))
            self.screen.blit(vt, (col_out, oy+16))

        # ---- エッジ線 ----
        W3 = getattr(brute, 'W3', None)
        W_act = getattr(brute, 'W_act', None)
        if W3 is not None:
            self._draw_edges(obs_arr, sy_obs, col_obs, node_r,
                             l3_acts, sy_l3, col_l3, node_r, W3, node_gap_a=node_gap)
        if W_act is not None:
            self._draw_edges(gru_acts, sy_gru, col_gru, node_r,
                             out_acts, sy_out, col_out, node_r, W_act,
                             node_gap_a=node_gap, node_gap_b=36)

        # ---- ラベル ----
        for lbl, lx, lc in [("IN", col_obs, (140,60,60)), ("L3", col_l3, (200,80,80)),
                              ("BUF", col_buf, (180,80,220)), ("GRU", col_gru, (210,90,90)),
                              ("OUT", col_out+bar_w//2, (240,120,120))]:
            lt = self.font_s.render(lbl, True, lc)
            self.screen.blit(lt, (lx - lt.get_width()//2, y+h-14))

    # ----------------------------------------------------------------
    def _draw_bottleneck_panel(self, bottleneck, x, y, w, h):
        """中央パネル: BOTTLENECK（方向別パルス配色）"""
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((20, 18, 5, 220))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (200, 180, 50), (x, y, w, h), 1, border_radius=4)

        title = self.font_s.render("BOTTLENECK  2bits / 10Hz", True, (220, 200, 80))
        self.screen.blit(title, (x + 4, y + 3))

        pulse     = bottleneck.get_current_pulse() if hasattr(bottleneck, 'get_current_pulse') else [0, 0]
        mode      = bottleneck.get_mode()           if hasattr(bottleneck, 'get_mode')           else 'listen'
        prog      = bottleneck.get_display_progress() if hasattr(bottleneck, 'get_display_progress') else 0.0
        hist      = bottleneck.get_display_history()  if hasattr(bottleneck, 'get_display_history')  else []
        phoneme   = bottleneck.get_display_phoneme()  if hasattr(bottleneck, 'get_display_phoneme')  else ''
        direction = getattr(bottleneck, 'direction', 'S→B')
        turn      = getattr(bottleneck, '_turn', 0)

        # 方向別パルス色
        pulse_on_c = PULSE_COLOR_S_TO_B if direction == 'S→B' else PULSE_COLOR_B_TO_S
        pulse_off_c = PULSE_COLOR_OFF

        # 方向・ターン表示
        dir_c = PULSE_COLOR_S_TO_B if direction == 'S→B' else PULSE_COLOR_B_TO_S
        dir_t = self.font_m.render(f"{direction}  Turn:{turn}", True, dir_c)
        self.screen.blit(dir_t, (x + 6, y + 18))

        # 進捗バー
        pygame.draw.rect(self.screen, (30, 30, 50), (x + 6, y + 38, w - 12, 5))
        pygame.draw.rect(self.screen, dir_c, (x + 6, y + 38, int((w - 12) * prog), 5))

        # 現在パルス（方向別配色）
        for i, bit in enumerate(pulse[:2]):
            bx = x + 6 + i * 36
            bc = pulse_on_c if bit else pulse_off_c
            pygame.draw.rect(self.screen, bc, (bx, y + 46, 28, 28))
            pygame.draw.rect(self.screen, C_GRAY, (bx, y + 46, 28, 28), 1)
            bt = self.font_m.render(str(bit), True, C_WHITE if bit else C_GRAY)
            self.screen.blit(bt, (bx + 14 - bt.get_width()//2, y + 52))

        # 音素表示（スロット直上）
        HIST_SLOTS = 20
        slot_w = max(6, (w - 12) // HIST_SLOTS)
        hy = y + h - 28
        if phoneme:
            pt = self.font_m.render(f"「{phoneme}」", True, (255, 220, 100))
            self.screen.blit(pt, (x + 6, hy - pt.get_height() - 2))

        # スロット履歴（方向別配色）
        padded = hist[-HIST_SLOTS:]
        for hi, hp in enumerate(padded):
            for bi, bit in enumerate(hp[:2]):
                bx = x + 6 + hi * slot_w
                by = hy + bi * 10
                c = pulse_on_c if bit else pulse_off_c
                pygame.draw.rect(self.screen, c, (bx, by, slot_w - 2, 8))

    def _draw_flow_arrows(self, x, y, panel_w, panel_h, gap, bottleneck):
        """パネル間の接続矢印（方向別配色・ボトルネック側を向く）。"""
        import time
        flash = (int(time.time() * 5) % 2 == 0)
        direction = getattr(bottleneck, 'direction', 'S→B')
        mid_y = y + panel_h // 2
        ax1 = x + panel_w; ax2 = ax1 + gap
        ax3 = ax2 + panel_w; ax4 = ax3 + gap

        # 左矢印（SAGE→BOTTLENECK）
        c1 = PULSE_COLOR_S_TO_B if (direction == 'S→B' and flash) else (50, 60, 80)
        pygame.draw.line(self.screen, c1, (ax1, mid_y), (ax2, mid_y), 2)
        pygame.draw.polygon(self.screen, c1, [(ax2, mid_y), (ax2-5, mid_y-4), (ax2-5, mid_y+4)])

        # 右矢印（BRUTE→BOTTLENECK・左向き）
        c2 = PULSE_COLOR_B_TO_S if (direction == 'B→S' and flash) else (50, 80, 50)
        pygame.draw.line(self.screen, c2, (ax4, mid_y), (ax3, mid_y), 2)
        pygame.draw.polygon(self.screen, c2, [(ax3, mid_y), (ax3+5, mid_y-4), (ax3+5, mid_y+4)])

    # ----------------------------------------------------------------
    def draw_intake_panel(self, agent, x: int, y: int, w: int = 300, h: int = 80):
        """摂取履歴パネル（中央下部）。"""
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((10, 15, 25, 200))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (120, 100, 50), (x, y, w, h), 1, border_radius=4)

        title = self.font_s.render("INTAKE HISTORY", True, (200, 180, 80))
        self.screen.blit(title, (x + 6, y + 4))

        biome_map = {'W': ('沼', BIOME_COLORS['W']), 'G': ('平地', BIOME_COLORS['G']),
                     'M': ('山', BIOME_COLORS['M'])}
        if hasattr(agent, 'field'):
            biome = agent.field.biome_at(agent.pos.x, agent.pos.y)
            bname, bcolor = biome_map.get(biome, ('?', C_GRAY))
            bt = self.font_s.render(f"Biome: {bname}", True, bcolor)
            self.screen.blit(bt, (x + w - bt.get_width() - 6, y + 4))

        history = getattr(agent, 'intake_history', [])
        toxic_c = getattr(agent, 'toxic_count', 0)
        for i, sk in enumerate(history[-5:]):
            bx = x + 6 + i * 54
            biome_c = BIOME_COLORS.get(sk[0], C_GRAY)
            grade_c = C_FOOD_HI if sk[1] == 'premium' else C_FOOD
            pygame.draw.rect(self.screen, biome_c, (bx, y+20, 48, 24), border_radius=3)
            pygame.draw.rect(self.screen, grade_c, (bx, y+20, 48, 24), 1, border_radius=3)
            label = f"{sk[0]}{sk[2]}"
            lt = self.font_s.render(label, True, C_WHITE)
            self.screen.blit(lt, (bx + 24 - lt.get_width()//2, y + 26))

        tc = self.font_s.render(f"Toxic: {toxic_c}/{TOXIC_COUNT}", True,
                                 (220, 80, 80) if toxic_c >= TOXIC_COUNT - 1 else C_GRAY)
        self.screen.blit(tc, (x + 6, y + h - 16))

    # ----------------------------------------------------------------
    def draw_ga_overlay(self, stats: dict, alive: int, x=20, y=20):
        lines = [
            f"Gen: {stats['generation']:4d}   Pop: {stats.get('pop_size', '?')}",
            f"Alive: {alive:3d}",
            f"Best:  {stats['best']:8.1f}",
            f"Avg:   {stats['avg']:8.1f}",
            f"Species: {stats.get('species', '-')}  Mut: {stats.get('mut_rate', 0):.3f}",
        ]
        hud_h = 12 + len(lines) * 18
        bg = pygame.Surface((280, hud_h), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 200))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (194, 24, 91), (x, y, 280, hud_h), 1)
        for i, line in enumerate(lines):
            t = self.font_s.render(line, True, (244, 143, 177))
            self.screen.blit(t, (x + 6, y + 6 + i * 18))

    def draw_fitness_graph(self, ga, x, y, w=280, h=100):
        if not ga.best_fitness_history:
            return
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((8, 10, 18, 200))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, C_GRAY, (x, y, w, h), 1)
        hist = ga.best_fitness_history[-w:]
        if len(hist) < 2:
            return
        mn, mx = min(hist), max(hist)
        rng = mx - mn if mx != mn else 1.0
        pts = [(x + int(i / len(hist) * w),
                y + h - int((v - mn) / rng * (h - 4)) - 2)
               for i, v in enumerate(hist)]
        pygame.draw.lines(self.screen, (100, 200, 120), False, pts, 1)

    def draw_mode_select(self):
        self.screen.fill(C_BG)
        title = self.font_l.render("INNER NOISE — Sage & Brute", True, C_WHITE)
        self.screen.blit(title, (SCREEN_W//2 - title.get_width()//2, 180))
        sub = self.font_m.render("10Hz Bottleneck Communication", True, C_GRAY)
        self.screen.blit(sub, (SCREEN_W//2 - sub.get_width()//2, 220))
        options = [
            ("2", "GA MODE",    "GAエージェントを学習させる",   (100, 200, 120)),
            ("3", "FAST MODE",  "高速学習モード",               (255, 200, 50)),
            ("L", "LOAD MODEL", "保存済みモデルをロードする",   (180, 140, 220)),
            ("O", "SOUND MODE", "音声入出力デバッグ",           (255, 120, 180)),
        ]
        for i, (key, name, desc, color) in enumerate(options):
            oy = 300 + i * 60
            kt = self.font_l.render(f"[{key}]", True, color)
            nt = self.font_m.render(name, True, C_WHITE)
            dt = self.font_s.render(desc, True, C_GRAY)
            self.screen.blit(kt, (SCREEN_W//2 - 200, oy))
            self.screen.blit(nt, (SCREEN_W//2 - 120, oy + 2))
            self.screen.blit(dt, (SCREEN_W//2 - 120, oy + 22))

    def draw_save_toast(self, msg: str, alpha: int = 220):
        t = self.font_m.render(msg, True, C_WHITE)
        w, h = t.get_width() + 24, t.get_height() + 14
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((10, 15, 25, alpha))
        x = SCREEN_W // 2 - w // 2
        self.screen.blit(bg, (x, 20))
        self.screen.blit(t, (x + 12, 27))

    def draw_overlay(self, text: str, color=C_WHITE):
        t = self.font_l.render(text, True, color)
        x = SCREEN_W//2 - t.get_width()//2
        y = SCREEN_H//2 - t.get_height()//2
        bg = pygame.Surface((t.get_width()+20, t.get_height()+12), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 220))
        self.screen.blit(bg, (x-10, y-6))
        self.screen.blit(t, (x, y))

    def draw_load_screen(self, models: list, sel_idx: int, error: str = ""):
        self.screen.fill(C_BG)
        title = self.font_l.render("LOAD MODEL", True, C_WHITE)
        self.screen.blit(title, (SCREEN_W//2 - title.get_width()//2, 40))
        if not models:
            t = self.font_m.render("保存済みモデルがありません", True, C_GRAY)
            self.screen.blit(t, (SCREEN_W//2 - t.get_width()//2, 200))
        for i, m in enumerate(models[:10]):
            oy = 100 + i * 40
            selected = (i == sel_idx)
            color = C_WHITE if selected else C_GRAY
            ok_tag = "[OK]" if m.get('compatible') else "[NG]"
            ok_c   = C_GOAL if m.get('compatible') else (220, 80, 80)
            line = f"ID:{m['id']}  Gen:{m['generation']}  Best:{m['best_fitness']:.1f}  {m.get('saved_at','')[:16]}"
            if selected:
                pygame.draw.rect(self.screen, (30, 40, 60), (80, oy-2, SCREEN_W-160, 34))
            t = self.font_m.render(line, True, color)
            self.screen.blit(t, (100, oy))
            ok_t = self.font_s.render(ok_tag, True, ok_c)
            self.screen.blit(ok_t, (SCREEN_W - 120, oy + 4))
        if error:
            et = self.font_m.render(error, True, (220, 80, 80))
            self.screen.blit(et, (SCREEN_W//2 - et.get_width()//2, SCREEN_H - 80))
        hint = self.font_s.render("↑↓: 選択   Enter: ロード   Del: 削除   ESC: 戻る", True, C_GRAY)
        self.screen.blit(hint, (SCREEN_W//2 - hint.get_width()//2, SCREEN_H - 30))

    # ================================================================
    # バックルーム（サウンドデバッグ）モード描画
    # ================================================================
    def draw_bottleneck_dummy(self, x: int, y: int,
                              pulse_state=None, history=None,
                              mode: str = "listen", turn_progress: float = 0.0,
                              phoneme: str = "", audio_on: bool = False,
                              is_dummy: bool = True):
        """ボトルネック通信路の可視化パネル（バックルーム用）。"""
        W = 260; H = 90
        bg = pygame.Surface((W, H), pygame.SRCALPHA)
        bg.fill((8, 10, 18, 220))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (80, 60, 20), (x, y, W, H), 1)

        dummy_tag = "  [DUMMY]" if is_dummy else ""
        title = self.font_s.render(f"BOTTLENECK  10Hz{dummy_tag}", True, (180, 140, 60))
        self.screen.blit(title, (x + 4, y + 3))

        if pulse_state is None:
            import time
            t_idx = int(time.time() * 5) % 4
            pulse_state = [(t_idx >> (1 - i)) & 1 for i in range(2)]
        if history is None:
            import time
            history = []
            for k in range(16):
                ts = int(time.time() * 5 - k) % 4
                history.append([(ts >> (1 - i)) & 1 for i in range(2)])

        mode_color = (100, 160, 220) if mode == "listen" else (100, 200, 120)
        gender_sym = "♀" if mode == "listen" else "♂"
        mode_str = f"LISTEN  S→B  {gender_sym}" if mode == "listen" else f"SPEAK   B→S  {gender_sym}"
        mode_lbl = self.font_s.render(mode_str, True, mode_color)
        self.screen.blit(mode_lbl, (x + 4, y + 17))

        pygame.draw.rect(self.screen, C_BG, (x + 4, y + 30, W - 8, 5))
        pygame.draw.rect(self.screen, mode_color,
                         (x + 4, y + 30, int((W - 8) * turn_progress), 5))

        for i, bit in enumerate(pulse_state[:2]):
            bx = x + 4 + i * 40
            by = y + 40
            col = C_PULSE_ON if bit else C_PULSE_OFF
            pygame.draw.rect(self.screen, col, (bx, by, 32, 32))
            pygame.draw.rect(self.screen, C_GRAY, (bx, by, 32, 32), 1)
            lt = self.font_s.render(str(bit), True, C_WHITE if bit else C_GRAY)
            self.screen.blit(lt, (bx + 16 - lt.get_width() // 2, by + 8))

        for hi, hp in enumerate(history[-16:]):
            for bi, bit in enumerate(hp[:2]):
                bx = x + 115 + hi * 9
                by = y + 40 + bi * 12
                col2 = C_PULSE_ON if bit else C_PULSE_OFF
                pygame.draw.rect(self.screen, col2, (bx, by, 7, 7))

        if phoneme:
            ph_t = self.font_m.render(f"「{phoneme}」", True, (255, 220, 100))
            self.screen.blit(ph_t, (x + 4, y + H - 20))

        audio_c = (80, 220, 100) if audio_on else (100, 100, 120)
        audio_t = self.font_s.render(f"V: 音声 {'● ON' if audio_on else '○ OFF'}", True, audio_c)
        self.screen.blit(audio_t, (x + W - audio_t.get_width() - 4, y + H - 16))

    def draw_backroom(self, bottleneck, manual_bits4: int, audio_on: bool,
                      waveform, mic_data: dict | None):
        """バックルームモードの全画面を描画する。"""
        self.screen.fill((0, 0, 0))
        title = self.font_l.render("SOUND MODE  —  音声入出力確認", True, (255, 120, 180))
        self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 20))

        if bottleneck is not None:
            bn_x = SCREEN_W // 2 - 130
            bn_y = 65
            self.draw_bottleneck_dummy(
                x=bn_x, y=bn_y,
                pulse_state=bottleneck.get_current_pulse() if hasattr(bottleneck, 'get_current_pulse') else None,
                history=bottleneck.get_pulse_history() if hasattr(bottleneck, 'get_pulse_history') else None,
                mode=bottleneck.get_mode() if hasattr(bottleneck, 'get_mode') else 'listen',
                turn_progress=bottleneck.get_turn_progress() if hasattr(bottleneck, 'get_turn_progress') else 0.0,
                phoneme=bottleneck.get_last_phoneme() if hasattr(bottleneck, 'get_last_phoneme') else '',
                audio_on=audio_on,
                is_dummy=True,
            )

        self._draw_output_debug(x=40, y=180, bits4=manual_bits4,
                                audio_on=audio_on, waveform=waveform)
        self._draw_input_debug(x=SCREEN_W // 2 + 40, y=180,
                               mic_data=mic_data, expected_bits4=manual_bits4)

        hints = ["0～9 / a～f: 手動パルス入力    V: 音声ON/OFF    ESC/M: メニューへ戻る"]
        for i, h in enumerate(hints):
            ht = self.font_s.render(h, True, C_GRAY)
            self.screen.blit(ht, (SCREEN_W // 2 - ht.get_width() // 2, SCREEN_H - 25 + i * 18))

    def _draw_output_debug(self, x: int, y: int, bits4: int, audio_on: bool, waveform):
        """OUTPUT DEBUGパネル。"""
        from config import PHONEME_TABLE, AUDIO_FRAME_SAMPLES
        W, H = SCREEN_W // 2 - 60, 460
        bg = pygame.Surface((W, H), pygame.SRCALPHA)
        bg.fill((5, 5, 15, 230))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (255, 120, 180), (x, y, W, H), 1, border_radius=6)
        title = self.font_m.render("OUTPUT DEBUG", True, (255, 120, 180))
        self.screen.blit(title, (x + 8, y + 8))

        ry = y + 36
        bits2 = bits4 & 0x3
        phoneme_char = PHONEME_TABLE.get(bits2, "?")
        rows = [
            ("Input bits2", f"{bits2:02b}  ({bits2})"),
            ("Phoneme",     f"「{phoneme_char}」"),
        ]
        for label, val in rows:
            lt = self.font_s.render(f"{label}:", True, C_GRAY)
            vt = self.font_m.render(val, True, C_WHITE)
            self.screen.blit(lt, (x + 8, ry))
            self.screen.blit(vt, (x + 140, ry - 2))
            ry += 26

        ry += 8
        bits_list = [(bits2 >> (1 - i)) & 1 for i in range(2)]
        dt = self.font_s.render("Bits:", True, C_GRAY)
        self.screen.blit(dt, (x + 8, ry))
        for i, bit in enumerate(bits_list):
            bx = x + 60 + i * 40
            col = C_PULSE_ON if bit else C_PULSE_OFF
            pygame.draw.rect(self.screen, col, (bx, ry - 2, 30, 30))
            pygame.draw.rect(self.screen, C_GRAY, (bx, ry - 2, 30, 30), 1)
            bt = self.font_s.render(str(bit), True, C_WHITE if bit else C_GRAY)
            self.screen.blit(bt, (bx + 15 - bt.get_width() // 2, ry + 6))
        ry += 40

        audio_c = (80, 220, 100) if audio_on else (100, 100, 120)
        at = self.font_s.render(f"音声: {'ON' if audio_on else 'OFF'}", True, audio_c)
        self.screen.blit(at, (x + 8, ry))
        ry += 24

        ww = W - 16; wh = 80
        pygame.draw.rect(self.screen, (20, 20, 40), (x + 8, ry, ww, wh), border_radius=4)
        pygame.draw.rect(self.screen, (60, 60, 80), (x + 8, ry, ww, wh), 1)
        if waveform is not None and len(waveform) > 0:
            import numpy as np
            n_draw = min(ww, len(waveform))
            step = max(1, len(waveform) // n_draw)
            pts = []
            for j in range(n_draw):
                sv = float(waveform[j * step])
                px = x + 8 + j
                py = int(ry + wh // 2 - sv * (wh // 2 - 2))
                py = max(ry, min(ry + wh - 1, py))
                pts.append((px, py))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, (100, 200, 255), False, pts, 1)
        else:
            nt = self.font_s.render("[play to preview]", True, (60, 80, 100))
            self.screen.blit(nt, (x + 8 + ww // 2 - nt.get_width() // 2, ry + wh // 2 - 6))

    def _draw_input_debug(self, x: int, y: int, mic_data: dict | None, expected_bits4: int):
        """INPUT DEBUGパネル。"""
        from config import PHONEME_TABLE
        W, H = SCREEN_W // 2 - 60, 460
        bg = pygame.Surface((W, H), pygame.SRCALPHA)
        bg.fill((5, 15, 5, 230))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (80, 220, 120), (x, y, W, H), 1, border_radius=6)
        title = self.font_m.render("INPUT DEBUG", True, (80, 220, 120))
        self.screen.blit(title, (x + 8, y + 8))

        if mic_data is None or not mic_data.get("available", False):
            nt = self.font_m.render("MIC UNAVAILABLE", True, (200, 80, 80))
            self.screen.blit(nt, (x + W // 2 - nt.get_width() // 2, y + H // 2 - 10))
            st = self.font_s.render("pyaudioマイクが検出できませんでした", True, C_GRAY)
            self.screen.blit(st, (x + W // 2 - st.get_width() // 2, y + H // 2 + 20))
            return

        ry = y + 36
        decoded_bits = mic_data.get("decoded_bits", 0)
        expected_char = PHONEME_TABLE.get(expected_bits4 & 0x3, "?")
        decoded_char  = PHONEME_TABLE.get(decoded_bits & 0x3, "?")
        match = ((decoded_bits & 0x3) == (expected_bits4 & 0x3))

        rows = [
            ("Detected F1", f"{mic_data.get('f1', 0):.0f} Hz"),
            ("Detected F2", f"{mic_data.get('f2', 0):.0f} Hz"),
            ("Vowel",       mic_data.get("vowel", "?")),
        ]
        for label, val in rows:
            lt = self.font_s.render(f"{label}:", True, C_GRAY)
            vt = self.font_m.render(val, True, C_WHITE)
            self.screen.blit(lt, (x + 8, ry))
            self.screen.blit(vt, (x + 140, ry - 2))
            ry += 26
        ry += 8

        decoded_bits2 = decoded_bits & 0x3
        bits_list = [(decoded_bits2 >> (1 - i)) & 1 for i in range(2)]
        dt = self.font_s.render("Decoded:", True, C_GRAY)
        self.screen.blit(dt, (x + 8, ry))
        for i, bit in enumerate(bits_list):
            bx = x + 100 + i * 40
            col = C_PULSE_ON if bit else C_PULSE_OFF
            pygame.draw.rect(self.screen, col, (bx, ry - 2, 30, 30))
            pygame.draw.rect(self.screen, C_GRAY, (bx, ry - 2, 30, 30), 1)
            bt = self.font_s.render(str(bit), True, C_WHITE if bit else C_GRAY)
            self.screen.blit(bt, (bx + 15 - bt.get_width() // 2, ry + 6))
        dc = self.font_m.render(f"「{decoded_char}」", True, C_WHITE)
        self.screen.blit(dc, (x + 200, ry - 2))
        ry += 40

        exp_bits2 = expected_bits4 & 0x3
        exp_bits_list = [(exp_bits2 >> (1 - i)) & 1 for i in range(2)]
        et = self.font_s.render("Expected:", True, C_GRAY)
        self.screen.blit(et, (x + 8, ry))
        for i, bit in enumerate(exp_bits_list):
            bx = x + 100 + i * 40
            col = (80, 120, 200) if bit else C_PULSE_OFF
            pygame.draw.rect(self.screen, col, (bx, ry - 2, 30, 30))
            pygame.draw.rect(self.screen, C_GRAY, (bx, ry - 2, 30, 30), 1)
            bt = self.font_s.render(str(bit), True, C_WHITE if bit else C_GRAY)
            self.screen.blit(bt, (bx + 15 - bt.get_width() // 2, ry + 6))
        ec = self.font_m.render(f"「{expected_char}」", True, (180, 180, 220))
        self.screen.blit(ec, (x + 200, ry - 2))
        ry += 40

        match_c = (80, 220, 100) if match else (220, 80, 80)
        match_sym = "✓ MATCH" if match else "✗ MISMATCH"
        mt = self.font_m.render(f"Match: {match_sym}", True, match_c)
        self.screen.blit(mt, (x + 8, ry))

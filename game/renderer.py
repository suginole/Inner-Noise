"""
renderer.py — 描画システム（Sage-Brute版）

3パネルモニター:
  左: SAGE（青系・5列）
  中: BOTTLENECK（黄系）
  右: BRUTE（赤系・5列）

摂取履歴パネル（中央下部）:
  直近5回の摂取キノコ種類・中毒カウント・バイオーム
"""
import math
import pygame
from config import *


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
        self._static_buf:    pygame.Surface | None = None
        self._minimap_surf:  pygame.Surface | None = None
        self._cone_surf:     pygame.Surface | None = None

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
                # バイオーム別の枠色
                bc = {'W': (60,120,200), 'G': (80,160,80), 'M': (160,120,60)}.get(m.biome, C_GRAY)
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
        # 向き線
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
    def draw_rnn_monitor_panels(self, genome, bottleneck,
                                 x=0, y=0, panel_w=280, panel_h=220):
        """3パネルモニター: SAGE / BOTTLENECK / BRUTE"""
        gap = 8
        cx_bn    = x + panel_w + gap
        cx_brute = x + (panel_w + gap) * 2

        self._draw_sage_panel(genome.sage, x, y, panel_w, panel_h)
        self._draw_bottleneck_panel(bottleneck, cx_bn, y, panel_w, panel_h)
        self._draw_brute_panel(genome.brute, cx_brute, y, panel_w, panel_h)
        self._draw_flow_arrows(x, y, panel_w, panel_h, gap, bottleneck)

    def _draw_sage_panel(self, sage, x, y, w, h):
        """左パネル: SAGE（青系）5列"""
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

        node_r = 3; node_gap = 12; row_top = y + 18
        col_l3    = x + int(w * 0.20)
        col_buf   = x + int(w * 0.40)
        col_gru   = x + int(w * 0.60)
        col_pulse = x + w - 16

        def bip(v):
            t = min(1.0, abs(float(v)))
            return (int(40+215*t), 40, 40) if float(v) >= 0 else (40, 40, int(40+215*t))

        # 第三層FF列（通常/バッファ区切り）
        n = len(l3_acts); sy = row_top + max(0, (h-28-n*node_gap)//2)
        for i, v in enumerate(l3_acts):
            ny = sy + i * node_gap
            border = (80, 130, 220) if i < SAGE_L3_NORMAL else (60, 60, 120)
            pygame.draw.circle(self.screen, bip(v), (col_l3, ny), node_r)
            pygame.draw.circle(self.screen, border, (col_l3, ny), node_r, 1)
        if n >= SAGE_L3_NORMAL:
            sep = sy + SAGE_L3_NORMAL * node_gap - node_gap // 2
            pygame.draw.line(self.screen, (60,80,140), (col_l3-8, sep), (col_l3+8, sep), 1)

        # バッファGRU列（紫系）
        buf_c = (180, 80, 220) if buf_act else (80, 30, 100)
        nb = len(buf_acts); syb = row_top + max(0, (h-28-nb*node_gap)//2)
        for i, v in enumerate(buf_acts):
            ny = syb + i * node_gap
            pygame.draw.circle(self.screen, bip(v), (col_buf, ny), node_r)
            pygame.draw.circle(self.screen, buf_c, (col_buf, ny), node_r, 1)
        if not buf_act:
            lt = self.font_s.render("■", True, (60, 20, 80))
            self.screen.blit(lt, (col_buf - lt.get_width()//2, syb - 12))

        # 記憶GRU列（継承/非継承）
        ng = len(gru_acts); syg = row_top + max(0, (h-28-ng*node_gap)//2)
        for i, v in enumerate(gru_acts):
            ny = syg + i * node_gap
            border = (80, 120, 200) if i < SAGE_MEM_INHERIT else (30, 50, 80)
            pygame.draw.circle(self.screen, bip(v), (col_gru, ny), node_r)
            pygame.draw.circle(self.screen, border, (col_gru, ny), node_r, 1)
        if ng > SAGE_MEM_INHERIT:
            sep = syg + SAGE_MEM_INHERIT * node_gap - node_gap // 2
            pygame.draw.line(self.screen, (60,80,120), (col_gru-8, sep), (col_gru+8, sep), 1)

        # パルス符号化列
        syp = row_top + max(0, (h-28-2*26)//2)
        for i, bit in enumerate(pulse_bits):
            ny = syp + i * 26
            c = C_PULSE_ON if bit else C_PULSE_OFF
            pygame.draw.rect(self.screen, c, (col_pulse-9, ny, 16, 18))
            pygame.draw.rect(self.screen, C_GRAY, (col_pulse-9, ny, 16, 18), 1)
            bt = self.font_s.render(str(bit), True, C_WHITE if bit else C_GRAY)
            self.screen.blit(bt, (col_pulse-9+8-bt.get_width()//2, ny+3))

        # ラベル
        for lbl, lx, lc in [("L3", col_l3, (90,130,220)), ("BUF", col_buf, (180,80,220)),
                              ("GRU", col_gru, (100,140,220)), ("P", col_pulse, (200,200,80))]:
            lt = self.font_s.render(lbl, True, lc)
            self.screen.blit(lt, (lx - lt.get_width()//2, y+h-14))

    def _draw_brute_panel(self, brute, x, y, w, h):
        """右パネル: BRUTE（赤系）5列"""
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

        node_r = 3; node_gap = 12; row_top = y + 18
        col_l3    = x + int(w * 0.20)
        col_buf   = x + int(w * 0.40)
        col_gru   = x + int(w * 0.60)
        col_out   = x + int(w * 0.80)
        bar_w     = w - (col_out - x) - 6

        def bip(v):
            t = min(1.0, abs(float(v)))
            return (int(40+215*t), 40, 40) if float(v) >= 0 else (40, 40, int(40+215*t))

        # 第三層FF列
        n = len(l3_acts); sy = row_top + max(0, (h-28-n*node_gap)//2)
        for i, v in enumerate(l3_acts):
            ny = sy + i * node_gap
            border = (220, 80, 80) if i < BRUTE_L3_NORMAL else (100, 30, 30)
            pygame.draw.circle(self.screen, bip(v), (col_l3, ny), node_r)
            pygame.draw.circle(self.screen, border, (col_l3, ny), node_r, 1)
        if n >= BRUTE_L3_NORMAL:
            sep = sy + BRUTE_L3_NORMAL * node_gap - node_gap // 2
            pygame.draw.line(self.screen, (120,40,40), (col_l3-8, sep), (col_l3+8, sep), 1)

        # バッファGRU列
        buf_c = (180, 80, 220) if buf_act else (80, 30, 100)
        nb = len(buf_acts); syb = row_top + max(0, (h-28-nb*node_gap)//2)
        for i, v in enumerate(buf_acts):
            ny = syb + i * node_gap
            pygame.draw.circle(self.screen, bip(v), (col_buf, ny), node_r)
            pygame.draw.circle(self.screen, buf_c, (col_buf, ny), node_r, 1)
        if not buf_act:
            lt = self.font_s.render("■", True, (60, 20, 80))
            self.screen.blit(lt, (col_buf - lt.get_width()//2, syb - 12))

        # 記憶GRU列
        ng = len(gru_acts); syg = row_top + max(0, (h-28-ng*node_gap)//2)
        for i, v in enumerate(gru_acts):
            ny = syg + i * node_gap
            border = (200, 80, 80) if i < BRUTE_MEM_INHERIT else (60, 20, 20)
            pygame.draw.circle(self.screen, bip(v), (col_gru, ny), node_r)
            pygame.draw.circle(self.screen, border, (col_gru, ny), node_r, 1)
        if ng > BRUTE_MEM_INHERIT:
            sep = syg + BRUTE_MEM_INHERIT * node_gap - node_gap // 2
            pygame.draw.line(self.screen, (100,40,40), (col_gru-8, sep), (col_gru+8, sep), 1)

        # OUTPUT列
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

        # ラベル
        for lbl, lx, lc in [("L3", col_l3, (200,80,80)), ("BUF", col_buf, (180,80,220)),
                              ("GRU", col_gru, (210,90,90)), ("OUT", col_out+bar_w//2, (240,120,120))]:
            lt = self.font_s.render(lbl, True, lc)
            self.screen.blit(lt, (lx - lt.get_width()//2, y+h-14))

    def _draw_bottleneck_panel(self, bottleneck, x, y, w, h):
        """中央パネル: BOTTLENECK（黄系）"""
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((20, 18, 5, 220))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (200, 180, 50), (x, y, w, h), 1, border_radius=4)

        title = self.font_s.render("BOTTLENECK  2bits / 10Hz", True, (220, 200, 80))
        self.screen.blit(title, (x + 4, y + 3))

        pulse   = bottleneck.get_current_pulse() if hasattr(bottleneck, 'get_current_pulse') else [0, 0]
        mode    = bottleneck.get_mode()           if hasattr(bottleneck, 'get_mode')           else 'listen'
        prog    = bottleneck.get_display_progress() if hasattr(bottleneck, 'get_display_progress') else 0.0
        hist    = bottleneck.get_display_history()  if hasattr(bottleneck, 'get_display_history')  else []
        phoneme = bottleneck.get_display_phoneme()  if hasattr(bottleneck, 'get_display_phoneme')  else ''
        direction = getattr(bottleneck, 'direction', 'S→B')
        turn      = getattr(bottleneck, '_turn', 0)

        # 方向・ターン表示
        dir_c = (100, 160, 255) if direction == 'S→B' else (100, 200, 120)
        dir_t = self.font_m.render(f"{direction}  Turn:{turn}", True, dir_c)
        self.screen.blit(dir_t, (x + 6, y + 18))

        # 進捗バー
        pygame.draw.rect(self.screen, (30, 30, 50), (x + 6, y + 38, w - 12, 5))
        pygame.draw.rect(self.screen, dir_c, (x + 6, y + 38, int((w - 12) * prog), 5))

        # 現在パルス
        for i, bit in enumerate(pulse[:2]):
            bx = x + 6 + i * 36
            bc = C_PULSE_ON if bit else C_PULSE_OFF
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

        # スロット履歴
        padded = hist[-HIST_SLOTS:]
        for hi, hp in enumerate(padded):
            for bi, bit in enumerate(hp[:2]):
                bx = x + 6 + hi * slot_w
                by = hy + bi * 10
                c = C_PULSE_ON if bit else C_PULSE_OFF
                pygame.draw.rect(self.screen, c, (bx, by, slot_w - 2, 8))

    def _draw_flow_arrows(self, x, y, panel_w, panel_h, gap, bottleneck):
        """パネル間の接続矢印（両矢印ともボトルネック側を向く）。"""
        import time
        flash = (int(time.time() * 5) % 2 == 0)
        direction = getattr(bottleneck, 'direction', 'S→B')
        mid_y = y + panel_h // 2
        ax1 = x + panel_w; ax2 = ax1 + gap
        ax3 = ax2 + panel_w; ax4 = ax3 + gap

        c1 = (100, 160, 255) if (direction == 'S→B' and flash) else (50, 60, 80)
        pygame.draw.line(self.screen, c1, (ax1, mid_y), (ax2, mid_y), 2)
        pygame.draw.polygon(self.screen, c1, [(ax2, mid_y), (ax2-5, mid_y-4), (ax2-5, mid_y+4)])

        c2 = (100, 200, 120) if (direction == 'B→S' and flash) else (50, 80, 50)
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

        # バイオーム
        biome_map = {'W': ('沼', (60,120,200)), 'G': ('平地', (80,160,80)), 'M': ('山', (160,120,60))}
        if hasattr(agent, 'field'):
            biome = agent.field.biome_at(agent.pos.x, agent.pos.y)
            bname, bcolor = biome_map.get(biome, ('?', C_GRAY))
            bt = self.font_s.render(f"Biome: {bname}", True, bcolor)
            self.screen.blit(bt, (x + w - bt.get_width() - 6, y + 4))

        # 直近5回の摂取履歴
        history = getattr(agent, 'intake_history', [])
        toxic_c = getattr(agent, 'toxic_count', 0)
        for i, sk in enumerate(history[-5:]):
            bx = x + 6 + i * 54
            biome_c = {'W': (60,120,200), 'G': (80,160,80), 'M': (160,120,60)}.get(sk[0], C_GRAY)
            grade_c = C_FOOD_HI if sk[1] == 'premium' else C_FOOD
            pygame.draw.rect(self.screen, biome_c, (bx, y+20, 48, 24), border_radius=3)
            pygame.draw.rect(self.screen, grade_c, (bx, y+20, 48, 24), 1, border_radius=3)
            label = f"{sk[0]}{sk[2]}"
            lt = self.font_s.render(label, True, C_WHITE)
            self.screen.blit(lt, (bx + 24 - lt.get_width()//2, y + 26))

        # 中毒カウント
        tc = self.font_s.render(f"Toxic: {toxic_c}/{TOXIC_COUNT}", True,
                                 (220, 80, 80) if toxic_c >= TOXIC_COUNT - 1 else C_GRAY)
        self.screen.blit(tc, (x + 6, y + h - 16))

    # ----------------------------------------------------------------
    def draw_ga_overlay(self, stats: dict, alive: int, x=20, y=20):
        """GA統計HUD。"""
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
        """適応度グラフ。"""
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
        """メニュー画面。"""
        self.screen.fill(C_BG)
        title = self.font_l.render("INNER NOISE — Sage & Brute", True, C_WHITE)
        self.screen.blit(title, (SCREEN_W//2 - title.get_width()//2, 180))
        sub = self.font_m.render("10Hz Bottleneck Communication", True, C_GRAY)
        self.screen.blit(sub, (SCREEN_W//2 - sub.get_width()//2, 220))
        options = [
            ("2", "GA MODE",      "GAエージェントを学習させる",   (100, 200, 120)),
            ("3", "FAST MODE",    "高速学習モード",               (255, 200, 50)),
            ("L", "LOAD MODEL",   "保存済みモデルをロードする",   (180, 140, 220)),
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
        y = 20
        self.screen.blit(bg, (x, y))
        self.screen.blit(t, (x + 12, y + 7))

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

"""
renderer.py — 描画システム
カメラ追従・地形・餌・ゴール・車・HUDを描画する。
"""
import math
import pygame
from config import *


class Renderer:
    """ゲーム画面の描画を担当する。

    レンダリング戦略:
      1. 地形ブリット: source_rectで画面分のみ転送（全体の1/11に削減）
      2. 静的レイヤ: 地形・ゴール・スタートは小型バッファに一度だけ複写
      3. 動的レイヤ: 車・餌・コーンは毎フレーム描画（小さいので高速）
      4. HUD: 画面座標固定のため常に高速
    """

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font_s = pygame.font.SysFont("monospace", 12)
        self.font_m = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 22, bold=True)

        # ---- キャッシュ群 ----
        self._field_surf_cache = None   # 不使用（field.get_surface()内部キャッシュに移行済み）

        # 静的バッファ: 地形 + ゴール + スタートを一度だけ合成した画面サイズのバッファ
        # カメラ移動時はここからsource_rectで切り出すだけ
        self._static_buf: pygame.Surface | None = None
        self._static_cam: pygame.Vector2 | None = None   # 前フレームのカメラ位置

        # ミニマップ地形（静的、初回のみ生成）
        self._minimap_surf: pygame.Surface | None = None

        # 視野コーン（SRCALPHA Surfaceを一度だけ確保して再利用）
        self._cone_surf: pygame.Surface | None = None

        # 固定テキストキャッシュ
        self._goal_label = None
        self._start_label = None

    # ----------------------------------------------------------------
    def calc_camera(self, car_pos: pygame.Vector2) -> pygame.Vector2:
        """カメラオフセットを計算する（車を画面中央に追従）。"""
        cx = car_pos.x - SCREEN_W // 2
        cy = car_pos.y - SCREEN_H // 2
        cx = max(0, min(WORLD_W - SCREEN_W, cx))
        cy = max(0, min(WORLD_H - SCREEN_H, cy))
        return pygame.Vector2(cx, cy)

    # ----------------------------------------------------------------
    def draw_vision_cone(self, car, cam: pygame.Vector2):
        """車の視野コーン（±45度 / VISION_RANGE）を描画する。
        SRCALPHA Surfaceは使わず、輪郭線のみ描画（高速）。
        """
        sx = int(car.pos.x - cam.x)
        sy = int(car.pos.y - cam.y)
        facing_rad = math.radians(car.angle)
        half_fov   = math.radians(VISION_ANGLE_DEG)
        r          = int(VISION_RANGE)

        pts = [(sx, sy)]
        steps = 8
        for i in range(steps + 1):
            t = i / steps
            a = facing_rad + half_fov * (2 * t - 1)
            pts.append((sx + math.cos(a) * r, sy + math.sin(a) * r))

        # 輪郭線のみ（内側塗りつぶしなし）→ SRCALPHA blit 不要
        pygame.draw.polygon(self.screen, (80, 70, 20), pts, 1)   # 暗い輪郭
        # 左右のレイ線
        pygame.draw.line(self.screen, (100, 90, 30),
                         (sx, sy), pts[1], 1)
        pygame.draw.line(self.screen, (100, 90, 30),
                         (sx, sy), pts[-1], 1)

    def _build_static_buf(self, field) -> pygame.Surface:
        """地形全体にゴール・スタートを合成した静的バッファを作る。
        地形シードごとに1回だけ生成。カメラ移動時は source_rect で切り出す。
        """
        buf = field.get_surface().copy()   # 4000x4000のコピー（初回のみ）

        # ゴール
        if self._goal_label is None:
            self._goal_label = self.font_m.render("GOAL", True, C_GOAL)
        gx, gy = int(field.goal_pos[0]), int(field.goal_pos[1])
        pygame.draw.circle(buf, C_GOAL, (gx, gy), GOAL_RADIUS, 3)
        buf.blit(self._goal_label, (gx - 20, gy - 30))

        # スタート
        sx, sy = int(field.start_pos[0]), int(field.start_pos[1])
        pygame.draw.circle(buf, C_WHITE, (sx, sy), 20, 2)
        if self._start_label is None:
            self._start_label = self.font_s.render("START", True, C_WHITE)
        buf.blit(self._start_label, (sx - 20, sy - 30))

        return buf

    def draw_field(self, field, cam: pygame.Vector2):
        """地形・餌・ゴールを描画する。
        地形・ゴール・スタートは静的バッファから source_rect で切り出す。
        カメラ移動分のピクセルのみ転送するので高速。
        """
        # 静的バッファの初回生成（terrain_seed単位でキャッシュ済み）
        if self._static_buf is None:
            self._static_buf = self._build_static_buf(field)

        # 地形: source_rectで画面分のみ切り出して転送（高速）
        src_rect = pygame.Rect(int(cam.x), int(cam.y), SCREEN_W, SCREEN_H)
        self.screen.blit(self._static_buf, (0, 0), src_rect)

        # 餌（動的、毎フレーム描画。画面内のものだけ）
        for food_pos, is_premium in field.foods:
            fx = int(food_pos.x - cam.x)
            fy = int(food_pos.y - cam.y)
            if -20 < fx < SCREEN_W + 20 and -20 < fy < SCREEN_H + 20:
                if is_premium:
                    pygame.draw.circle(self.screen, (255, 180, 0), (fx, fy), 10)
                    pygame.draw.circle(self.screen, (255, 240, 100), (fx, fy), 10, 2)
                    pygame.draw.circle(self.screen, (255, 255, 200), (fx, fy), 4)
                else:
                    pygame.draw.circle(self.screen, C_FOOD, (fx, fy), 6)
                    pygame.draw.circle(self.screen, C_WHITE, (fx, fy), 6, 1)

    # ----------------------------------------------------------------
    def draw_minimap(self, field, car_pos: pygame.Vector2,
                     goal_pos: tuple, x: int = 20, y: int = 20,
                     w: int = 160, h: int = 160):
        """右上にミニマップを描画する。
        地形は静的バッファを pygame.transform.scale でスケールダウン（高速）。
        """
        mx = SCREEN_W - w - x
        my = y

        # 地形キャッシュ: 静的バッファをミニマップサイズにスケール（初回のみ）
        if self._minimap_surf is None:
            if self._static_buf is not None:
                # 静的バッファ（4000x4000）をミニマップサイズにスケールダウン
                self._minimap_surf = pygame.transform.scale(self._static_buf, (w, h))
            else:
                # 静的バッファがまだない場合は地形サーフェスを直接スケール
                self._minimap_surf = pygame.transform.scale(
                    field.get_surface(), (w, h))

        self.screen.blit(self._minimap_surf, (mx, my))
        pygame.draw.rect(self.screen, C_GRAY, (mx, my, w, h), 1)

        scale_x = w / WORLD_W
        scale_y = h / WORLD_H

        # 餌（高級餌は金色）
        for food_pos, is_premium in field.foods:
            fx = int(mx + food_pos.x * scale_x)
            fy = int(my + food_pos.y * scale_y)
            c = (255, 180, 0) if is_premium else C_FOOD
            pygame.draw.circle(self.screen, c, (fx, fy), 2 if is_premium else 1)

        # ゴール
        gx = int(mx + goal_pos[0] * scale_x)
        gy = int(my + goal_pos[1] * scale_y)
        pygame.draw.circle(self.screen, C_GOAL, (gx, gy), 4)

        # 車
        cx = int(mx + car_pos.x * scale_x)
        cy = int(my + car_pos.y * scale_y)
        pygame.draw.circle(self.screen, C_CAR, (cx, cy), 4)

    # ----------------------------------------------------------------
    def draw_hud_player(self, car, bottleneck=None):
        """プレイヤーモードのHUD。"""
        self._draw_energy_bar(car.energy, 20, SCREEN_H - 40)
        self._draw_info(car, 20, SCREEN_H - 80)
        if bottleneck:
            self._draw_pulse_display(bottleneck, SCREEN_W // 2 - 120, SCREEN_H - 70)

    def draw_hud_ga(self, car, ga, bottleneck=None, agent_idx=0, total=1):
        """GAモードのHUD。"""
        self._draw_energy_bar(car.energy, 20, SCREEN_H - 40)
        self._draw_info(car, 20, SCREEN_H - 80)
        self._draw_ga_stats(ga, 20, 20)
        self._draw_agent_progress(agent_idx, total, 20, 120)
        if bottleneck:
            self._draw_pulse_display(bottleneck, SCREEN_W // 2 - 120, SCREEN_H - 70)

    # ----------------------------------------------------------------
    def _draw_energy_bar(self, energy: float, x: int, y: int):
        w, h = 200, 16
        bg = pygame.Surface((w + 4, h + 4), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 180))
        self.screen.blit(bg, (x - 2, y - 2))
        # バー
        t = max(0.0, min(1.0, energy))
        color = (
            int(C_ENERGY_LO[0] + (C_ENERGY_HI[0] - C_ENERGY_LO[0]) * t),
            int(C_ENERGY_LO[1] + (C_ENERGY_HI[1] - C_ENERGY_LO[1]) * t),
            int(C_ENERGY_LO[2] + (C_ENERGY_HI[2] - C_ENERGY_LO[2]) * t),
        )
        pygame.draw.rect(self.screen, C_DARK, (x, y, w, h))
        pygame.draw.rect(self.screen, color, (x, y, int(w * t), h))
        pygame.draw.rect(self.screen, C_GRAY, (x, y, w, h), 1)
        label = self.font_s.render(f"ENERGY {energy*100:.0f}%", True, C_HUD_TEXT)
        self.screen.blit(label, (x + 4, y + 2))

    def _draw_info(self, car, x: int, y: int):
        bg = pygame.Surface((220, 36), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 160))
        self.screen.blit(bg, (x, y))
        t1 = self.font_s.render(
            f"Food:{car.food_collected}  Dist:{car.dist_to_goal:.0f}px", True, C_HUD_TEXT)
        self.screen.blit(t1, (x + 4, y + 4))
        t2 = self.font_s.render(
            f"Speed:{car.speed:.2f}  Reward:{car.energy*100:.0f}", True, C_GRAY)
        self.screen.blit(t2, (x + 4, y + 18))

    def _draw_ga_stats(self, ga, x: int, y: int):
        stats = ga.get_stats()
        h = 112
        bg = pygame.Surface((260, h), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 200))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (194, 24, 91), (x, y, 260, h), 1)
        lines = [
            f"GA  Gen:{stats['generation']:4d}   Pop:{ga.pop_size}",
            f"Best:  {stats['best']:8.1f}",
            f"Avg:   {stats['avg']:8.1f}",
            f"Worst: {stats['worst']:8.1f}",
            f"Species: {stats.get('species', '-'):3}   "
            f"Mut: {stats.get('mut_rate', 0):.3f}/{stats.get('mut_std', 0):.3f}",
        ]
        for i, line in enumerate(lines):
            t = self.font_s.render(line, True, (244, 143, 177))
            self.screen.blit(t, (x + 6, y + 6 + i * 20))

    def _draw_agent_progress(self, idx: int, total: int, x: int, y: int):
        bg = pygame.Surface((260, 24), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 160))
        self.screen.blit(bg, (x, y))
        t = self.font_s.render(
            f"Agent {idx+1}/{total}  (SPACE=skip)", True, C_GRAY)
        self.screen.blit(t, (x + 4, y + 4))

    def _draw_pulse_display(self, bottleneck, x: int, y: int):
        """ボトルネックパルス表示（HUD下部中央）。"""
        mode = bottleneck.get_mode()
        progress = bottleneck.get_turn_progress()
        pulse = bottleneck.get_current_pulse()
        history = bottleneck.get_pulse_history()

        # 背景
        bg = pygame.Surface((240, 60), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 200))
        self.screen.blit(bg, (x, y))

        # モード表示
        mode_color = (144, 202, 249) if mode == "listen" else (165, 214, 167)
        mode_label = "LISTEN S→M" if mode == "listen" else "SPEAK  M→S"
        t = self.font_s.render(mode_label, True, mode_color)
        self.screen.blit(t, (x + 4, y + 4))

        # ターン進捗バー
        pygame.draw.rect(self.screen, C_DARK, (x + 4, y + 18, 232, 6))
        pygame.draw.rect(self.screen, mode_color,
                         (x + 4, y + 18, int(232 * progress), 6))

        # 現在のパルス（4 bits）
        for i, bit in enumerate(pulse):
            bx = x + 4 + i * 28
            by = y + 28
            color = C_PULSE_ON if bit else C_PULSE_OFF
            pygame.draw.rect(self.screen, color, (bx, by, 22, 22))
            pygame.draw.rect(self.screen, C_GRAY, (bx, by, 22, 22), 1)
            label = self.font_s.render(str(bit), True,
                                       C_WHITE if bit else C_GRAY)
            self.screen.blit(label, (bx + 7, by + 5))

        # 直近パルス履歴（小さく）
        for hi, hp in enumerate(history[-16:]):
            for bi, bit in enumerate(hp):
                bx = x + 120 + hi * 8
                by = y + 28 + bi * 8
                color = C_PULSE_ON if bit else C_PULSE_OFF
                pygame.draw.rect(self.screen, color, (bx, by, 6, 6))

    # ----------------------------------------------------------------
    def draw_mode_select(self):
        """モード選択画面を描画する。"""
        self.screen.fill(C_BG)
        title = self.font_l.render("BLIND DRIVING SURVIVAL", True, C_WHITE)
        self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 180))

        sub = self.font_m.render(
            "5Hz / 4bits Bottleneck Communication Architecture", True, C_GRAY)
        self.screen.blit(sub, (SCREEN_W // 2 - sub.get_width() // 2, 220))

        options = [
            ("1", "PLAYER MODE",     "キーボードで車を直接操作する",                   C_CAR),
            ("2", "GA MODE",         "遠伝的アルゴリズムの学習を観察する",             C_CAR_GA),
            ("3", "FAST LEARN MODE", "描画スキップの高速学習モード（Tabで監視切替）", (255, 200, 50)),
        ]
        for i, (key, name, desc, color) in enumerate(options):
            oy = 320 + i * 100
            pygame.draw.rect(self.screen, (20, 22, 35),
                             (SCREEN_W // 2 - 280, oy, 560, 70), border_radius=8)
            pygame.draw.rect(self.screen, color,
                             (SCREEN_W // 2 - 280, oy, 560, 70), 2, border_radius=8)
            k = self.font_l.render(f"[{key}]", True, color)
            self.screen.blit(k, (SCREEN_W // 2 - 240, oy + 18))
            n = self.font_m.render(name, True, C_WHITE)
            self.screen.blit(n, (SCREEN_W // 2 - 180, oy + 12))
            d = self.font_s.render(desc, True, C_GRAY)
            self.screen.blit(d, (SCREEN_W // 2 - 180, oy + 38))

        hint = self.font_s.render("ESC: 終了  /  R: リセット  /  M: モード選択に戻る",
                                  True, C_GRAY)
        self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 40))

    def draw_overlay(self, text: str, color=C_WHITE):
        """画面中央にオーバーレイテキストを表示する。"""
        t = self.font_l.render(text, True, color)
        x = SCREEN_W // 2 - t.get_width() // 2
        y = SCREEN_H // 2 - t.get_height() // 2
        bg = pygame.Surface((t.get_width() + 20, t.get_height() + 12), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 220))
        self.screen.blit(bg, (x - 10, y - 6))
        self.screen.blit(t, (x, y))

    # ----------------------------------------------------------------
    def draw_activation_panel(self, genome, x: int, y: int):
        """
        4層アーキテクチャのアクティベーションをリアルタイムで描画するパネル。
        レイアウト: [INPUT(12)] -> [H1(16)] -> [H2(12)] -> [OUTPUT(3)]
        """
        W_PANEL = 500
        H_PANEL = 220

        bg = pygame.Surface((W_PANEL, H_PANEL), pygame.SRCALPHA)
        bg.fill((8, 10, 18, 220))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (60, 60, 80), (x, y, W_PANEL, H_PANEL), 1)

        title = self.font_s.render(
            "ACTIVATION MONITOR  [INPUT(12) → H1(16) → H2(12) → OUT(3)]",
            True, (160, 160, 200))
        self.screen.blit(title, (x + 6, y + 4))

        # ---- アクティベーション取得 ----
        inp_acts  = getattr(genome, 'last_input_act',   [])
        hid1_acts = getattr(genome, 'last_hidden_act',  [])
        hid2_acts = getattr(genome, 'last_hidden2_act', [])
        out_acts  = getattr(genome, 'last_output_act',  [])

        # 入力層ラベル
        inp_labels = ["E","Ga","Gx","Gy","Fx","Fy"] + \
                     [f"R{i}" for i in range(VISION_RAYS)] + ["Fc"]
        out_labels = ["Acc", "Str", "Brk"]

        # 列位置
        col_inp  = x + 30
        col_hid1 = x + 145
        col_hid2 = x + 280
        col_out  = x + 420
        row_top  = y + 18
        node_r   = 5
        node_gap = 12

        def _node_color(v: float, bipolar: bool = False) -> tuple:
            if bipolar:
                if v >= 0:
                    t = min(1.0, v)
                    return (int(40 + 215 * t), int(40 * (1 - t)), 40)
                else:
                    t = min(1.0, -v)
                    return (40, int(40 * (1 - t)), int(40 + 215 * t))
            else:
                t = max(0.0, min(1.0, v))
                return (int(30 + 200 * t), int(30 + 200 * t), int(30 + 60 * t))

        def _draw_layer(acts, labels, cx, bipolar=False, show_val=False):
            n = len(acts)
            total_h = n * node_gap
            sy = row_top + max(0, (H_PANEL - 28 - total_h) // 2)
            for i, v in enumerate(acts):
                ny = sy + i * node_gap
                c = _node_color(v, bipolar)
                pygame.draw.circle(self.screen, c, (cx, ny), node_r)
                pygame.draw.circle(self.screen, (70, 70, 90), (cx, ny), node_r, 1)
                if i < len(labels):
                    lbl = self.font_s.render(labels[i], True, (90, 90, 110))
                    self.screen.blit(lbl, (cx - node_r - lbl.get_width() - 2, ny - 5))
                if show_val:
                    vt = self.font_s.render(f"{v:.2f}", True, (120, 120, 140))
                    self.screen.blit(vt, (cx + node_r + 3, ny - 5))
            return sy, n

        def _draw_connections(acts_a, sy_a, cx_a, W, cx_b, sy_b, n_b, max_draw=8):
            """2層間の接続線（重みの大きい上位max_draw本のみ）"""
            n_a = len(acts_a)
            # 重みの絶対値でソートして上位max_draw本だけ描画
            pairs = []
            for i in range(min(n_a, 12)):
                for j in range(min(n_b, 12)):
                    try:
                        w = float(W[j, i])
                        pairs.append((abs(w), w, i, j))
                    except Exception:
                        pass
            pairs.sort(reverse=True)
            for _, w, i, j in pairs[:max_draw]:
                iy = sy_a + i * node_gap
                jy = sy_b + j * node_gap
                alpha = min(100, int(abs(w) * 35))
                c = (alpha, 20, 20) if w < 0 else (20, alpha, 20)
                pygame.draw.line(self.screen, c,
                                 (cx_a + node_r, iy), (cx_b - node_r, jy), 1)

        # ---- 各層描画 ----
        sy_inp,  n_inp  = _draw_layer(inp_acts,  inp_labels, col_inp,  bipolar=False)
        sy_hid1, n_hid1 = _draw_layer(hid1_acts, [],         col_hid1, bipolar=True)
        sy_hid2, n_hid2 = _draw_layer(hid2_acts, [],         col_hid2, bipolar=True)
        sy_out,  n_out  = _draw_layer(out_acts,  out_labels, col_out,  bipolar=False, show_val=True)

        # ---- 接続線 ----
        _draw_connections(inp_acts,  sy_inp,  col_inp,  genome.W1, col_hid1, sy_hid1, n_hid1)
        _draw_connections(hid1_acts, sy_hid1, col_hid1, genome.W2, col_hid2, sy_hid2, n_hid2)
        _draw_connections(hid2_acts, sy_hid2, col_hid2, genome.W3, col_out,  sy_out,  n_out)

        # ---- 層ラベル ----
        for lbl_text, lx, color in [
            ("INPUT",  col_inp,  (100, 160, 220)),
            ("H1",     col_hid1, (180, 140,  80)),
            ("H2",     col_hid2, (140, 180,  80)),
            ("OUTPUT", col_out,  (100, 200, 120)),
        ]:
            lt = self.font_s.render(lbl_text, True, color)
            self.screen.blit(lt, (lx - lt.get_width() // 2, y + H_PANEL - 14))

    # ----------------------------------------------------------------
    def draw_player_obs_panel(self, obs: list[float], x: int, y: int):
        """プレイヤーモード用：感覚器官の観測ベクトルをバーチャートで表示。"""
        W_PANEL = 420
        H_PANEL = 200
        bg = pygame.Surface((W_PANEL, H_PANEL), pygame.SRCALPHA)
        bg.fill((8, 10, 18, 220))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (60, 80, 60), (x, y, W_PANEL, H_PANEL), 1)

        title = self.font_s.render("SENSORY INPUT MONITOR", True, (100, 200, 120))
        self.screen.blit(title, (x + 6, y + 4))

        labels = [
            ("ENERGY",      "E",  (80, 220, 100)),
            ("GOAL ANGLE",  "Ga", (100, 180, 255)),
            ("GRAD X",      "Gx", (255, 180, 80)),
            ("GRAD Y",      "Gy", (255, 140, 60)),
            ("FOOD DX",     "Fx", (255, 220, 50)),
            ("FOOD DY",     "Fy", (255, 200, 30)),
        ] + [(f"RAY {i}", f"R{i}", (200, 200, 80)) for i in range(VISION_RAYS)] + \
            [("FOCUS", "Fc", (255, 160, 60))]

        bar_x   = x + 60
        bar_w   = W_PANEL - 70
        bar_h   = 12
        row_gap = 16
        row_y   = y + 18

        for i, (full_lbl, short_lbl, color) in enumerate(labels):
            if i >= len(obs):
                break
            v = obs[i]
            ry = row_y + i * row_gap

            # ラベル
            lt = self.font_s.render(short_lbl, True, color)
            self.screen.blit(lt, (x + 4, ry))

            # バーバック
            pygame.draw.rect(self.screen, (25, 28, 38), (bar_x, ry, bar_w, bar_h))

            # 値に応じたバー（二極値は中央基準）
            if full_lbl in ("GOAL ANGLE", "GRAD X", "GRAD Y", "FOOD DX", "FOOD DY"):
                # -1〜1 の値: 中央から左右に伸びる
                mid = bar_x + bar_w // 2
                fill_w = int(abs(v) * bar_w // 2)
                fill_x = mid if v >= 0 else mid - fill_w
                c = (80, 160, 255) if v >= 0 else (255, 100, 80)
                pygame.draw.rect(self.screen, c, (fill_x, ry, fill_w, bar_h))
                pygame.draw.line(self.screen, (80, 80, 100),
                                 (mid, ry), (mid, ry + bar_h), 1)
            else:
                # 0〜1 の値: 左から右に伸びる
                t = max(0.0, min(1.0, v))
                fill_w = int(t * bar_w)
                pygame.draw.rect(self.screen, color, (bar_x, ry, fill_w, bar_h))

            pygame.draw.rect(self.screen, (50, 55, 65), (bar_x, ry, bar_w, bar_h), 1)

            # 数値
            vt = self.font_s.render(f"{v:+.2f}", True, (120, 120, 140))
            self.screen.blit(vt, (bar_x + bar_w + 4, ry))

    # ----------------------------------------------------------------
    def draw_fitness_graph(self, ga, x: int, y: int, w: int = 260, h: int = 100):
        """適応度の推移グラフ（上段）と種数グラフ（下段）を描画する。"""
        if len(ga.best_fitness_history) < 2:
            return

        h_fit = int(h * 0.65)
        h_sp  = h - h_fit - 2

        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 200))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (194, 24, 91), (x, y, w, h), 1)

        # ---- 適応度グラフ ----
        hist     = ga.best_fitness_history[-w:]
        avg_hist = ga.avg_fitness_history[-w:]
        all_vals = hist + avg_hist
        lo = min(all_vals) - 1
        hi_v = max(all_vals) + 1
        rng  = hi_v - lo if hi_v != lo else 1
        n    = len(hist)

        def to_fit(v, i):
            px = x + int(i * w / n)
            py = y + h_fit - int((v - lo) / rng * (h_fit - 4)) - 2
            return px, py

        pts_b = [to_fit(v, i) for i, v in enumerate(hist)]
        if len(pts_b) > 1:
            pygame.draw.lines(self.screen, (244, 143, 177), False, pts_b, 1)
        pts_a = [to_fit(v, i) for i, v in enumerate(avg_hist)]
        if len(pts_a) > 1:
            pygame.draw.lines(self.screen, C_GRAY, False, pts_a, 1)

        label = self.font_s.render("Fitness  best/avg", True, C_GRAY)
        self.screen.blit(label, (x + 4, y + 2))

        # ---- 種数グラフ ----
        sp_hist = getattr(ga, 'species_count_history', [])[-w:]
        if len(sp_hist) > 1:
            sy0 = y + h_fit + 2
            sp_max = max(sp_hist) + 1
            n_sp = len(sp_hist)
            pts_sp = []
            for i, v in enumerate(sp_hist):
                px = x + int(i * w / n_sp)
                py = sy0 + h_sp - int(v / sp_max * (h_sp - 2)) - 1
                pts_sp.append((px, py))
            pygame.draw.lines(self.screen, (100, 200, 255), False, pts_sp, 1)
            sp_label = self.font_s.render(
                f"Species: {sp_hist[-1]}", True, (100, 200, 255))
            self.screen.blit(sp_label, (x + 4, sy0 + 1))

"""
renderer.py — 描画システム
カメラ追従・地形・餌・ゴール・車・HUDを描画する。
"""
import math
import pygame
from config import *


class Renderer:
    """ゲーム画面の描画を担当する。"""

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font_s = pygame.font.SysFont("monospace", 12)
        self.font_m = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 22, bold=True)
        self._field_surf_cache = None

    # ----------------------------------------------------------------
    def calc_camera(self, car_pos: pygame.Vector2) -> pygame.Vector2:
        """カメラオフセットを計算する（車を画面中央に追従）。"""
        cx = car_pos.x - SCREEN_W // 2
        cy = car_pos.y - SCREEN_H // 2
        cx = max(0, min(WORLD_W - SCREEN_W, cx))
        cy = max(0, min(WORLD_H - SCREEN_H, cy))
        return pygame.Vector2(cx, cy)

    # ----------------------------------------------------------------
    def draw_field(self, field, cam: pygame.Vector2):
        """地形・餌・ゴールを描画する。"""
        # 地形（キャッシュサーフェスをブリット）
        surf = field.get_surface()
        self.screen.blit(surf, (-cam.x, -cam.y))

        # ゴール
        gx = int(field.goal_pos[0] - cam.x)
        gy = int(field.goal_pos[1] - cam.y)
        pygame.draw.circle(self.screen, C_GOAL, (gx, gy), GOAL_RADIUS, 3)
        label = self.font_m.render("GOAL", True, C_GOAL)
        self.screen.blit(label, (gx - 20, gy - 30))

        # スタート
        sx = int(field.start_pos[0] - cam.x)
        sy = int(field.start_pos[1] - cam.y)
        pygame.draw.circle(self.screen, C_WHITE, (sx, sy), 20, 2)

        # 餌
        for food in field.foods:
            fx = int(food.x - cam.x)
            fy = int(food.y - cam.y)
            if -20 < fx < SCREEN_W + 20 and -20 < fy < SCREEN_H + 20:
                pygame.draw.circle(self.screen, C_FOOD, (fx, fy), 6)
                pygame.draw.circle(self.screen, C_WHITE, (fx, fy), 6, 1)

    # ----------------------------------------------------------------
    def draw_minimap(self, field, car_pos: pygame.Vector2,
                     goal_pos: tuple, x: int = 20, y: int = 20,
                     w: int = 160, h: int = 160):
        """右上にミニマップを描画する。"""
        mx = SCREEN_W - w - x
        my = y
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 200))
        self.screen.blit(bg, (mx, my))
        pygame.draw.rect(self.screen, C_GRAY, (mx, my, w, h), 1)

        # 地形（ダウンサンプル）
        scale_x = w / WORLD_W
        scale_y = h / WORLD_H
        step = max(1, WORLD_W // (w * 2))
        for gy_i in range(0, field.hmap.shape[0], step):
            for gx_i in range(0, field.hmap.shape[1], step):
                hv = float(field.hmap[gy_i, gx_i])
                px = int(mx + gx_i * TILE * scale_x)
                py = int(my + gy_i * TILE * scale_y)
                if hv >= MOUNTAIN_THRESHOLD:
                    c = C_MOUNTAIN
                elif hv <= VALLEY_THRESHOLD:
                    c = C_VALLEY
                else:
                    c = C_PLAIN
                pygame.draw.rect(self.screen, c, (px, py, 2, 2))

        # 餌
        for food in field.foods:
            fx = int(mx + food.x * scale_x)
            fy = int(my + food.y * scale_y)
            pygame.draw.circle(self.screen, C_FOOD, (fx, fy), 2)

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
        bg = pygame.Surface((260, 80), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 200))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (194, 24, 91), (x, y, 260, 80), 1)
        lines = [
            f"GA  Gen:{stats['generation']}",
            f"Best: {stats['best']:.1f}",
            f"Avg:  {stats['avg']:.1f}",
            f"Pop:  {ga.pop_size}",
        ]
        for i, line in enumerate(lines):
            t = self.font_s.render(line, True, (244, 143, 177))
            self.screen.blit(t, (x + 6, y + 6 + i * 16))

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
            ("1", "PLAYER MODE",  "キーボードで車を直接操作する",         C_CAR),
            ("2", "GA MODE",      "遺伝的アルゴリズムの学習を観察する",   C_CAR_GA),
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

    def draw_fitness_graph(self, ga, x: int, y: int, w: int = 260, h: int = 80):
        """GA適応度の推移グラフを描画する。"""
        if len(ga.best_fitness_history) < 2:
            return
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 200))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (194, 24, 91), (x, y, w, h), 1)

        hist = ga.best_fitness_history[-w:]
        avg_hist = ga.avg_fitness_history[-w:]
        all_vals = hist + avg_hist
        lo = min(all_vals) - 1
        hi = max(all_vals) + 1
        rng = hi - lo if hi != lo else 1

        def to_px(v, i):
            px = x + int(i * w / len(hist))
            py = y + h - int((v - lo) / rng * (h - 4)) - 2
            return px, py

        # best
        pts_b = [to_px(v, i) for i, v in enumerate(hist)]
        if len(pts_b) > 1:
            pygame.draw.lines(self.screen, (244, 143, 177), False, pts_b, 1)
        # avg
        pts_a = [to_px(v, i) for i, v in enumerate(avg_hist)]
        if len(pts_a) > 1:
            pygame.draw.lines(self.screen, C_GRAY, False, pts_a, 1)

        label = self.font_s.render("Fitness (best/avg)", True, C_GRAY)
        self.screen.blit(label, (x + 4, y + 2))

"""
main.py — Blind Driving Survival エントリーポイント
モード選択 → プレイヤーモード / GAモード / 高速学習モード
"""
import sys
import os
import copy

import pygame
from config import *
from game.field import Field
from game.car import Car
from game.player_agent import PlayerAgent
from game.ga_agent import GAAgent, GAGenome, GeneticAlgorithm
from game.bottleneck import Bottleneck
from game.renderer import Renderer


# ================================================================
class GameState:
    MENU        = "menu"
    PLAYER      = "player"
    GA          = "ga"          # 通常GAモード（描画あり）
    GA_FAST_CFG = "ga_fast_cfg" # 高速モード設定画面
    GA_FAST     = "ga_fast"     # 高速学習モード（描画完全スキップ）


# カメラ切替閾値
CAMERA_SWITCH_THRESHOLD = 10


class Game:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption(TITLE)
        self.clock    = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.state    = GameState.MENU

        # フィールド（共有）
        self.field: Field | None = None

        # プレイヤーモード
        self.player_car:   Car | None = None
        self.player_agent: PlayerAgent | None = None
        self.player_bn:    Bottleneck | None = None

        # GAモード共通
        self.ga:           GeneticAlgorithm | None = None
        self.ga_agents:    list[GAAgent] = []
        self.ga_frame:     int = 0
        self.ga_running:   bool = False
        self.goal_reached_count: int = 0

        # 前世代の最高適応度ゲノムのスナップショット
        self.prev_best_genome: GAGenome | None = None

        # 高速モード設定
        self.fast_cfg_goal_count: int = 10   # 終了条件（1〜10）
        self.fast_cfg_pop_size:   int = GA_POP_SIZE
        self.fast_cfg_focus:      str = "goal_count"  # 現在フォーカス中の設定項目

        # リセット確認ダイアログ
        self.confirm_reset: bool = False

        self.done_message: str = ""
        self.done_timer:   int = 0

    # ----------------------------------------------------------------
    def init_player_mode(self):
        self.field        = Field(terrain_seed=42, food_episode=0)
        self.player_car   = Car(*self.field.start_pos)
        self.player_agent = PlayerAgent(self.player_car, self.field)
        self.player_bn    = Bottleneck()
        self.done_message = ""

    def init_ga_mode(self, pop_size: int | None = None):
        pop = pop_size if pop_size is not None else GA_POP_SIZE
        self.field      = Field(terrain_seed=42, food_episode=0)
        self.ga         = GeneticAlgorithm(pop_size=pop, seed=0)
        self._spawn_ga_agents()
        self.ga_frame   = 0
        self.ga_running = True
        self.goal_reached_count = 0
        self.prev_best_genome   = None
        self.done_message = ""

    def _spawn_ga_agents(self):
        self.ga_agents = []
        for genome in self.ga.population:
            car = Car(*self.field.start_pos)
            agent = GAAgent(car, self.field, genome)
            self.ga_agents.append(agent)

    # ----------------------------------------------------------------
    def run(self):
        while True:
            # 高速モード中は描画を完全スキップ
            if self.state == GameState.GA_FAST:
                self._run_fast_frame()
                continue

            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                self._handle_event(event)

            self._update(dt)
            self._draw()
            pygame.display.flip()

    def _run_fast_frame(self):
        """
        高速モード専用ループ。
        描画・flip を完全スキップし、イベントキューだけ捌く。
        1ループあたり FAST_STEPS 回の物理+NN計算を実行。
        """
        FAST_STEPS = 50   # 1ループあたりの計算ステップ数

        # イベントキューを捌く（描画なし）
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_TAB:
                    # 監視モードへ切替（描画ループに戻る）
                    self.state = GameState.GA
                    return
                if event.key == pygame.K_RETURN:
                    self.state = GameState.GA
                    return
                if event.key == pygame.K_m:
                    self.state = GameState.MENU
                    return
                if event.key == pygame.K_r:
                    self.confirm_reset = True
                    self.state = GameState.GA_FAST_CFG
                    return

        # 物理+NN計算をFAST_STEPS回実行
        for _ in range(FAST_STEPS):
            done = self._step_ga_once()
            if done:
                break

        # HUDだけ描画（最小限）
        self.screen.fill(C_BG)
        self._draw_fast_overlay()
        pygame.display.flip()

    def _step_ga_once(self) -> bool:
        """
        1フレーム分のGA更新。
        世代進化が発生したら True を返す。
        """
        if not self.ga_running:
            return False

        self.ga_frame += 1
        all_done = True

        for agent in self.ga_agents:
            if not agent.car.alive:
                continue
            all_done = False
            result = agent.step()
            if result["done"]:
                agent.genome.fitness = agent.total_reward
                agent.car.alive = False
                if agent.car.dist_to_goal < GOAL_RADIUS:
                    self.goal_reached_count += 1

        if all_done:
            # 学習完了判定
            if self.goal_reached_count >= self.fast_cfg_goal_count:
                self._evolve_generation()
                self.state = GameState.GA   # 監視モードへ自動遷移
                return True
            self._evolve_generation()
            return True

        return False

    def _evolve_generation(self):
        """世代進化の共通処理。"""
        # 前世代の最高ゲノムをスナップショット保存
        best = self.ga.get_best()
        self.prev_best_genome = copy.deepcopy(best)
        # ダミーデータでアクティベーションを更新（表示用）
        import numpy as np
        dummy_obs = [0.5] * 12
        self.prev_best_genome.forward(dummy_obs)

        self.ga.evolve()
        self.field.reset_foods(food_episode=self.ga.generation)
        self._spawn_ga_agents()
        self.ga_frame = 0
        self.goal_reached_count = 0

    # ----------------------------------------------------------------
    def _handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.confirm_reset:
                    self.confirm_reset = False
                else:
                    pygame.quit()
                    sys.exit()

            # リセット確認ダイアログ中
            if self.confirm_reset:
                if event.key == pygame.K_y:
                    self.confirm_reset = False
                    self.init_ga_mode(pop_size=self.fast_cfg_pop_size)
                    self.state = GameState.GA_FAST
                elif event.key == pygame.K_n or event.key == pygame.K_ESCAPE:
                    self.confirm_reset = False
                return

            if self.state == GameState.MENU:
                if event.key == pygame.K_1:
                    self.init_player_mode()
                    self.state = GameState.PLAYER
                elif event.key == pygame.K_2:
                    self.init_ga_mode()
                    self.state = GameState.GA
                elif event.key == pygame.K_3:
                    self.state = GameState.GA_FAST_CFG

            elif self.state == GameState.GA_FAST_CFG:
                self._handle_fast_cfg_key(event.key)

            elif self.state == GameState.PLAYER:
                if event.key == pygame.K_m:
                    self.state = GameState.MENU
                if event.key == pygame.K_r:
                    self.init_player_mode()

            elif self.state == GameState.GA:
                if event.key == pygame.K_m:
                    self.state = GameState.MENU
                if event.key == pygame.K_r:
                    self.confirm_reset = True
                if event.key == pygame.K_TAB:
                    if self.ga is None:
                        self.init_ga_mode(pop_size=self.fast_cfg_pop_size)
                    self.state = GameState.GA_FAST

    def _handle_fast_cfg_key(self, key):
        """高速モード設定画面のキー操作。"""
        if key == pygame.K_RETURN or key == pygame.K_SPACE:
            # 設定確定 → 高速モード開始
            self.init_ga_mode(pop_size=self.fast_cfg_pop_size)
            self.state = GameState.GA_FAST
            return
        if key == pygame.K_ESCAPE or key == pygame.K_m:
            self.state = GameState.MENU
            return

        # Tab でフォーカス切替
        if key == pygame.K_TAB:
            items = ["goal_count", "pop_size"]
            idx = items.index(self.fast_cfg_focus)
            self.fast_cfg_focus = items[(idx + 1) % len(items)]
            return

        # 上下で値変更
        if self.fast_cfg_focus == "goal_count":
            if key == pygame.K_UP or key == pygame.K_RIGHT:
                self.fast_cfg_goal_count = min(10, self.fast_cfg_goal_count + 1)
            elif key == pygame.K_DOWN or key == pygame.K_LEFT:
                self.fast_cfg_goal_count = max(1, self.fast_cfg_goal_count - 1)
            # 数字キー 1〜9,0(=10)
            for n in range(1, 11):
                kn = getattr(pygame, f"K_{n % 10}", None)
                if kn and key == kn:
                    self.fast_cfg_goal_count = n
        elif self.fast_cfg_focus == "pop_size":
            if key == pygame.K_UP or key == pygame.K_RIGHT:
                self.fast_cfg_pop_size = min(2000, self.fast_cfg_pop_size + 100)
            elif key == pygame.K_DOWN or key == pygame.K_LEFT:
                self.fast_cfg_pop_size = max(50, self.fast_cfg_pop_size - 100)

    # ----------------------------------------------------------------
    def _update(self, dt: float):
        if self.state == GameState.PLAYER:
            self._update_player(dt)
        elif self.state == GameState.GA:
            self._update_ga_monitor(dt)

    def _update_player(self, dt: float):
        if not self.player_car.alive:
            if self.done_timer > 0:
                self.done_timer -= 1
            return
        result = self.player_agent.step()
        if result["done"]:
            if self.player_car.dist_to_goal < GOAL_RADIUS:
                self.done_message = "GOAL!"
            else:
                self.done_message = "GAME OVER"
            self.done_timer = FPS * 3

    def _update_ga_monitor(self, dt: float):
        """監視モード（通常速度）のGA更新。"""
        if not self.ga_running:
            return
        self._step_ga_once()

    # ----------------------------------------------------------------
    def _get_camera_focus(self) -> pygame.Vector2:
        alive_agents = [ag for ag in self.ga_agents if ag.car.alive]
        if not alive_agents:
            return pygame.Vector2(self.field.start_pos)
        if len(alive_agents) > CAMERA_SWITCH_THRESHOLD:
            cx = sum(ag.car.pos.x for ag in alive_agents) / len(alive_agents)
            cy = sum(ag.car.pos.y for ag in alive_agents) / len(alive_agents)
            return pygame.Vector2(cx, cy)
        else:
            best = max(alive_agents, key=lambda ag: ag.total_reward)
            return pygame.Vector2(best.car.pos)

    # ----------------------------------------------------------------
    def _draw(self):
        self.screen.fill(C_BG)

        # リセット確認ダイアログ（最前面）
        if self.confirm_reset:
            self._draw_confirm_reset()
            pygame.display.flip()
            return

        if self.state == GameState.MENU:
            self.renderer.draw_mode_select()

        elif self.state == GameState.GA_FAST_CFG:
            self._draw_fast_cfg()

        elif self.state == GameState.PLAYER:
            cam = self.renderer.calc_camera(self.player_car.pos)
            self.renderer.draw_field(self.field, cam)
            self._ensure_minimap()
            self.renderer.draw_vision_cone(self.player_car, cam)
            self.player_car.draw(self.screen, cam, color=C_CAR)
            self.renderer.draw_minimap(
                self.field, self.player_car.pos, self.field.goal_pos)
            self.renderer.draw_hud_player(self.player_car, None)
            if self.done_message:
                color = C_GOAL if self.done_message == "GOAL!" else C_ENERGY_LO
                self.renderer.draw_overlay(self.done_message, color)
            self.renderer.draw_player_obs_panel(
                self.player_car.get_observation(self.field),
                x=SCREEN_W - 430, y=SCREEN_H - 210)
            hint = self.renderer.font_s.render(
                "M: Menu  R: Reset  Tab: Fast Mode  ESC: Quit", True, C_GRAY)
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 20))

        elif self.state == GameState.GA:
            self._draw_ga_monitor()

    def _ensure_minimap(self):
        if self.renderer._minimap_surf is None and self.renderer._static_buf is not None:
            from pygame import transform
            self.renderer._minimap_surf = transform.scale(
                self.renderer._static_buf, (160, 160))

    # ----------------------------------------------------------------
    def _draw_ga_monitor(self):
        """通常GAモードの描画。"""
        focus = self._get_camera_focus()
        cam   = self.renderer.calc_camera(focus)

        self.renderer.draw_field(self.field, cam)
        self._ensure_minimap()

        alive_count = 0
        best_agent  = None
        for ag in self.ga_agents:
            if not ag.car.alive:
                continue
            alive_count += 1
            if best_agent is None or ag.total_reward > best_agent.total_reward:
                best_agent = ag
            ag.car.draw(self.screen, cam, color=(40, 100, 160))

        if best_agent:
            self.renderer.draw_vision_cone(best_agent.car, cam)
            best_agent.car.draw(self.screen, cam, color=(255, 200, 50))
            if alive_count <= CAMERA_SWITCH_THRESHOLD:
                self._draw_tracking_indicator(best_agent.car, cam)

        self.renderer.draw_minimap(self.field, focus, self.field.goal_pos)
        self._draw_ga_overlay(alive_count, best_agent)
        self.renderer.draw_fitness_graph(self.ga, 20, 150, w=280, h=100)

        # 前世代NNスナップショット（右下）
        if self.prev_best_genome is not None:
            self._draw_snapshot_label()
            self.renderer.draw_activation_panel(
                self.prev_best_genome,
                x=SCREEN_W - 510, y=SCREEN_H - 230)
        elif best_agent:
            self.renderer.draw_activation_panel(
                best_agent.genome,
                x=SCREEN_W - 510, y=SCREEN_H - 230)

        hint = self.renderer.font_s.render(
            "Tab: Fast Mode  R: Reset  M: Menu  ESC: Quit", True, C_GRAY)
        self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 20))

    def _draw_snapshot_label(self):
        """前世代スナップショットのラベルを描画する。"""
        gen = self.ga.generation - 1
        t = self.renderer.font_s.render(
            f"PREV GEN {gen} BEST  (snapshot / dummy input)",
            True, (180, 140, 80))
        self.screen.blit(t, (SCREEN_W - 510, SCREEN_H - 245))

    # ----------------------------------------------------------------
    def _draw_ga_overlay(self, alive_count: int, best_agent):
        stats = self.ga.get_stats()
        font  = self.renderer.font_s
        x, y  = 20, 20
        cam_mode = "BEST" if alive_count <= CAMERA_SWITCH_THRESHOLD else "CENTROID"

        bg = pygame.Surface((280, 130), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 200))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (194, 24, 91), (x, y, 280, 130), 1)

        lines = [
            f"Gen: {stats['generation']:4d}   Pop: {self.ga.pop_size}",
            f"Alive: {alive_count:3d} / {self.ga.pop_size}  [{cam_mode}]",
            f"Best:  {stats['best']:8.1f}",
            f"Avg:   {stats['avg']:8.1f}",
            f"Species: {stats.get('species','-'):3}  "
            f"Mut: {stats.get('mut_rate',0):.3f}",
            f"Frame: {self.ga_frame}",
        ]
        for i, line in enumerate(lines):
            t = font.render(line, True, (244, 143, 177))
            self.screen.blit(t, (x + 6, y + 6 + i * 18))

        # ゴール到達カウンタ
        goal_color = (50, 220, 150) if self.goal_reached_count > 0 else C_GRAY
        gt = font.render(
            f"GOAL REACHED: {self.goal_reached_count}/{self.fast_cfg_goal_count}",
            True, goal_color)
        self.screen.blit(gt, (x + 6, y + 130 + 4))

    # ----------------------------------------------------------------
    def _draw_fast_overlay(self):
        """高速モードの最小HUD（描画スキップ中に表示）。"""
        if self.ga is None:
            return
        stats = self.ga.get_stats()
        font  = self.renderer.font_m
        font_s = self.renderer.font_s

        bg = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        bg.fill((5, 5, 10, 245))
        self.screen.blit(bg, (0, 0))

        t = font.render("⚡ FAST LEARNING MODE", True, (255, 200, 50))
        self.screen.blit(t, (SCREEN_W // 2 - t.get_width() // 2, 60))

        lines = [
            f"Generation: {stats['generation']}",
            f"Best Fitness: {stats['best']:.1f}",
            f"Avg Fitness:  {stats['avg']:.1f}",
            f"Species: {stats.get('species', '-')}",
            f"Frame: {self.ga_frame}",
            f"Pop: {self.ga.pop_size}",
        ]
        for i, line in enumerate(lines):
            t = font_s.render(line, True, (200, 200, 220))
            self.screen.blit(t, (SCREEN_W // 2 - 130, 140 + i * 22))

        goal_color = (50, 220, 150) if self.goal_reached_count > 0 else (100, 100, 120)
        gt = font.render(
            f"GOAL REACHED: {self.goal_reached_count} / {self.fast_cfg_goal_count}",
            True, goal_color)
        self.screen.blit(gt, (SCREEN_W // 2 - gt.get_width() // 2, 290))

        # 前世代スナップショット（右側に小さく）
        if self.prev_best_genome is not None:
            gen = self.ga.generation - 1
            sl = font_s.render(f"PREV GEN {gen} BEST (snapshot)", True, (180, 140, 80))
            self.screen.blit(sl, (SCREEN_W - 510, SCREEN_H - 245))
            self.renderer.draw_activation_panel(
                self.prev_best_genome,
                x=SCREEN_W - 510, y=SCREEN_H - 230)

        hints = [
            "Tab: Switch to Monitor  Enter: Exit Fast Mode",
            "R: Reset (with confirm)  M: Menu  ESC: Quit",
        ]
        for i, h in enumerate(hints):
            ht = font_s.render(h, True, C_GRAY)
            self.screen.blit(ht, (SCREEN_W // 2 - ht.get_width() // 2,
                                  SCREEN_H - 50 + i * 20))

    # ----------------------------------------------------------------
    def _draw_fast_cfg(self):
        """高速モード設定画面。"""
        font_l = self.renderer.font_l
        font_m = self.renderer.font_m
        font_s = self.renderer.font_s

        t = font_l.render("⚡ FAST LEARN MODE — SETTINGS", True, (255, 200, 50))
        self.screen.blit(t, (SCREEN_W // 2 - t.get_width() // 2, 100))

        # ---- ゴール終了条件 ----
        focus_goal = self.fast_cfg_focus == "goal_count"
        gc_color = (255, 220, 80) if focus_goal else (160, 160, 180)
        gc_bg = (30, 25, 5) if focus_goal else (15, 15, 20)
        pygame.draw.rect(self.screen, gc_bg, (SCREEN_W // 2 - 300, 200, 600, 80),
                         border_radius=8)
        pygame.draw.rect(self.screen, gc_color, (SCREEN_W // 2 - 300, 200, 600, 80),
                         2, border_radius=8)

        tl = font_m.render("Goal Completion Count  (Tab to switch)", True, gc_color)
        self.screen.blit(tl, (SCREEN_W // 2 - tl.get_width() // 2, 210))

        # 1〜10 の選択肢
        for n in range(1, 11):
            bx = SCREEN_W // 2 - 270 + (n - 1) * 54
            by = 238
            selected = (n == self.fast_cfg_goal_count)
            bc = (255, 180, 0) if selected else (50, 50, 60)
            pygame.draw.rect(self.screen, bc, (bx, by, 44, 28), border_radius=4)
            pygame.draw.rect(self.screen, gc_color, (bx, by, 44, 28), 1, border_radius=4)
            nt = font_m.render(str(n), True, (0, 0, 0) if selected else (180, 180, 180))
            self.screen.blit(nt, (bx + 22 - nt.get_width() // 2, by + 5))

        # ---- エージェント数 ----
        focus_pop = self.fast_cfg_focus == "pop_size"
        pp_color = (100, 200, 255) if focus_pop else (160, 160, 180)
        pp_bg = (5, 20, 30) if focus_pop else (15, 15, 20)
        pygame.draw.rect(self.screen, pp_bg, (SCREEN_W // 2 - 300, 310, 600, 70),
                         border_radius=8)
        pygame.draw.rect(self.screen, pp_color, (SCREEN_W // 2 - 300, 310, 600, 70),
                         2, border_radius=8)

        pl = font_m.render(
            f"Population Size:  {self.fast_cfg_pop_size}  (←→ to change, Tab to switch)",
            True, pp_color)
        self.screen.blit(pl, (SCREEN_W // 2 - pl.get_width() // 2, 320))

        # バー表示
        bar_w = int(self.fast_cfg_pop_size / 2000 * 560)
        pygame.draw.rect(self.screen, (30, 50, 70), (SCREEN_W // 2 - 280, 348, 560, 14),
                         border_radius=4)
        pygame.draw.rect(self.screen, pp_color, (SCREEN_W // 2 - 280, 348, bar_w, 14),
                         border_radius=4)

        # ---- リセットボタン ----
        rx = SCREEN_W // 2 - 120
        ry = 420
        pygame.draw.rect(self.screen, (60, 10, 10), (rx, ry, 240, 50), border_radius=8)
        pygame.draw.rect(self.screen, (220, 50, 50), (rx, ry, 240, 50), 2, border_radius=8)
        rt = font_m.render("R: Reset All Progress", True, (220, 100, 100))
        self.screen.blit(rt, (rx + 120 - rt.get_width() // 2, ry + 14))

        # ---- 開始ボタン ----
        st = font_l.render("Enter / Space: START", True, (50, 220, 150))
        self.screen.blit(st, (SCREEN_W // 2 - st.get_width() // 2, 500))

        hint = font_s.render("ESC / M: Back to Menu", True, C_GRAY)
        self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 30))

        # リセット確認
        if self.confirm_reset:
            self._draw_confirm_reset()

    def _draw_confirm_reset(self):
        """リセット確認ダイアログ。"""
        font_m = self.renderer.font_m
        font_s = self.renderer.font_s

        # 半透明オーバーレイ
        ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 180))
        self.screen.blit(ov, (0, 0))

        bx, by, bw, bh = SCREEN_W // 2 - 220, SCREEN_H // 2 - 80, 440, 160
        pygame.draw.rect(self.screen, (20, 10, 10), (bx, by, bw, bh), border_radius=10)
        pygame.draw.rect(self.screen, (220, 50, 50), (bx, by, bw, bh), 2, border_radius=10)

        t1 = font_m.render("⚠ RESET ALL PROGRESS?", True, (220, 80, 80))
        self.screen.blit(t1, (SCREEN_W // 2 - t1.get_width() // 2, by + 20))

        t2 = font_s.render("All generations and fitness history will be lost.", True, C_GRAY)
        self.screen.blit(t2, (SCREEN_W // 2 - t2.get_width() // 2, by + 55))

        ty = font_m.render("[Y] Yes — Reset", True, (220, 80, 80))
        tn = font_m.render("[N] No — Cancel", True, (80, 180, 80))
        self.screen.blit(ty, (SCREEN_W // 2 - 180, by + 95))
        self.screen.blit(tn, (SCREEN_W // 2 + 10, by + 95))

    # ----------------------------------------------------------------
    def _draw_tracking_indicator(self, car, cam: pygame.Vector2):
        sx = int(car.pos.x - cam.x)
        sy = int(car.pos.y - cam.y)
        pts = [(sx, sy - 22), (sx - 8, sy - 36), (sx + 8, sy - 36)]
        pygame.draw.polygon(self.screen, (255, 200, 50), pts)
        t = self.renderer.font_s.render("★", True, (255, 200, 50))
        self.screen.blit(t, (sx - t.get_width() // 2, sy - 52))


# ================================================================
if __name__ == "__main__":
    game = Game()
    game.run()

"""
main.py — Blind Driving Survival エントリーポイント
モード選択 → プレイヤーモード / GAモード / 高速学習モード
"""
import sys
import os
# ヘッドレス環境（CI等）では SDL_VIDEODRIVER=dummy を設定して実行
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "dummy"

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
    MENU    = "menu"
    PLAYER  = "player"
    GA      = "ga"       # 通常GAモード（描画あり）
    GA_FAST = "ga_fast"  # 高速学習モード（描画スキップ）


# 生存数がこの値を下回ったらカメラを1位個体に切替
CAMERA_SWITCH_THRESHOLD = 10

# この世代でゴール到達した個体がこの数以上で学習完了
GOAL_COMPLETE_COUNT = 10


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

        # ゴール到達カウンタ（世代ごとにリセット）
        self.goal_reached_count: int = 0

        self.done_message: str = ""
        self.done_timer:   int = 0

    # ----------------------------------------------------------------
    def init_player_mode(self):
        self.field        = Field(terrain_seed=42, food_episode=0)
        self.player_car   = Car(*self.field.start_pos)
        self.player_agent = PlayerAgent(self.player_car, self.field)
        self.player_bn    = Bottleneck()
        self.done_message = ""

    def init_ga_mode(self):
        self.field      = Field(terrain_seed=42, food_episode=0)
        self.ga         = GeneticAlgorithm(pop_size=GA_POP_SIZE, seed=0)
        self._spawn_ga_agents()
        self.ga_frame   = 0
        self.ga_running = True
        self.goal_reached_count = 0
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
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                self._handle_event(event)

            self._update(dt)
            self._draw()
            pygame.display.flip()

    # ----------------------------------------------------------------
    def _handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            if self.state == GameState.MENU:
                if event.key == pygame.K_1:
                    self.init_player_mode()
                    self.state = GameState.PLAYER
                elif event.key == pygame.K_2:
                    self.init_ga_mode()
                    self.state = GameState.GA
                elif event.key == pygame.K_3:
                    self.init_ga_mode()
                    self.state = GameState.GA_FAST

            elif self.state == GameState.PLAYER:
                if event.key == pygame.K_m:
                    self.state = GameState.MENU
                if event.key == pygame.K_r:
                    self.init_player_mode()

            elif self.state == GameState.GA:
                if event.key == pygame.K_m:
                    self.state = GameState.MENU
                if event.key == pygame.K_r:
                    self.init_ga_mode()
                # Tab: 高速モードへ切替
                if event.key == pygame.K_TAB:
                    self.state = GameState.GA_FAST

            elif self.state == GameState.GA_FAST:
                # Tab: 監視モードへ切替
                if event.key == pygame.K_TAB:
                    self.state = GameState.GA
                # Enter: 高速モード終了 → 監視モードへ
                if event.key == pygame.K_RETURN:
                    self.state = GameState.GA
                if event.key == pygame.K_m:
                    self.state = GameState.MENU
                if event.key == pygame.K_r:
                    self.init_ga_mode()
                    self.state = GameState.GA_FAST

    # ----------------------------------------------------------------
    def _update(self, dt: float):
        if self.state == GameState.PLAYER:
            self._update_player(dt)
        elif self.state in (GameState.GA, GameState.GA_FAST):
            self._update_ga(dt)

    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    def _update_ga(self, dt: float):
        """全エージェントを同時に1フレーム進める。全員終了で世代進化。"""
        if not self.ga_running:
            return

        # 高速モードは複数フレームをまとめて処理
        steps = 10 if self.state == GameState.GA_FAST else 1

        for _ in range(steps):
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
                    # ゴール到達チェック
                    if agent.car.dist_to_goal < GOAL_RADIUS:
                        self.goal_reached_count += 1

            # 全員餓死で世代進化
            if all_done:
                # ③ 学習完了判定: 高速モード中にゴール到達10体以上
                if (self.state == GameState.GA_FAST
                        and self.goal_reached_count >= GOAL_COMPLETE_COUNT):
                    self._evolve_generation()
                    self.state = GameState.GA   # 監視モードへ自動遷移
                    return
                self._evolve_generation()
                # 高速モードならループを継続
                if self.state == GameState.GA:
                    break

    def _evolve_generation(self):
        """世代進化の共通処理。"""
        self.ga.evolve()
        self.field.reset_foods(food_episode=self.ga.generation)
        self._spawn_ga_agents()
        self.ga_frame = 0
        self.goal_reached_count = 0

    # ----------------------------------------------------------------
    def _get_camera_focus(self) -> pygame.Vector2:
        """
        ① カメラトラッキング切替:
        - 生存数 > CAMERA_SWITCH_THRESHOLD: 全体重心
        - 生存数 <= CAMERA_SWITCH_THRESHOLD: 累積報酬1位の個体を追跡
        """
        alive_agents = [ag for ag in self.ga_agents if ag.car.alive]
        if not alive_agents:
            return pygame.Vector2(self.field.start_pos)

        if len(alive_agents) > CAMERA_SWITCH_THRESHOLD:
            # 全体重心
            cx = sum(ag.car.pos.x for ag in alive_agents) / len(alive_agents)
            cy = sum(ag.car.pos.y for ag in alive_agents) / len(alive_agents)
            return pygame.Vector2(cx, cy)
        else:
            # 累積報酬1位を追跡
            best = max(alive_agents, key=lambda ag: ag.total_reward)
            return pygame.Vector2(best.car.pos)

    # ----------------------------------------------------------------
    def _draw_ga_overlay(self, alive_count: int, best_agent):
        """GAモードのオーバーレイHUDを描画する。"""
        stats = self.ga.get_stats()
        font  = self.renderer.font_s
        x, y  = 20, 20

        # カメラモード表示
        cam_mode = "BEST" if alive_count <= CAMERA_SWITCH_THRESHOLD else "CENTROID"

        bg = pygame.Surface((280, 120), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 200))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (194, 24, 91), (x, y, 280, 120), 1)

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

        # ③ ゴール到達カウンタ
        goal_color = (50, 220, 150) if self.goal_reached_count > 0 else C_GRAY
        gt = font.render(
            f"GOAL REACHED: {self.goal_reached_count}/{GOAL_COMPLETE_COUNT}",
            True, goal_color)
        self.screen.blit(gt, (x + 6, y + 120 + 4))

    def _draw_fast_overlay(self):
        """高速学習モードのオーバーレイHUDを描画する。"""
        stats = self.ga.get_stats()
        font  = self.renderer.font_m
        font_s = self.renderer.font_s

        # 背景
        bg = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        bg.fill((5, 5, 10, 240))
        self.screen.blit(bg, (0, 0))

        # タイトル
        t = font.render("⚡ FAST LEARNING MODE", True, (255, 200, 50))
        self.screen.blit(t, (SCREEN_W // 2 - t.get_width() // 2, 80))

        # 統計
        lines = [
            f"Generation: {stats['generation']}",
            f"Best Fitness: {stats['best']:.1f}",
            f"Avg Fitness:  {stats['avg']:.1f}",
            f"Species: {stats.get('species', '-')}",
            f"Frame: {self.ga_frame}",
        ]
        for i, line in enumerate(lines):
            t = font_s.render(line, True, (200, 200, 220))
            self.screen.blit(t, (SCREEN_W // 2 - 120, 160 + i * 24))

        # ゴール到達カウンタ（大きく）
        goal_color = (50, 220, 150) if self.goal_reached_count > 0 else (100, 100, 120)
        gt = font.render(
            f"GOAL REACHED: {self.goal_reached_count} / {GOAL_COMPLETE_COUNT}",
            True, goal_color)
        self.screen.blit(gt, (SCREEN_W // 2 - gt.get_width() // 2, 300))

        if self.goal_reached_count >= GOAL_COMPLETE_COUNT:
            ct = font.render("✓ COMPLETE — switching to monitor...", True, (50, 255, 150))
            self.screen.blit(ct, (SCREEN_W // 2 - ct.get_width() // 2, 360))

        # キーヒント
        hints = [
            "Tab: Switch to Monitor Mode",
            "Enter: Exit Fast Mode → Monitor",
            "R: Reset   M: Menu   ESC: Quit",
        ]
        for i, h in enumerate(hints):
            ht = font_s.render(h, True, C_GRAY)
            self.screen.blit(ht, (SCREEN_W // 2 - ht.get_width() // 2,
                                  SCREEN_H - 80 + i * 22))

    # ----------------------------------------------------------------
    def _draw(self):
        self.screen.fill(C_BG)

        if self.state == GameState.MENU:
            self.renderer.draw_mode_select()

        elif self.state == GameState.PLAYER:
            cam = self.renderer.calc_camera(self.player_car.pos)
            self.renderer.draw_field(self.field, cam)
            if self.renderer._minimap_surf is None and self.renderer._static_buf is not None:
                from pygame import transform
                self.renderer._minimap_surf = transform.scale(
                    self.renderer._static_buf, (160, 160))
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
                x=SCREEN_W - 430,
                y=SCREEN_H - 210)
            hint = self.renderer.font_s.render(
                "M: Menu  R: Reset  Tab: Fast Mode  ESC: Quit", True, C_GRAY)
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 20))

        elif self.state == GameState.GA_FAST:
            # ② 高速モード: 描画は最小限のオーバーレイのみ
            self._draw_fast_overlay()

        elif self.state == GameState.GA:
            # ① カメラトラッキング切替
            focus = self._get_camera_focus()
            cam   = self.renderer.calc_camera(focus)

            self.renderer.draw_field(self.field, cam)
            if self.renderer._minimap_surf is None and self.renderer._static_buf is not None:
                from pygame import transform
                self.renderer._minimap_surf = transform.scale(
                    self.renderer._static_buf, (160, 160))

            # 全エージェントを一括描画
            alive_count = 0
            best_agent  = None
            alive_agents = [ag for ag in self.ga_agents if ag.car.alive]
            for ag in alive_agents:
                alive_count += 1
                if best_agent is None or ag.total_reward > best_agent.total_reward:
                    best_agent = ag
                ag.car.draw(self.screen, cam, color=(40, 100, 160))

            # 最高報酬エージェントを金色で強調
            if best_agent:
                self.renderer.draw_vision_cone(best_agent.car, cam)
                best_agent.car.draw(self.screen, cam, color=(255, 200, 50))

            # 生存数が少ない場合、カメラ追従中の個体に矢印を表示
            if alive_count <= CAMERA_SWITCH_THRESHOLD and best_agent:
                self._draw_tracking_indicator(best_agent.car, cam)

            self.renderer.draw_minimap(
                self.field, focus, self.field.goal_pos)

            self._draw_ga_overlay(alive_count, best_agent)
            self.renderer.draw_fitness_graph(
                self.ga, 20, 150, w=280, h=100)
            if best_agent:
                self.renderer.draw_activation_panel(
                    best_agent.genome,
                    x=SCREEN_W - 510,
                    y=SCREEN_H - 230)
            hint = self.renderer.font_s.render(
                "Tab: Fast Mode  M: Menu  R: Reset  ESC: Quit", True, C_GRAY)
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 20))

    def _draw_tracking_indicator(self, car, cam: pygame.Vector2):
        """カメラ追従中の個体に小さな矢印マーカーを表示する。"""
        sx = int(car.pos.x - cam.x)
        sy = int(car.pos.y - cam.y)
        # 上向き三角形
        pts = [(sx, sy - 22), (sx - 8, sy - 36), (sx + 8, sy - 36)]
        pygame.draw.polygon(self.screen, (255, 200, 50), pts)
        t = self.renderer.font_s.render("★", True, (255, 200, 50))
        self.screen.blit(t, (sx - t.get_width() // 2, sy - 52))


# ================================================================
if __name__ == "__main__":
    game = Game()
    game.run()

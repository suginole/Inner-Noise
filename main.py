"""
main.py — Blind Driving Survival エントリーポイント
モード選択 → プレイヤーモード / GAモード
"""
import sys
import os
# ヘッドレス環境（CI等）では SDL_VIDEODRIVER=dummy を設定して実行
# 通常の実行では不要（コメントアウト）
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
    MENU   = "menu"
    PLAYER = "player"
    GA     = "ga"


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

        # GAモード
        self.ga:           GeneticAlgorithm | None = None
        self.ga_agents:    list[GAAgent] = []
        self.ga_frame:     int = 0
        self.ga_running:   bool = False

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
        # 地形固定・初回エピソードの餌配置
        self.field      = Field(terrain_seed=42, food_episode=0)
        self.ga         = GeneticAlgorithm(pop_size=GA_POP_SIZE, seed=0)
        self._spawn_ga_agents()
        self.ga_frame   = 0
        self.ga_running = True
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

            elif self.state in (GameState.PLAYER, GameState.GA):
                if event.key == pygame.K_m:
                    self.state = GameState.MENU
                if event.key == pygame.K_r:
                    if self.state == GameState.PLAYER:
                        self.init_player_mode()
                    else:
                        self.init_ga_mode()
                # GAモード: SPACEで現在のエージェントをスキップ
                if self.state == GameState.GA and event.key == pygame.K_SPACE:
                    self._ga_next_agent()

    # ----------------------------------------------------------------
    def _update(self, dt: float):
        if self.state == GameState.PLAYER:
            self._update_player(dt)
        elif self.state == GameState.GA:
            self._update_ga(dt)

    # ----------------------------------------------------------------
    def _update_player(self, dt: float):
        if not self.player_car.alive:
            if self.done_timer > 0:
                self.done_timer -= 1
            return

        # プレイヤーの行動
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

        self.ga_frame += 1
        all_done = True

        for agent in self.ga_agents:
            if not agent.car.alive:
                continue   # 既に終了済み
            all_done = False

            result = agent.step()
            if result["done"]:
                agent.genome.fitness = agent.total_reward
                agent.car.alive = False

        # 全員餐死で世代進化（時間制限なし）
        if all_done:
            self.ga.evolve()
            self.field.reset_foods(food_episode=self.ga.generation)
            self._spawn_ga_agents()
            self.ga_frame = 0

    # ----------------------------------------------------------------
    def _draw_ga_overlay(self, alive_count: int, best_agent):
        """GAモードのオーバーレイHUDを描画する。"""
        stats = self.ga.get_stats()
        font  = self.renderer.font_s
        font_m = self.renderer.font_m
        x, y  = 20, 20
        bg = pygame.Surface((260, 100), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 200))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (194, 24, 91), (x, y, 260, 100), 1)
        lines = [
            f"Gen: {stats['generation']:4d}   Pop: {self.ga.pop_size}",
            f"Alive: {alive_count:3d} / {self.ga.pop_size}",
            f"Best:  {stats['best']:8.1f}",
            f"Avg:   {stats['avg']:8.1f}",
            f"Species: {stats.get('species','-'):3}  "
            f"Mut: {stats.get('mut_rate',0):.3f}",
        ]
        for i, line in enumerate(lines):
            t = font.render(line, True, (244, 143, 177))
            self.screen.blit(t, (x + 6, y + 6 + i * 18))

        # フレームカウンタ
        bar_y = y + 100 + 4
        t = font.render(f"Frame: {self.ga_frame}  (no time limit)", True, C_GRAY)
        self.screen.blit(t, (x, bar_y))

    # ----------------------------------------------------------------
    def _draw(self):
        self.screen.fill(C_BG)

        if self.state == GameState.MENU:
            self.renderer.draw_mode_select()

        elif self.state == GameState.PLAYER:
            cam = self.renderer.calc_camera(self.player_car.pos)
            self.renderer.draw_field(self.field, cam)   # 静的バッファを初回生成
            # ミニマップは静的バッファ生成後に初回キャッシュ
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
            # プレイヤーモード用のアクティベーションパネル（観測ベクトルのみ）
            self.renderer.draw_player_obs_panel(
                self.player_car.get_observation(self.field),
                x=SCREEN_W - 430,
                y=SCREEN_H - 210)
            hint = self.renderer.font_s.render("M: Menu  R: Reset  ESC: Quit", True, C_GRAY)
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 20))

        elif self.state == GameState.GA:
            # 生存中のエージェントの重心にカメラを合わせる
            alive_agents = [ag for ag in self.ga_agents if ag.car.alive]
            if alive_agents:
                cx = sum(ag.car.pos.x for ag in alive_agents) / len(alive_agents)
                cy = sum(ag.car.pos.y for ag in alive_agents) / len(alive_agents)
                focus = pygame.Vector2(cx, cy)
            else:
                focus = pygame.Vector2(self.field.start_pos)
            cam = self.renderer.calc_camera(focus)

            self.renderer.draw_field(self.field, cam)
            if self.renderer._minimap_surf is None and self.renderer._static_buf is not None:
                from pygame import transform
                self.renderer._minimap_surf = transform.scale(
                    self.renderer._static_buf, (160, 160))

            # 全エージェントを一括描画
            alive_count = 0
            best_agent  = None
            for ag in self.ga_agents:
                if not ag.car.alive:
                    continue
                alive_count += 1
                # 最高報酬のエージェントを強調表示用に記録
                if best_agent is None or ag.total_reward > best_agent.total_reward:
                    best_agent = ag
                ag.car.draw(self.screen, cam, color=(40, 100, 160))

            # 最高報酬エージェントを金色で強調
            if best_agent:
                self.renderer.draw_vision_cone(best_agent.car, cam)
                best_agent.car.draw(self.screen, cam, color=(255, 200, 50))

            self.renderer.draw_minimap(
                self.field, focus, self.field.goal_pos)

            # HUD: 生存数・世代・適応度
            self._draw_ga_overlay(alive_count, best_agent)
            self.renderer.draw_fitness_graph(
                self.ga, 20, 130, w=260, h=100)
            if best_agent:
                self.renderer.draw_activation_panel(
                    best_agent.genome,
                    x=SCREEN_W - 510,
                    y=SCREEN_H - 230)
            hint = self.renderer.font_s.render(
                "M: Menu  R: Reset  ESC: Quit", True, C_GRAY)
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 20))


# ================================================================
if __name__ == "__main__":
    game = Game()
    game.run()

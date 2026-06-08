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
        self.ga_idx:       int = 0
        self.ga_frame:     int = 0
        self.ga_running:   bool = False

        self.done_message: str = ""
        self.done_timer:   int = 0

    # ----------------------------------------------------------------
    def init_player_mode(self):
        self.field        = Field(seed=42)
        self.player_car   = Car(*self.field.start_pos)
        self.player_agent = PlayerAgent(self.player_car, self.field)
        self.player_bn    = Bottleneck()
        self.done_message = ""

    def init_ga_mode(self):
        self.field      = Field(seed=42)
        self.ga         = GeneticAlgorithm(pop_size=GA_POP_SIZE, seed=0)
        self._spawn_ga_agents()
        self.ga_idx     = 0
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

        # ボトルネック経由（観察用）
        obs = self.player_car.get_observation(self.field)
        self.player_bn.push(obs)
        self.player_bn.tick(dt)

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
        if not self.ga_running:
            return

        agent = self.ga_agents[self.ga_idx]
        car   = agent.car

        # エピソード終了判定
        if not car.alive or self.ga_frame >= GA_EPISODE_FRAMES:
            # 適応度を記録
            agent.genome.fitness = agent.total_reward
            self._ga_next_agent()
            return

        result = agent.step()
        self.ga_frame += 1

        if result["done"]:
            agent.genome.fitness = agent.total_reward
            self._ga_next_agent()

    def _ga_next_agent(self):
        """次のエージェントへ進む。全員終わったら進化。"""
        self.ga_idx   += 1
        self.ga_frame  = 0

        if self.ga_idx >= len(self.ga_agents):
            # 世代進化
            self.ga.evolve()
            # フィールドをリセット（新世代用）
            self.field = Field(seed=42 + self.ga.generation)
            self._spawn_ga_agents()
            self.ga_idx = 0

    # ----------------------------------------------------------------
    def _draw(self):
        self.screen.fill(C_BG)

        if self.state == GameState.MENU:
            self.renderer.draw_mode_select()

        elif self.state == GameState.PLAYER:
            cam = self.renderer.calc_camera(self.player_car.pos)
            self.renderer.draw_field(self.field, cam)
            self.renderer.draw_vision_cone(self.player_car, cam)
            self.player_car.draw(self.screen, cam, color=C_CAR)
            self.renderer.draw_minimap(
                self.field, self.player_car.pos, self.field.goal_pos)
            self.renderer.draw_hud_player(self.player_car, self.player_bn)
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
            if self.ga_idx < len(self.ga_agents):
                agent = self.ga_agents[self.ga_idx]
                cam   = self.renderer.calc_camera(agent.car.pos)
                self.renderer.draw_field(self.field, cam)
                # 全エージェントの車を薄く描画
                for i, ag in enumerate(self.ga_agents):
                    if i == self.ga_idx:
                        continue
                    ag.car.draw(self.screen, cam,
                                color=(40, 80, 120))
                # 現在のエージェントの視野コーンと車を強調
                self.renderer.draw_vision_cone(agent.car, cam)
                agent.car.draw(self.screen, cam, color=C_CAR_GA)
                self.renderer.draw_minimap(
                    self.field, agent.car.pos, self.field.goal_pos)
                self.renderer.draw_hud_ga(
                    agent.car, self.ga,
                    bottleneck=agent.bottleneck,
                    agent_idx=self.ga_idx,
                    total=len(self.ga_agents))
                self.renderer.draw_fitness_graph(
                    self.ga, 20, 110, w=260, h=80)
                # アクティベーションパネル（画面右下）
                self.renderer.draw_activation_panel(
                    agent.genome,
                    x=SCREEN_W - 430,
                    y=SCREEN_H - 210)
                hint = self.renderer.font_s.render("M: Menu  R: Reset  SPACE: Skip  ESC: Quit", True, C_GRAY)
                self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 20))


# ================================================================
if __name__ == "__main__":
    game = Game()
    game.run()

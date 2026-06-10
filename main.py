"""
main.py — Inner Noise: Sage & Brute エントリーポイント
"""
import sys
import copy
import pygame
from config import *
from game.field     import Field
from game.ga_agent  import GAGenome, GeneticAlgorithm, SageBruteAgent
from game.renderer  import Renderer
from game           import save_manager


# ================================================================
class GameState:
    MENU        = "menu"
    GA          = "ga"
    GA_FAST     = "ga_fast"
    LOAD        = "load"
    LOAD_RESUME = "load_resume"


FAST_STEPS = 200
CAMERA_SWITCH_THRESHOLD = 10


class Game:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen   = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption(TITLE)
        self.clock    = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.state    = GameState.MENU

        self.field:   Field | None = None
        self.ga:      GeneticAlgorithm | None = None
        self.agents:  list[SageBruteAgent] = []
        self.ga_frame: int = 0
        self.ga_running: bool = False
        self.goal_count: int = 0

        self.prev_best_genome: GAGenome | None = None
        self.tracked_agent: SageBruteAgent | None = None

        self.fast_cfg_goal_count: int = 5
        self.fast_cfg_pop_size:   int = GA_POP_SIZE

        self.confirm_reset: bool = False
        self.confirm_menu:  bool = False

        self.load_models:   list[dict] = []
        self.load_sel_idx:  int = 0
        self.load_error:    str = ""
        self.save_toast_msg:   str = ""
        self.save_toast_timer: int = 0

        self.loaded_from_gen: int | None = None
        self.load_resume_meta: dict | None = None

        save_manager.init_db()

    # ----------------------------------------------------------------
    def init_ga_mode(self, pop_size: int | None = None):
        pop = pop_size or GA_POP_SIZE
        self.field      = Field(terrain_seed=TERRAIN_SEED, food_episode=0)
        self.ga         = GeneticAlgorithm(pop_size=pop, seed=0)
        self._spawn_agents()
        self.ga_frame   = 0
        self.ga_running = True
        self.goal_count = 0
        self.prev_best_genome = None
        self.tracked_agent    = None

    def _spawn_agents(self):
        self.agents = []
        self.tracked_agent = None
        for genome in self.ga.population:
            agent_field = self.field.clone_foods()
            agent = SageBruteAgent(genome, agent_field, self.field.start_pos)
            self.agents.append(agent)

    TRACK_UPDATE_INTERVAL = 1000

    def _switch_tracked_agent(self, alive):
        if not alive:
            if self.tracked_agent and hasattr(self.tracked_agent, 'bn'):
                self.tracked_agent.bn.disable_audio()
            self.tracked_agent = None
            return
        new_best = max(alive, key=lambda a: a.total_reward)
        if self.tracked_agent is not new_best:
            if self.tracked_agent and hasattr(self.tracked_agent, 'bn'):
                self.tracked_agent.bn.disable_audio()
            self.tracked_agent = new_best

    # ----------------------------------------------------------------
    def run(self):
        while True:
            if self.state == GameState.GA_FAST:
                self._run_fast_frame()
                continue
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                self._handle_event(event)
            self._update(dt)
            self._draw()
            pygame.display.flip()

    def _run_fast_frame(self):
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key in (pygame.K_TAB, pygame.K_RETURN):
                    self.state = GameState.GA; return
                if event.key == pygame.K_m:
                    self.state = GameState.MENU; return
        for _ in range(FAST_STEPS):
            if self._step_ga_once():
                break
        # 高速モード中の追従エージェントを更新
        alive = [a for a in self.agents if a.alive]
        if self.tracked_agent is None or not self.tracked_agent.alive:
            self._switch_tracked_agent(alive)
        elif self.ga_frame % self.TRACK_UPDATE_INTERVAL == 0 and alive:
            self._switch_tracked_agent(alive)

        self.screen.fill(C_BG)
        self._draw_fast_overlay()
        self._draw_monitor_panels()   # 3パネルモニターを追加
        pygame.display.flip()

    def _step_ga_once(self) -> bool:
        if not self.ga_running:
            return False
        self.ga_frame += 1
        all_done = True
        for agent in self.agents:
            if not agent.alive:
                continue
            all_done = False
            result = agent.step()
            if result['done']:
                agent.genome.fitness = agent.total_reward
                agent.alive = False
                if (pygame.Vector2(*self.field.goal_pos) - agent.pos).length() < GOAL_RADIUS:
                    self.goal_count += 1
        if all_done:
            if self.goal_count >= self.fast_cfg_goal_count:
                self._evolve_generation(auto_save=True)
                self.state = GameState.GA
                return True
            self._evolve_generation()
            return True
        return False

    def _evolve_generation(self, auto_save: bool = False):
        if auto_save:
            self._do_save(auto=True)
        self.ga.evolve()
        self.field.reset_foods(food_episode=self.ga.generation)
        self._spawn_agents()
        self.ga_frame   = 0
        self.goal_count = 0

    # ----------------------------------------------------------------
    def _handle_event(self, event):
        if event.type != pygame.KEYDOWN:
            return
        key = event.key

        if key == pygame.K_ESCAPE and not self.confirm_reset:
            pygame.quit(); sys.exit()

        if self.confirm_menu:
            if key == pygame.K_y:
                self.confirm_menu = False; self.state = GameState.MENU
            elif key in (pygame.K_n, pygame.K_ESCAPE):
                self.confirm_menu = False
            return

        if self.confirm_reset:
            if key == pygame.K_y:
                self.confirm_reset = False
                self.init_ga_mode(pop_size=self.fast_cfg_pop_size)
                self.state = GameState.GA_FAST
            elif key in (pygame.K_n, pygame.K_ESCAPE):
                self.confirm_reset = False
            return

        if self.state == GameState.MENU:
            if key == pygame.K_2:
                self.init_ga_mode(); self.state = GameState.GA
            elif key == pygame.K_3:
                self.init_ga_mode(pop_size=self.fast_cfg_pop_size)
                self.state = GameState.GA_FAST
            elif key == pygame.K_l:
                self._open_load_screen()

        elif self.state == GameState.LOAD:
            if key in (pygame.K_ESCAPE, pygame.K_m):
                self.state = GameState.MENU
            elif key == pygame.K_UP:
                self.load_sel_idx = max(0, self.load_sel_idx - 1)
            elif key == pygame.K_DOWN:
                self.load_sel_idx = min(len(self.load_models)-1, self.load_sel_idx+1)
            elif key == pygame.K_RETURN:
                self._do_load()
            elif key == pygame.K_DELETE:
                if self.load_models:
                    save_manager.delete_model(self.load_models[self.load_sel_idx]['id'])
                    self.load_models = save_manager.list_models()
                    self.load_sel_idx = min(self.load_sel_idx, max(0, len(self.load_models)-1))

        elif self.state == GameState.LOAD_RESUME:
            if key == pygame.K_ESCAPE:
                self.state = GameState.LOAD
            elif key == pygame.K_2:
                self.save_toast_msg   = f"ロード完了  Gen {self.ga.generation}"
                self.save_toast_timer = FPS * 3
                self.state = GameState.GA
            elif key == pygame.K_3:
                self.save_toast_msg   = f"ロード完了  Gen {self.ga.generation}"
                self.save_toast_timer = FPS * 3
                self.state = GameState.GA_FAST

        elif self.state == GameState.GA:
            if key == pygame.K_m:
                if self.ga and self.ga.generation > 0:
                    self.confirm_menu = True
                else:
                    self.state = GameState.MENU
            elif key == pygame.K_s:
                self._do_save()
            elif key == pygame.K_TAB:
                if self.ga is None:
                    self.init_ga_mode()
                self.state = GameState.GA_FAST

    # ----------------------------------------------------------------
    def _update(self, dt: float):
        if self.state == GameState.GA:
            self._update_ga_monitor()

    def _update_ga_monitor(self):
        if not self.ga_running:
            return
        self._step_ga_once()
        alive = [a for a in self.agents if a.alive]
        if self.tracked_agent and not self.tracked_agent.alive:
            self._switch_tracked_agent(alive)
        elif self.tracked_agent is None and alive:
            self._switch_tracked_agent(alive)
        elif self.ga_frame % self.TRACK_UPDATE_INTERVAL == 0 and alive:
            self._switch_tracked_agent(alive)

    # ----------------------------------------------------------------
    def _draw(self):
        self.screen.fill(C_BG)

        if self.save_toast_timer > 0:
            self.save_toast_timer -= 1
            self.renderer.draw_save_toast(self.save_toast_msg,
                                           min(220, self.save_toast_timer * 10))

        if self.state == GameState.LOAD:
            self.renderer.draw_load_screen(self.load_models, self.load_sel_idx, self.load_error)
            return

        if self.state == GameState.LOAD_RESUME:
            self._draw_load_resume()
            return

        if self.state == GameState.MENU:
            self.renderer.draw_mode_select()

        elif self.state == GameState.GA:
            self._draw_ga_monitor()

    def _draw_ga_monitor(self):
        tracked = self.tracked_agent
        focus = pygame.Vector2(tracked.pos) if (tracked and tracked.alive) else pygame.Vector2(self.field.start_pos)
        cam   = self.renderer.calc_camera(focus)

        self.renderer.draw_field(self.field, cam)

        alive_count = 0
        for ag in self.agents:
            if not ag.alive:
                continue
            alive_count += 1
            color = (255, 200, 50) if ag is tracked else (40, 100, 160)
            self.renderer.draw_agent(ag, cam, color=color, is_best=(ag is tracked))

        self.renderer.draw_minimap(self.field, focus, self.field.goal_pos)

        stats = self.ga.get_stats()
        stats['pop_size'] = self.ga.pop_size
        self.renderer.draw_ga_overlay(stats, alive_count)
        self.renderer.draw_fitness_graph(self.ga, 20, 150, w=280, h=100)

        # 3パネルモニター（監視モード・高速モード共通メソッド）
        self._draw_monitor_panels()

        hint = self.renderer.font_s.render(
            "S: Save  Tab: Fast Mode  M: Menu  ESC: Quit", True, C_GRAY)
        self.screen.blit(hint, (SCREEN_W//2 - hint.get_width()//2, SCREEN_H - 20))

    def _draw_monitor_panels(self):
        """高速モード・監視モード共通の3パネルモニター描画。"""
        tracked = self.tracked_agent
        if tracked is None:
            return
        genome = tracked.genome
        bn     = tracked.bn
        total_w = 280 * 3 + 8 * 2
        mx = SCREEN_W // 2 - total_w // 2
        my = SCREEN_H - 230
        self.renderer.draw_rnn_monitor_panels(genome, bn, x=mx, y=my, panel_w=280, panel_h=220)
        # 捨取履歴パネル
        iw = 300
        ix = SCREEN_W // 2 - iw // 2
        self.renderer.draw_intake_panel(tracked, ix, my - 90, w=iw, h=80)

    def _draw_fast_overlay(self):
        if not self.ga:
            return
        stats = self.ga.get_stats()
        font  = self.renderer.font_m
        font_s = self.renderer.font_s
        bg = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        bg.fill((5, 5, 10, 245))
        self.screen.blit(bg, (0, 0))
        t = font.render("FAST LEARNING MODE", True, (255, 200, 50))
        self.screen.blit(t, (SCREEN_W//2 - t.get_width()//2, 60))
        for i, line in enumerate([
            f"Generation: {stats['generation']}",
            f"Best: {stats['best']:.1f}",
            f"Avg:  {stats['avg']:.1f}",
            f"Species: {stats.get('species', '-')}",
            f"Frame: {self.ga_frame}",
        ]):
            lt = font_s.render(line, True, (200, 200, 220))
            self.screen.blit(lt, (SCREEN_W//2 - 130, 140 + i * 22))
        hints = ["Tab/Enter: Monitor Mode  M: Menu  ESC: Quit"]
        for i, h in enumerate(hints):
            ht = font_s.render(h, True, C_GRAY)
            self.screen.blit(ht, (SCREEN_W//2 - ht.get_width()//2, SCREEN_H - 30 + i*20))

    def _draw_load_resume(self):
        font_l = self.renderer.font_l
        font_m = self.renderer.font_m
        font_s = self.renderer.font_s
        ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((5, 8, 18, 240))
        self.screen.blit(ov, (0, 0))
        t = font_l.render("モデルをロードしました", True, (100, 220, 255))
        self.screen.blit(t, (SCREEN_W//2 - t.get_width()//2, 120))
        m = self.load_resume_meta
        if m:
            gen_val = self.ga.generation if self.ga else m.get('generation', '?')
            best_val = m.get('best_fitness', 0.0)
            saved_at = m.get('saved_at', '')
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(saved_at)
                saved_at_str = dt.astimezone().strftime("%Y-%m-%d %H:%M")
            except Exception:
                saved_at_str = saved_at[:16]
            for i, line in enumerate([f"Gen: {gen_val}    Best: {best_val:.1f}",
                                       f"保存日時: {saved_at_str}"]):
                lt = font_m.render(line, True, (200, 200, 220))
                self.screen.blit(lt, (SCREEN_W//2 - lt.get_width()//2, 200 + i*34))
        bx, by, bw, bh = SCREEN_W//2 - 260, 300, 520, 180
        pygame.draw.rect(self.screen, (10, 15, 25), (bx, by, bw, bh), border_radius=10)
        pygame.draw.rect(self.screen, (100, 180, 255), (bx, by, bw, bh), 2, border_radius=10)
        for i, (label, color) in enumerate([("[2]  監視モードで再開", (50, 220, 150)),
                                             ("[3]  高速学習モードで再開", (255, 200, 50))]):
            lt = font_m.render(label, True, color)
            self.screen.blit(lt, (SCREEN_W//2 - lt.get_width()//2, by + 24 + i*60))
        hint = font_s.render("[ESC] キャンセル (ロード画面に戻る)", True, C_GRAY)
        self.screen.blit(hint, (SCREEN_W//2 - hint.get_width()//2, SCREEN_H - 30))

    # ----------------------------------------------------------------
    def _do_save(self, auto: bool = False):
        if not self.ga:
            return
        try:
            rid = save_manager.save_model(self.ga, terrain_seed=TERRAIN_SEED,
                                           goal_count=self.goal_count)
            prefix = "自動セーブ" if auto else "セーブ"
            self.save_toast_msg   = f"{prefix}完了  Gen {self.ga.generation}  ID={rid}"
            self.save_toast_timer = FPS * 3
        except Exception as e:
            self.save_toast_msg   = f"セーブ失敗: {e}"
            self.save_toast_timer = FPS * 3

    def _open_load_screen(self):
        self.load_models  = save_manager.list_models()
        self.load_sel_idx = 0
        self.load_error   = ""
        self.state        = GameState.LOAD

    def _do_load(self):
        if not self.load_models:
            return
        m = self.load_models[self.load_sel_idx]
        if not m['compatible']:
            self.load_error = "互換性エラー: NN構造が一致しません"
            return
        try:
            pop_size = m['pop_size'] if isinstance(m['pop_size'], int) else GA_POP_SIZE
            self.init_ga_mode(pop_size=pop_size)
            save_manager.load_model(m['id'], self.ga)
            self._spawn_agents()
            self.loaded_from_gen  = self.ga.generation
            self.load_resume_meta = m
            self.load_error = ""
            self.state = GameState.LOAD_RESUME
        except Exception as e:
            self.load_error = f"ロード失敗: {e}"


# ================================================================
if __name__ == "__main__":
    game = Game()
    game.run()

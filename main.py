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
from game.ga_agent import GAGenome, GeneticAlgorithm, SageBruteAgent
from game.bottleneck import Bottleneck
from game.renderer import Renderer
from game import save_manager


# ================================================================
class _AudioBN:
    """音声専用ダミーボトルネック。サウンドモードと音声共有用。"""
    def __init__(self):
        self.audio_enabled = False
        self.converter     = None
        self._last_pulse   = [0, 0]
        self._history      = []
        self._frame        = 0
        self.direction     = 'S→B'

    def enable_audio(self):
        if self.converter is None:
            from game.phoneme import PhonemeConverter
            self.converter = PhonemeConverter()
        self.audio_enabled = True

    def disable_audio(self):
        self.audio_enabled = False

    def toggle_audio(self) -> bool:
        if self.audio_enabled:
            self.disable_audio()
        else:
            self.enable_audio()
        return self.audio_enabled

    def tick(self, dt=0):
        pass

    def get_mode(self):             return 'listen'
    def get_turn_progress(self):    return 0.0
    def get_display_progress(self): return 0.0
    def get_current_pulse(self):    return [0, 0]
    def get_pulse_history(self):    return []
    def get_display_history(self):  return []
    def get_display_phoneme(self):  return ''
    def get_last_phoneme(self):     return ''


# ================================================================
class GameState:
    MENU        = "menu"
    PLAYER      = "player"
    GA          = "ga"          # 通常GAモード（描画あり）
    GA_FAST_CFG = "ga_fast_cfg" # 高速モード設定画面
    GA_FAST     = "ga_fast"     # 高速学習モード（描画完全スキップ）
    LOAD        = "load"        # モデルロード画面
    LOAD_RESUME = "load_resume" # ロード後の再開モード選択画面
    BACKROOM    = "backroom"    # 音声入出力確認・デバッグモード


# カメラ切替閖値
CAMERA_SWITCH_THRESHOLD = 10

# 高速モード: 1ループあたりの計算ステップ数
# 増やすほど高速になるが、イベント応答が遅れる
FAST_STEPS = 200


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
        self.ga_agents:    list = []
        self.ga_frame:     int = 0
        self.ga_running:   bool = False
        self.goal_reached_count: int = 0

        # 前世代の最高適応度ゲノムのスナップショット
        self.prev_best_genome: GAGenome | None = None

        # 追従中エージェント（更新・音声・モニター描画を共通の参照に統一）
        self.tracked_agent = None

        # 音声入出力用ボトルネック（GAモード共有）
        self.audio_bn: 'AudioBN' = _AudioBN()  # 音声専用ダミー

        # 高速モード設定
        self.fast_cfg_goal_count: int = 10   # 終了条件（1〜10）
        self.fast_cfg_pop_size:   int = GA_POP_SIZE
        self.fast_cfg_focus:      str = "goal_count"  # 現在フォーカス中の設定項目

        # リセット確認ダイアログ
        self.confirm_reset: bool = False
        # メニュー返報確認ダイアログ
        self.confirm_menu:  bool = False

        # セーブ・ロード
        self.load_models:   list[dict] = []
        self.load_sel_idx:  int = 0
        self.load_error:    str = ""
        self.save_toast_msg:   str = ""   # トースト表示メッセージ
        self.save_toast_timer: int = 0    # 表示フレーム数

        # ロード元情報（LOAD_RESUME画面・HUD表示用）
        self.loaded_from_gen: int | None = None  # ロード元の世代数（None=新規）
        self.load_resume_meta: dict | None = None  # ロード後の表示用メタデータ

        self.done_message: str = ""
        self.done_timer:   int = 0

        # バックルームモード状態
        self.br_bits4:      int = 0          # 手動入力された4bits値
        self.br_waveform    = None           # 直近の合成波形
        self.br_mic_data:   dict | None = None  # マイク解析結果
        self.br_bn:         'AudioBN' = _AudioBN()  # バックルーム専用BN
        self.br_mic_thread_running: bool = False

        # DB初期化
        save_manager.init_db()

    # ----------------------------------------------------------------
    def init_player_mode(self):
        self.field        = Field(terrain_seed=42, food_episode=0)
        self.player_car   = Car(*self.field.start_pos)
        self.player_agent = PlayerAgent(self.player_car, self.field)
        self.player_bn    = Bottleneck()
        self.done_message = ""

    def init_ga_mode(self, pop_size: int | None = None):
        # ローディング画面を表示して画面を更新
        self.screen.fill(C_BG)
        t = self.renderer.font_m.render("Loading...", True, C_WHITE)
        self.screen.blit(t, (SCREEN_W//2 - t.get_width()//2, SCREEN_H//2))
        pygame.display.flip()
        pygame.event.pump()

        pop = pop_size if pop_size is not None else GA_POP_SIZE
        self.field      = Field(terrain_seed=TERRAIN_SEED, food_episode=0)
        self.ga         = GeneticAlgorithm(pop_size=pop, seed=0)
        self._spawn_agents()
        self.ga_frame   = 0
        self.ga_running = True
        self.goal_count = 0
        self.goal_reached_count = 0
        self.tracked_agent = None
        self.prev_best_genome   = None
        self.done_message = ""

    def _spawn_ga_agents(self):
        """_spawn_agentsの互換エイリアス。"""
        self._spawn_agents()

    def _spawn_agents(self):
        self.ga_agents = []
        self.tracked_agent = None
        for genome in self.ga.population:
            agent_field = self.field.clone_foods()
            agent = SageBruteAgent(genome, agent_field, self.field.start_pos)
            self.ga_agents.append(agent)

    # ----------------------------------------------------------------
    def init_backroom_mode(self):
        """バックルームモードを初期化する。"""
        self.br_bits4    = 0
        self.br_waveform = None
        self.br_mic_data = None
        self.br_bn       = _AudioBN()
        # マイクスレッドを開始
        self._start_mic_thread()

    def _start_mic_thread(self):
        """PhonemeDecoderを別スレッドで定期実行する。"""
        import threading
        if self.br_mic_thread_running:
            return
        self.br_mic_thread_running = True

        def _mic_worker():
            from game.phoneme import PhonemeDecoder
            from config import AUDIO_SAMPLE_RATE, PHONEME_VOWEL, PHONEME_FORMANTS
            import numpy as np

            dec = PhonemeDecoder()
            if not dec.available:
                self.br_mic_data = {
                    "available":    False,
                    "f1":           0,
                    "f2":           0,
                    "vowel":        "",
                    "decoded_bits": 0,
                }
                self.br_mic_thread_running = False
                return

            while self.br_mic_thread_running:
                frame = dec.record_frame()
                decoded = dec.decode(frame)   # 2bits

                # F1/F2ピーク周波数をPhonemeDecoderのメソッドで取得
                f1, f2 = dec.detect_f1_f2(frame)

                vowel_bits = decoded & 0x3
                vowel_char = PHONEME_VOWEL.get(vowel_bits, "?")

                self.br_mic_data = {
                    "available":    True,
                    "f1":           f1,
                    "f2":           f2,
                    "vowel":        vowel_char,
                    "decoded_bits": decoded,
                }

        t = threading.Thread(target=_mic_worker, daemon=True)
        t.start()

    def _stop_mic_thread(self):
        self.br_mic_thread_running = False

    def _update_backroom(self, dt: float):
        """バックルームモードの更新。"""
        self.br_bn.tick(dt)

    def _handle_backroom_key(self, key):
        """バックルームモードのキー入力処理。"""
        # ESC / M: メニューへ
        if key in (pygame.K_ESCAPE, pygame.K_m):
            self._stop_mic_thread()
            self.state = GameState.MENU
            return

        # V: 音声ON/OFF
        if key == pygame.K_v:
            self.br_bn.toggle_audio()
            return

        # 0～9 / a～f: 4bits手動入力
        hex_map = {
            pygame.K_0: 0,  pygame.K_1: 1,  pygame.K_2: 2,  pygame.K_3: 3,
            pygame.K_4: 4,  pygame.K_5: 5,  pygame.K_6: 6,  pygame.K_7: 7,
            pygame.K_8: 8,  pygame.K_9: 9,
            pygame.K_a: 10, pygame.K_b: 11, pygame.K_c: 12,
            pygame.K_d: 13, pygame.K_e: 14, pygame.K_f: 15,
        }
        if key in hex_map:
            bits4 = hex_map[key]
            bits2 = bits4 & 0x3   # 2bitsのみ使用
            self.br_bits4 = bits2

            # 波形を合成（常に実行）
            from game.phoneme import PhonemeConverter
            # br_bnのコンバーターを使い回す（または新規作成）
            if self.br_bn.converter is None:
                self.br_bn.enable_audio()   # converterを初期化
            conv = self.br_bn.converter if self.br_bn.converter else PhonemeConverter()
            self.br_waveform = conv.synthesize(bits2)

            # 音声は常に再生（Vキーの状態に関わらず）
            # バックルームはデバッグモードなので常に音を出す
            conv.play(bits2)

    # ----------------------------------------------------------------
    def _do_save(self, auto: bool = False):
        """現在のGA状態をDBに保存する。"""
        if self.ga is None:
            return
        try:
            rid = save_manager.save_model(
                self.ga,
                terrain_seed=42,
                goal_count=self.goal_reached_count,
            )
            prefix = "自動セーブ" if auto else "セーブ"
            self.save_toast_msg   = f"{prefix}完了  Gen {self.ga.generation}  ID={rid}"
            self.save_toast_timer = FPS * 3
        except Exception as e:
            self.save_toast_msg   = f"セーブ失敗: {e}"
            self.save_toast_timer = FPS * 3

    def _open_load_screen(self):
        """ロード画面を開く。"""
        self.load_models  = save_manager.list_models()
        self.load_sel_idx = 0
        self.load_error   = ""
        self.state        = GameState.LOAD

    def _do_load(self):
        """選択中のモデルをロードしてLOAD_RESUME画面へ遷移する。"""
        if not self.load_models:
            return
        m = self.load_models[self.load_sel_idx]
        if not m["compatible"]:
            self.load_error = "互換性エラー: NN構造が一致しません"
            return
        try:
            pop_size = m["pop_size"] if isinstance(m["pop_size"], int) else GA_POP_SIZE
            self.init_ga_mode(pop_size=pop_size)
            save_manager.load_model(m["id"], self.ga)
            # エージェントを再生成
            self._spawn_ga_agents()
            # ロード元情報を記録
            self.loaded_from_gen  = self.ga.generation
            self.load_resume_meta = m
            self.load_error = ""
            # LOAD_RESUME画面へ遷移（モード選択を促す）
            self.state = GameState.LOAD_RESUME
        except Exception as e:
            self.load_error = f"ロード失敗: {e}"

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
            if not agent.alive:
                continue
            all_done = False
            result = agent.step()
            if result["done"]:
                agent.genome.fitness = agent.total_reward
                agent.alive = False
                goal_pos = pygame.Vector2(*self.field.goal_pos)
                if (agent.pos - goal_pos).length() < GOAL_RADIUS:
                    self.goal_reached_count += 1

        if all_done:
            # 学習完了判定
            if self.goal_reached_count >= self.fast_cfg_goal_count:
                self._evolve_generation(auto_save=True)  # 完了時自動セーブ
                self.state = GameState.GA   # 監視モードへ自動遷移
                return True
            self._evolve_generation()
            return True

        return False

    def _evolve_generation(self, auto_save: bool = False):
        """世代進化の共通処理。"""
        # 前世代の最高ゲノムをスナップショット保存
        best = self.ga.get_best()
        self.prev_best_genome = copy.deepcopy(best)
        # 自動セーブ（学習完了時）
        if auto_save:
            self._do_save(auto=True)
        self.ga.evolve()
        self.field.reset_foods(food_episode=self.ga.generation)
        self._spawn_agents()
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

            # メニュー返報確認ダイアログ中
            if self.confirm_menu:
                if event.key == pygame.K_y:
                    self.confirm_menu = False
                    self.state = GameState.MENU
                elif event.key in (pygame.K_n, pygame.K_ESCAPE):
                    self.confirm_menu = False
                return

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
                elif event.key == pygame.K_l:
                    self._open_load_screen()
                elif event.key == pygame.K_o:
                    self.init_backroom_mode()
                    self.state = GameState.BACKROOM

            elif self.state == GameState.BACKROOM:
                self._handle_backroom_key(event.key)

            elif self.state == GameState.LOAD:
                if event.key in (pygame.K_ESCAPE, pygame.K_m):
                    self.state = GameState.MENU
                elif event.key == pygame.K_UP:
                    self.load_sel_idx = max(0, self.load_sel_idx - 1)
                    self.load_error = ""
                elif event.key == pygame.K_DOWN:
                    self.load_sel_idx = min(
                        len(self.load_models) - 1, self.load_sel_idx + 1)
                    self.load_error = ""
                elif event.key == pygame.K_RETURN:
                    self._do_load()
                elif event.key == pygame.K_DELETE:
                    if self.load_models:
                        mid = self.load_models[self.load_sel_idx]["id"]
                        save_manager.delete_model(mid)
                        self.load_models = save_manager.list_models()
                        self.load_sel_idx = min(
                            self.load_sel_idx, max(0, len(self.load_models) - 1))

            elif self.state == GameState.LOAD_RESUME:
                if event.key == pygame.K_ESCAPE:
                    # ロード画面に戻る
                    self.state = GameState.LOAD
                elif event.key == pygame.K_2:
                    # 監視モードで再開
                    self.save_toast_msg   = f"ロード完了  Gen {self.ga.generation}  ID={self.load_resume_meta['id']}"
                    self.save_toast_timer = FPS * 3
                    self.state = GameState.GA
                elif event.key == pygame.K_3:
                    # 高速モードで再開
                    self.save_toast_msg   = f"ロード完了  Gen {self.ga.generation}  ID={self.load_resume_meta['id']}"
                    self.save_toast_timer = FPS * 3
                    self.state = GameState.GA_FAST

            elif self.state == GameState.GA_FAST_CFG:
                self._handle_fast_cfg_key(event.key)

            elif self.state == GameState.PLAYER:
                if event.key == pygame.K_m:
                    self.state = GameState.MENU
                if event.key == pygame.K_r:
                    self.init_player_mode()

            elif self.state == GameState.GA:
                if event.key == pygame.K_m:
                    if self.ga is not None and self.ga.generation > 0:
                        self.confirm_menu = True
                    else:
                        self.state = GameState.MENU
                elif event.key == pygame.K_s:
                    self._do_save(auto=False)
                elif event.key == pygame.K_v:
                    # 音声ON/OFF切替
                    new_state = self.audio_bn.toggle_audio()
                    # 全エージェントのbnを一度全部オフにしてから、
                    # tracked_agentのみに音声を属当する
                    for ag in self.ga_agents:
                        if hasattr(ag, 'bn'):
                            ag.bn.audio_enabled = False
                    if new_state and self.tracked_agent is not None:
                        if hasattr(self.tracked_agent, 'bn'):
                            self.tracked_agent.bn.audio_enabled = True
                            self.tracked_agent.bn.converter     = self.audio_bn.converter
                elif event.key == pygame.K_TAB:
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
        elif self.state == GameState.BACKROOM:
            self._update_backroom(dt)

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

    # tracked_agentを再計算する間隔（1000フレームごと）
    TRACK_UPDATE_INTERVAL = 1000

    def _update_ga_monitor(self, dt: float):
        """監視モード（通常速度）のGA更新。
        tracked_agentは1000フレームごとに再計算する。
        頑繁に切り替わるとUIと音声の整合性が崩れるため。
        """
        if not self.ga_running:
            return
        self._step_ga_once()

        alive_agents = [ag for ag in self.ga_agents if ag.alive]

        # tracked_agentが死亡した場合は即座に再選択する
        if self.tracked_agent is not None and not self.tracked_agent.alive:
            self._switch_tracked_agent(alive_agents)
        # tracked_agentが未設定の場合も即座に選択する
        elif self.tracked_agent is None and alive_agents:
            self._switch_tracked_agent(alive_agents)
        # 1000フレームごとに一位個体を再計算して追従先を更新
        elif self.ga_frame % self.TRACK_UPDATE_INTERVAL == 0 and alive_agents:
            self._switch_tracked_agent(alive_agents)

        # 追従エージェントに音声を属当（音声ON時のみ）
        if self.tracked_agent is not None and hasattr(self.tracked_agent, 'bn'):
            if self.audio_bn.audio_enabled:
                self.tracked_agent.bn.audio_enabled = True
                self.tracked_agent.bn.converter     = self.audio_bn.converter
            else:
                self.tracked_agent.bn.audio_enabled = False

    def _switch_tracked_agent(self, alive_agents: list):
        """追従エージェントを切り替える。
        旧追従先の音声を必ずオフにしてから新しい追従先を設定する。
        """
        if not alive_agents:
            if self.tracked_agent is not None and hasattr(self.tracked_agent, 'bn'):
                self.tracked_agent.bn.audio_enabled = False
            self.tracked_agent = None
            return
        new_best = max(alive_agents, key=lambda ag: ag.total_reward)
        if self.tracked_agent is not new_best:
            if self.tracked_agent is not None and hasattr(self.tracked_agent, 'bn'):
                self.tracked_agent.bn.audio_enabled = False
            self.tracked_agent = new_best

    # ----------------------------------------------------------------
    def _get_camera_focus(self) -> pygame.Vector2:
        alive_agents = [ag for ag in self.ga_agents if ag.alive]
        if not alive_agents:
            return pygame.Vector2(self.field.start_pos)
        if len(alive_agents) > CAMERA_SWITCH_THRESHOLD:
            cx = sum(ag.pos.x for ag in alive_agents) / len(alive_agents)
            cy = sum(ag.pos.y for ag in alive_agents) / len(alive_agents)
            return pygame.Vector2(cx, cy)
        else:
            if self.tracked_agent is not None and self.tracked_agent.alive:
                return pygame.Vector2(self.tracked_agent.pos)
            best = max(alive_agents, key=lambda ag: ag.total_reward)
            return pygame.Vector2(best.pos)

    # ----------------------------------------------------------------
    def _draw(self):
        self.screen.fill(C_BG)

        # メニュー返報確認ダイアログ（最前面）
        if self.confirm_menu:
            self._draw_confirm_menu()
            pygame.display.flip()
            return

        # リセット確認ダイアログ（最前面）
        if self.confirm_reset:
            self._draw_confirm_reset()
            pygame.display.flip()
            return

        # セーブ・ロードトースト（全状態で表示）
        if self.save_toast_timer > 0:
            self.save_toast_timer -= 1
            alpha = min(220, self.save_toast_timer * 10)
            self.renderer.draw_save_toast(self.save_toast_msg, alpha)

        if self.state == GameState.LOAD:
            self.renderer.draw_load_screen(
                self.load_models, self.load_sel_idx, self.load_error)
            return

        if self.state == GameState.LOAD_RESUME:
            self._draw_load_resume()
            return

        if self.state == GameState.BACKROOM:
            self.renderer.draw_backroom(
                bottleneck=self.br_bn,
                manual_bits4=self.br_bits4,
                audio_on=self.br_bn.audio_enabled,
                waveform=self.br_waveform,
                mic_data=self.br_mic_data,
            )
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
        """通常GAモードの描画。
        追従エージェントはself.tracked_agentを使用し、
        更新・音声と同じエージェントを参照する。
        """
        focus = self._get_camera_focus()
        cam   = self.renderer.calc_camera(focus)

        self.renderer.draw_field(self.field, cam)
        self._ensure_minimap()

        alive_count = 0
        for ag in self.ga_agents:
            if not ag.alive:
                continue
            alive_count += 1
            self._draw_agent_dot(ag, cam, color=(40, 100, 160))

        # tracked_agentを追従エージェントとして使用
        tracked = self.tracked_agent
        if tracked and tracked.alive:
            self._draw_agent_dot(tracked, cam, color=(255, 200, 50))

        self.renderer.draw_minimap(self.field, focus, self.field.goal_pos)
        self._draw_ga_overlay(alive_count, tracked)
        self.renderer.draw_fitness_graph(self.ga, 20, 150, w=280, h=100)

        # アクティベーションモニター: tracked_agentのゲノムとbnを使用
        if tracked and tracked.alive:
            display_genome = tracked.genome
            bn_for_panel   = tracked.bn if hasattr(tracked, 'bn') else self.audio_bn
        elif self.prev_best_genome is not None:
            display_genome = self.prev_best_genome
            bn_for_panel   = self.audio_bn
        else:
            display_genome = None
            bn_for_panel   = self.audio_bn

        if display_genome is not None:
            # 3パネルモニター（画面下部中央）
            total_w = 280 * 3 + 8 * 2
            mx = SCREEN_W // 2 - total_w // 2
            my = SCREEN_H - 230
            self.renderer.draw_rnn_monitor_panels(
                display_genome, bn_for_panel,
                x=mx, y=my, panel_w=280, panel_h=220)

        hint = self.renderer.font_s.render(
            "S: Save  V: Audio  Tab: Fast Mode  M: Menu  ESC: Quit", True, C_GRAY)
        self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 20))

    def _draw_agent_dot(self, agent, cam, color):
        """SageBruteAgentを円として描画する。"""
        sx = int(agent.pos.x - cam.x)
        sy = int(agent.pos.y - cam.y)
        if 0 <= sx < SCREEN_W and 0 <= sy < SCREEN_H:
            pygame.draw.circle(self.screen, color, (sx, sy), 6)
            # 進行方向を線で示す
            import math
            rad = math.radians(agent.angle)
            ex = sx + int(math.cos(rad) * 10)
            ey = sy + int(math.sin(rad) * 10)
            pygame.draw.line(self.screen, (255, 255, 255), (sx, sy), (ex, ey), 2)

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

        lines = [
            f"Gen: {stats['generation']:4d}   Pop: {self.ga.pop_size}",
            f"Alive: {alive_count:3d} / {self.ga.pop_size}  [{cam_mode}]",
            f"Best:  {stats['best']:8.1f}",
            f"Avg:   {stats['avg']:8.1f}",
            f"Species: {stats.get('species','-'):3}  "
            f"Mut: {stats.get('mut_rate',0):.3f}",
            f"Frame: {self.ga_frame}",
        ]
        if self.loaded_from_gen is not None:
            lines.insert(0, f"LOADED  Gen:{self.loaded_from_gen} → 継続中")

        hud_h = 12 + len(lines) * 18
        bg = pygame.Surface((280, hud_h), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 200))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (194, 24, 91), (x, y, 280, hud_h), 1)

        for i, line in enumerate(lines):
            color = (255, 220, 80) if (i == 0 and self.loaded_from_gen is not None) else (244, 143, 177)
            t = font.render(line, True, color)
            self.screen.blit(t, (x + 6, y + 6 + i * 18))

        # ゴール到達カウンタ
        goal_color = (50, 220, 150) if self.goal_reached_count > 0 else C_GRAY
        gt = font.render(
            f"GOAL REACHED: {self.goal_reached_count}/{self.fast_cfg_goal_count}",
            True, goal_color)
        self.screen.blit(gt, (x + 6, y + hud_h + 4))

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

        # アクティベーションモニター: 高速モード中は最高適応度ゲノムを表示
        fast_display_genome = self.ga.get_best() if self.ga else None
        if fast_display_genome is not None:
            total_w = 280 * 3 + 8 * 2
            mx = SCREEN_W // 2 - total_w // 2
            self.renderer.draw_rnn_monitor_panels(
                fast_display_genome, self.audio_bn,
                x=mx, y=SCREEN_H - 230, panel_w=280, panel_h=220)

        hints = [
            "Tab: Switch to Monitor  Enter: Exit Fast Mode",
            "R: Reset (with confirm)  M: Menu  ESC: Quit",
        ]
        for i, h in enumerate(hints):
            ht = font_s.render(h, True, C_GRAY)
            self.screen.blit(ht, (SCREEN_W // 2 - ht.get_width() // 2,
                                  SCREEN_H - 50 + i * 20))

    # ----------------------------------------------------------------
    def _draw_load_resume(self):
        """ロード後の再開モード選択画面。"""
        font_l = self.renderer.font_l
        font_m = self.renderer.font_m
        font_s = self.renderer.font_s

        # 背景オーバーレイ
        ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((5, 8, 18, 240))
        self.screen.blit(ov, (0, 0))

        # タイトル
        t_title = font_l.render("モデルをロードしました", True, (100, 220, 255))
        self.screen.blit(t_title, (SCREEN_W // 2 - t_title.get_width() // 2, 120))

        # メタデータ表示
        m = self.load_resume_meta
        if m is not None:
            gen_val  = self.ga.generation if self.ga else m.get("generation", "?")
            best_val = m.get("best_fitness", 0.0)
            saved_at = m.get("saved_at", "")
            # ISO形式の日時を「YYYY-MM-DD HH:MM」形式に変換
            try:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(saved_at)
                saved_at_str = dt.astimezone().strftime("%Y-%m-%d %H:%M")
            except Exception:
                saved_at_str = saved_at[:16] if len(saved_at) >= 16 else saved_at

            info_lines = [
                f"Gen: {gen_val}    Best: {best_val:.1f}",
                f"保存日時: {saved_at_str}",
            ]
            for i, line in enumerate(info_lines):
                t = font_m.render(line, True, (200, 200, 220))
                self.screen.blit(t, (SCREEN_W // 2 - t.get_width() // 2, 200 + i * 34))

        # 選択ボタン
        bx, by, bw, bh = SCREEN_W // 2 - 260, 300, 520, 180
        pygame.draw.rect(self.screen, (10, 15, 25), (bx, by, bw, bh), border_radius=10)
        pygame.draw.rect(self.screen, (100, 180, 255), (bx, by, bw, bh), 2, border_radius=10)

        btn_items = [
            ("[2]  監視モードで再開",   (50, 220, 150)),
            ("[3]  高速学習モードで再開", (255, 200, 50)),
        ]
        for i, (label, color) in enumerate(btn_items):
            t = font_m.render(label, True, color)
            self.screen.blit(t, (SCREEN_W // 2 - t.get_width() // 2, by + 24 + i * 60))

        # ESCキャンセルヒント
        hint = font_s.render("[ESC] キャンセル (ロード画面に戻る)", True, C_GRAY)
        self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 30))

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

        # ---- 開始ボタン ----
        st = font_l.render("Enter / Space: START", True, (50, 220, 150))
        self.screen.blit(st, (SCREEN_W // 2 - st.get_width() // 2, 420))

        hint = font_s.render("ESC / M: Back to Menu", True, C_GRAY)
        self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 30))

        # リセット確認
        if self.confirm_reset:
            self._draw_confirm_reset()

    def _draw_confirm_menu(self):
        """メニュー返報確認ダイアログ。GA進行中にMキーを押したときに表示。"""
        font_m = self.renderer.font_m
        font_s = self.renderer.font_s

        ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 180))
        self.screen.blit(ov, (0, 0))

        bx, by, bw, bh = SCREEN_W // 2 - 260, SCREEN_H // 2 - 80, 520, 160
        pygame.draw.rect(self.screen, (10, 15, 25), (bx, by, bw, bh), border_radius=10)
        pygame.draw.rect(self.screen, (255, 160, 30), (bx, by, bw, bh), 2, border_radius=10)

        t1 = font_m.render("メニューに戻りますか？", True, (255, 180, 50))
        self.screen.blit(t1, (SCREEN_W // 2 - t1.get_width() // 2, by + 18))

        t2 = font_s.render(
            f"Gen {self.ga.generation} までの学習データは保持されます。",
            True, C_GRAY)
        self.screen.blit(t2, (SCREEN_W // 2 - t2.get_width() // 2, by + 52))

        t3 = font_s.render(
            "（再度 [2] または [3] で再開できます）",
            True, (120, 120, 140))
        self.screen.blit(t3, (SCREEN_W // 2 - t3.get_width() // 2, by + 74))

        ty = font_m.render("[Y] はい", True, (255, 160, 30))
        tn = font_m.render("[N] キャンセル", True, (80, 180, 80))
        self.screen.blit(ty, (SCREEN_W // 2 - 180, by + 108))
        self.screen.blit(tn, (SCREEN_W // 2 + 20, by + 108))

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

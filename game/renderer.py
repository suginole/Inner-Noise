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

    # 日本語対応フォントのパス（優先順: Mac → Linux → フォールバック）
    _JP_FONTS = [
        # macOS
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # Linux (IPAゴシック)
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/truetype/ipafont/ipag.ttf",
        # Linux (Noto CJK)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]

    # クラスレベルキャッシュ（同じサイズなら再読み込み不要）
    _font_cache: dict = {}

    @staticmethod
    def _load_font(size: int) -> pygame.font.Font:
        """日本語対応フォントを読み込む。見つからなければSysFontにフォールバック。"""
        import os
        if size in Renderer._font_cache:
            # キャッシュされたパスから再読み込み
            cached_path = Renderer._font_cache[size]
            if cached_path is None:
                return pygame.font.SysFont(None, size)
            return pygame.font.Font(cached_path, size)

        for path in Renderer._JP_FONTS:
            if os.path.exists(path):
                Renderer._font_cache[size] = path
                return pygame.font.Font(path, size)

        # フォールバック: pygameのデフォルトフォント
        Renderer._font_cache[size] = None
        return pygame.font.SysFont(None, size)

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font_s = self._load_font(12)
        self.font_m = self._load_font(15)
        self.font_l = self._load_font(20)

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
    def draw_bottleneck_dummy(self, x: int, y: int,
                              pulse_state: list[int] | None = None,
                              history: list[list[int]] | None = None,
                              mode: str = "listen",
                              turn_progress: float = 0.0,
                              phoneme: str = "",
                              audio_on: bool = False,
                              is_dummy: bool = True):
        """
        ボトルネック通信路の可視化パネル。

        将来のRNN実装時は引数を実際のパルスデータに差し替えるだけでよい。

        Args:
            x, y: 描画座標
            pulse_state: 4要素のバイナリリスト [p1,p2,p3,p4] (Noneの場合はダミー)
            history: 直近20パルスの履歴 (Noneの場合はダミー)
            mode: "listen" | "speak"
            turn_progress: ターン進捗 0.0～1.0
            phoneme: 現在の音素文字列（例: 'ま'）
            audio_on: 音声出力が有効かどうか
            is_dummy: Trueの場合は [DUMMY] ラベルを表示
        """
        import random
        W = 260
        H = 90

        bg = pygame.Surface((W, H), pygame.SRCALPHA)
        bg.fill((8, 10, 18, 220))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (80, 60, 20), (x, y, W, H), 1)

        # タイトル
        dummy_tag = "  [DUMMY]" if is_dummy else ""
        title = self.font_s.render(
            f"BOTTLENECK  5Hz / 4bits{dummy_tag}", True, (180, 140, 60))
        self.screen.blit(title, (x + 4, y + 3))

        # ダミーパルス状態（全パターン均等サイクル）
        if pulse_state is None:
            # 5Hzパルスに合わせて全パターン（00/01/10/11）を均等に循環
            import time
            # 0.2秒ごとに切り替わる（全パターンを展開）
            t_idx = int(time.time() * 5) % 4   # 0,1,2,3 を循環
            pulse_state = [(t_idx >> (1 - i)) & 1 for i in range(2)]

        if history is None:
            # ダミー履歴（全パターンを展開）
            import time
            history = []
            for k in range(16):
                ts = int(time.time() * 5 - k) % 4
                history.append([(ts >> (1 - i)) & 1 for i in range(2)])

        # モード表示（方向ラベル + 性別記号）
        mode_color = (100, 160, 220) if mode == "listen" else (100, 200, 120)
        gender_sym = "♀" if mode == "listen" else "♂"   # S→M=高ピッチ♀ / M→S=低ピッチ♂
        mode_str = f"LISTEN  S→M  {gender_sym}" if mode == "listen" else f"SPEAK   M→S  {gender_sym}"
        mode_lbl = self.font_s.render(mode_str, True, mode_color)
        self.screen.blit(mode_lbl, (x + 4, y + 17))

        # ターン進捗バー
        pygame.draw.rect(self.screen, C_DARK, (x + 4, y + 30, W - 8, 5))
        pygame.draw.rect(self.screen, mode_color,
                         (x + 4, y + 30, int((W - 8) * turn_progress), 5))

        # 現在パルス 2 bits
        for i, bit in enumerate(pulse_state[:2]):
            bx = x + 4 + i * 40   # 2マスなので間隔を広く
            by = y + 40
            col = C_PULSE_ON if bit else C_PULSE_OFF
            pygame.draw.rect(self.screen, col, (bx, by, 32, 32))
            pygame.draw.rect(self.screen, C_GRAY, (bx, by, 32, 32), 1)
            lt = self.font_s.render(str(bit), True, C_WHITE if bit else C_GRAY)
            self.screen.blit(lt, (bx + 16 - lt.get_width() // 2, by + 8))

        # 履歴（小さく）
        for hi, hp in enumerate(history[-16:]):
            for bi, bit in enumerate(hp[:2]):
                bx = x + 115 + hi * 9
                by = y + 40 + bi * 12
                col2 = C_PULSE_ON if bit else C_PULSE_OFF
                pygame.draw.rect(self.screen, col2, (bx, by, 7, 7))

        # 音素表示
        if phoneme:
            ph_t = self.font_m.render(f"「{phoneme}」", True, (255, 220, 100))
            self.screen.blit(ph_t, (x + 4, y + H - 20))

        # 音声出力インジケーター
        audio_c = (80, 220, 100) if audio_on else (100, 100, 120)
        audio_sym = "● ON" if audio_on else "○ OFF"
        audio_t = self.font_s.render(f"V: 音声 {audio_sym}", True, audio_c)
        self.screen.blit(audio_t, (x + W - audio_t.get_width() - 4, y + H - 16))

    # ----------------------------------------------------------------
    def draw_load_screen(self, models: list[dict], selected_idx: int,
                         error_msg: str = "") -> None:
        """モデルロード画面。

        Args:
            models: save_manager.list_models() の返り値
            selected_idx: 現在選択中のインデックス
            error_msg: 互換性エラーなどのメッセージ
        """
        self.screen.fill(C_BG)

        title = self.font_l.render("モデルをロード", True, C_WHITE)
        self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 30))

        if not models:
            t = self.font_m.render("保存済みモデルがありません", True, C_GRAY)
            self.screen.blit(t, (SCREEN_W // 2 - t.get_width() // 2, SCREEN_H // 2))
        else:
            # ヘッダー
            hx = 40
            cols = [40, 220, 340, 430, 530, 650, 760]
            headers = ["ID", "保存日時", "Gen", "Best", "Avg", "Goal", "Pop"]
            for i, (hdr, cx) in enumerate(zip(headers, cols)):
                ht = self.font_s.render(hdr, True, (160, 160, 200))
                self.screen.blit(ht, (cx, 80))
            pygame.draw.line(self.screen, (60, 60, 80),
                             (30, 96), (SCREEN_W - 30, 96), 1)

            # リスト表示（最大10件）
            visible = models[:10]
            for i, m in enumerate(visible):
                ry = 104 + i * 38
                is_sel = (i == selected_idx)

                # 選択中は背景色
                if is_sel:
                    pygame.draw.rect(self.screen, (20, 30, 50),
                                     (30, ry - 2, SCREEN_W - 60, 34),
                                     border_radius=4)
                    pygame.draw.rect(self.screen, (80, 120, 200),
                                     (30, ry - 2, SCREEN_W - 60, 34),
                                     1, border_radius=4)

                # 互換性アイコン
                compat_c = (80, 200, 100) if m["compatible"] else (200, 80, 80)
                compat_t = self.font_s.render(
                    "OK" if m["compatible"] else "NG", True, compat_c)

                # 日時を短縮
                saved_at = m["saved_at"][:16].replace("T", " ") if m["saved_at"] else "---"

                row_color = C_WHITE if is_sel else (180, 180, 200)
                vals = [
                    str(m["id"]),
                    saved_at,
                    str(m["generation"]),
                    f"{m['best_fitness']:.1f}",
                    f"{m['avg_fitness']:.1f}",
                    str(m["goal_count"]),
                    str(m["pop_size"]),
                ]
                for val, cx in zip(vals, cols):
                    vt = self.font_s.render(val, True, row_color)
                    self.screen.blit(vt, (cx, ry + 4))
                self.screen.blit(compat_t, (SCREEN_W - 60, ry + 4))

        # エラーメッセージ
        if error_msg:
            et = self.font_m.render(error_msg, True, (220, 80, 80))
            self.screen.blit(et, (SCREEN_W // 2 - et.get_width() // 2, SCREEN_H - 120))

        # キーヒント
        hints = [
            "↑↓: 選択    Enter: 監視モードでロード    Tab: 高速モードでロード    Del: 削除    ESC: 戻る",
        ]
        for i, h in enumerate(hints):
            ht = self.font_s.render(h, True, C_GRAY)
            self.screen.blit(ht, (SCREEN_W // 2 - ht.get_width() // 2,
                                  SCREEN_H - 30 - i * 20))

    # ----------------------------------------------------------------
    def draw_save_toast(self, msg: str, alpha: int = 220):
        """セーブ完了トースト通知。"""
        t = self.font_m.render(msg, True, C_WHITE)
        w, h = t.get_width() + 24, t.get_height() + 14
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((20, 60, 30, alpha))
        x = SCREEN_W // 2 - w // 2
        y = 20
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (80, 200, 100), (x, y, w, h), 1, border_radius=4)
        self.screen.blit(t, (x + 12, y + 7))

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
            ("1", "PLAYER MODE",     "キーボードで車を直接操作する",                         C_CAR),
            ("2", "GA MODE",         "遠伝的アルゴリズムの学習を観察する",                   C_CAR_GA),
            ("3", "FAST LEARN MODE", "描画スキップの高速学習モード（Tabで監視切替）",   (255, 200, 50)),
            ("L", "LOAD MODEL",      "保存済みモデルをロードする",                         (180, 140, 220)),
            ("O", "BACKROOM MODE",   "音声入出力確認・デバッグモード",                   (255, 120, 180)),
        ]
        for i, (key, name, desc, color) in enumerate(options):
            oy = 270 + i * 82   # 5項目に収まるよう間隔を縮小
            pygame.draw.rect(self.screen, (20, 22, 35),
                             (SCREEN_W // 2 - 280, oy, 560, 68), border_radius=8)
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

    # ================================================================
    # バックルームモード描画
    # ================================================================
    def draw_backroom(self,
                      bottleneck,
                      manual_bits4: int,
                      audio_on: bool,
                      waveform: "np.ndarray | None",
                      mic_data: dict | None):
        """
        バックルームモードの全画面を描画する。

        Args:
            bottleneck: Bottleneckインスタンス
            manual_bits4: 手動入力された4bits値（0～15）
            audio_on: 音声出力が有効かどうか
            waveform: synthesize()の結果（np.ndarray）またはNone
            mic_data: マイク解析結果の辞書またはNone
              {
                'f1': float, 'f2': float,
                'vowel': str, 'consonant': str,
                'decoded_bits': int,
                'available': bool,
              }
        """
        import numpy as np
        from config import PHONEME_TABLE, PHONEME_FORMANTS, PHONEME_VOWEL

        self.screen.fill((0, 0, 0))   # 黒背景

        # ---- タイトル ----
        title = self.font_l.render("BACKROOM MODE  —  音声入出力確認", True, (255, 120, 180))
        self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 20))

        # ---- ボトルネックパネル（中央上部） ----
        bn_x = SCREEN_W // 2 - 130
        bn_y = 65
        bn_pulse = bottleneck.get_current_pulse()
        bn_history = bottleneck.get_pulse_history()
        self.draw_bottleneck_dummy(
            x=bn_x, y=bn_y,
            pulse_state=bn_pulse,
            history=bn_history,
            mode=bottleneck.get_mode(),
            turn_progress=bottleneck.get_turn_progress(),
            phoneme=bottleneck.get_last_phoneme(),
            audio_on=audio_on,
            is_dummy=True,
        )

        # ---- 出力デバッグパネル（左列） ----
        self._draw_output_debug(
            x=40, y=180,
            bits4=manual_bits4,
            audio_on=audio_on,
            waveform=waveform,
        )

        # ---- 入力デバッグパネル（右列） ----
        self._draw_input_debug(
            x=SCREEN_W // 2 + 40, y=180,
            mic_data=mic_data,
            expected_bits4=manual_bits4,
        )

        # ---- キーヒント ----
        hints = [
            "0～9 / a～f: 手動パルス入力    V: 音声ON/OFF    ESC/M: メニューへ戻る",
        ]
        for i, h in enumerate(hints):
            ht = self.font_s.render(h, True, C_GRAY)
            self.screen.blit(ht, (SCREEN_W // 2 - ht.get_width() // 2,
                                  SCREEN_H - 25 + i * 18))

    def _draw_output_debug(self, x: int, y: int, bits4: int,
                           audio_on: bool, waveform):
        """OUTPUT DEBUGパネル。"""
        import numpy as np
        from config import (
            PHONEME_TABLE, PHONEME_FORMANTS, PHONEME_VOWEL,
            AUDIO_FRAME_SAMPLES,
        )

        W, H = SCREEN_W // 2 - 60, 460
        bg = pygame.Surface((W, H), pygame.SRCALPHA)
        bg.fill((5, 5, 15, 230))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (255, 120, 180), (x, y, W, H), 1, border_radius=6)

        title = self.font_m.render("OUTPUT DEBUG", True, (255, 120, 180))
        self.screen.blit(title, (x + 8, y + 8))

        # bits2 をビット列に分解
        bits2     = bits4 & 0x3   # 下位2bitsのみ使用
        bits_list = [(bits2 >> (1 - i)) & 1 for i in range(2)]
        vowel_char   = PHONEME_VOWEL.get(bits2, "?")
        phoneme_char = PHONEME_TABLE.get(bits2, "?")
        f1, f2       = PHONEME_FORMANTS.get(vowel_char, (0, 0))

        # パルスビット表示（2マス）
        bx0 = x + 8
        by0 = y + 32
        for i, bit in enumerate(bits_list):
            bx = bx0 + i * 50   # 2マスなので間隔を広く
            col = C_PULSE_ON if bit else C_PULSE_OFF
            pygame.draw.rect(self.screen, col, (bx, by0, 40, 40))
            pygame.draw.rect(self.screen, C_GRAY, (bx, by0, 40, 40), 1)
            bt = self.font_m.render(str(bit), True, C_WHITE if bit else C_GRAY)
            self.screen.blit(bt, (bx + 20 - bt.get_width() // 2, by0 + 10))

        # 音素情報（子音なし）
        ry = by0 + 52
        rows = [
            ("Phoneme", f"「{phoneme_char}」"),
            ("F1",      f"{f1} Hz"),
            ("F2",      f"{f2} Hz"),
            ("Vowel",   vowel_char),
        ]
        for label, val in rows:
            lt = self.font_s.render(f"{label}:", True, C_GRAY)
            vt = self.font_m.render(val, True, C_WHITE)
            self.screen.blit(lt, (x + 8, ry))
            self.screen.blit(vt, (x + 120, ry - 2))
            ry += 26

        # 波形プレビュー
        ry += 8
        wh = H - (ry - y) - 12
        ww = W - 16
        pygame.draw.rect(self.screen, (15, 15, 25), (x + 8, ry, ww, wh))
        pygame.draw.rect(self.screen, (50, 50, 70), (x + 8, ry, ww, wh), 1)

        if waveform is not None and len(waveform) > 0:
            # ダウンサンプルして描画
            n_draw = min(ww, len(waveform))
            step   = max(1, len(waveform) // n_draw)
            pts    = []
            for j in range(n_draw):
                sv = float(waveform[j * step])
                px = x + 8 + j
                py = int(ry + wh // 2 - sv * (wh // 2 - 2))
                py = max(ry, min(ry + wh - 1, py))
                pts.append((px, py))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, (100, 200, 255), False, pts, 1)
            wt = self.font_s.render("waveform", True, (60, 80, 100))
            self.screen.blit(wt, (x + 10, ry + 2))
        else:
            nt = self.font_s.render("[play to preview]", True, (60, 80, 100))
            self.screen.blit(nt, (x + 8 + ww // 2 - nt.get_width() // 2,
                                  ry + wh // 2 - 6))

    def _draw_input_debug(self, x: int, y: int,
                          mic_data: dict | None, expected_bits4: int):
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
            # マイク不可
            nt = self.font_m.render("MIC UNAVAILABLE", True, (200, 80, 80))
            self.screen.blit(nt, (x + W // 2 - nt.get_width() // 2,
                                  y + H // 2 - 10))
            st = self.font_s.render(
                "pyaudioマイクが検出できませんでした", True, C_GRAY)
            self.screen.blit(st, (x + W // 2 - st.get_width() // 2, y + H // 2 + 20))
            return

        ry = y + 36
        decoded_bits = mic_data.get("decoded_bits", 0)
        expected_char = PHONEME_TABLE.get(expected_bits4, "?")
        decoded_char  = PHONEME_TABLE.get(decoded_bits, "?")
        match = ((decoded_bits & 0x3) == (expected_bits4 & 0x3))

        rows = [
            ("Detected F1", f"{mic_data.get('f1', 0):.0f} Hz"),
            ("Detected F2", f"{mic_data.get('f2', 0):.0f} Hz"),
            ("Vowel",       mic_data.get("vowel", "?"),),
            ("Consonant",   str(mic_data.get("consonant", "?"))),
        ]
        for label, val in rows:
            lt = self.font_s.render(f"{label}:", True, C_GRAY)
            vt = self.font_m.render(val, True, C_WHITE)
            self.screen.blit(lt, (x + 8, ry))
            self.screen.blit(vt, (x + 140, ry - 2))
            ry += 26

        ry += 8
        # Decoded bits (2bits)
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

        # Expected bits (2bits)
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

        # Match
        match_c = (80, 220, 100) if match else (220, 80, 80)
        match_sym = "✓ MATCH" if match else "✗ MISMATCH"
        mt = self.font_m.render(f"Match: {match_sym}", True, match_c)
        self.screen.blit(mt, (x + 8, ry))

    # ================================================================
    # RNNボトルネック3パネルモニター
    # ================================================================
    def draw_rnn_monitor_panels(self, genome, bottleneck,
                                x: int = 0, y: int = 0,
                                panel_w: int = 280, panel_h: int = 220):
        """
        3パネル横並びモニター。
        左: SENSORY NN（青系）
        中: BOTTLENECK（黄白系）
        右: MOTOR NN（赤系）

        Args:
            genome:     GAGenome インスタンス
            bottleneck: RNNBottleneck または Bottleneck インスタンス
            x, y:       左端の描画座標
            panel_w/h:  各パネルのサイズ
        """
        gap = 8
        cx_bn = x + panel_w + gap
        cx_motor = x + (panel_w + gap) * 2

        # ---- 左パネル: SENSORY NN ----
        self._draw_sensory_panel(genome, x, y, panel_w, panel_h)

        # ---- 中央パネル: BOTTLENECK ----
        self._draw_bottleneck_panel(bottleneck, cx_bn, y, panel_w, panel_h)

        # ---- 右パネル: MOTOR NN ----
        self._draw_motor_panel(genome, cx_motor, y, panel_w, panel_h)

        # ---- 接続矢印 ----
        mode = bottleneck.get_mode() if hasattr(bottleneck, 'get_mode') else 'listen'
        self._draw_flow_arrows(x, y, panel_w, panel_h, gap, mode)

    def _draw_sensory_panel(self, genome, x, y, w, h):
        """左パネル: SENSORY NN（青系）
        4列: INPUT(12) / 感覚皮質FF(16) / 感覚GRU(16) / パルス符号化(2)
        """
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((5, 10, 30, 220))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (60, 100, 200), (x, y, w, h), 1, border_radius=4)

        title = self.font_s.render("SENSORY NN", True, (100, 160, 255))
        self.screen.blit(title, (x + 6, y + 4))

        inp_acts    = getattr(genome, 'last_input_act',   [0.0] * 12)
        cortex_acts = getattr(genome, 'last_cortex_act',  [0.0] * 16)
        gru_acts    = getattr(genome, 'last_sensory_gru', [0.0] * 16)
        pulse_acts  = getattr(genome, 'last_pulse_act',   [0, 0])

        inp_labels = ["E","Ga","Gx","Gy","Fx","Fy"] + \
                     [f"R{i}" for i in range(VISION_RAYS)] + ["Fc"]

        node_r   = 4
        node_gap = 12
        row_top  = y + 18
        # 4列の水平位置
        col_inp    = x + 26
        col_cortex = x + int(w * 0.38)
        col_gru    = x + int(w * 0.62)
        col_pulse  = x + w - 18

        def _bipolar_color(v):
            if v >= 0:
                t = min(1.0, v)
                return (int(40 + 215 * t), int(40 * (1-t)), 40)
            else:
                t = min(1.0, -v)
                return (40, int(40 * (1-t)), int(40 + 215 * t))

        def _unipolar_color(v):
            t = max(0.0, min(1.0, v))
            return (int(30 + 200 * t), int(30 + 200 * t), int(30 + 60 * t))

        def _draw_col(acts, cx, bipolar=True, labels=None, border_c=(50,80,150)):
            n = len(acts)
            sy = row_top + max(0, (h - 28 - n * node_gap) // 2)
            for i, v in enumerate(acts):
                ny = sy + i * node_gap
                c = _bipolar_color(float(v)) if bipolar else _unipolar_color(float(v))
                pygame.draw.circle(self.screen, c, (cx, ny), node_r)
                pygame.draw.circle(self.screen, border_c, (cx, ny), node_r, 1)
                if labels and i < len(labels):
                    lt = self.font_s.render(labels[i], True, (70, 100, 160))
                    self.screen.blit(lt, (cx - node_r - lt.get_width() - 2, ny - 5))
            return sy, n

        sy_inp,    n_inp    = _draw_col(inp_acts,    col_inp,    bipolar=False, labels=inp_labels)
        sy_cortex, n_cortex = _draw_col(cortex_acts, col_cortex, bipolar=True)
        sy_gru,    n_gru    = _draw_col(gru_acts,    col_gru,    bipolar=True)

        # パルス符号化列（2ビット）
        n_pulse = len(pulse_acts)
        sy_pulse = row_top + max(0, (h - 28 - n_pulse * 28) // 2)
        for i, bit in enumerate(pulse_acts):
            ny = sy_pulse + i * 28
            c = C_PULSE_ON if bit else C_PULSE_OFF
            pygame.draw.rect(self.screen, c, (col_pulse - 10, ny, 18, 20))
            pygame.draw.rect(self.screen, C_GRAY, (col_pulse - 10, ny, 18, 20), 1)
            bt = self.font_s.render(str(bit), True, C_WHITE if bit else C_GRAY)
            self.screen.blit(bt, (col_pulse - 10 + 9 - bt.get_width()//2, ny + 4))

        # 層間エッジ（上位8本）
        sensory_nn = getattr(genome, 'sensory', None)
        if sensory_nn is not None:
            self._draw_edges(inp_acts,    sy_inp,    col_inp,    node_r,
                             cortex_acts, sy_cortex, col_cortex, node_r,
                             getattr(sensory_nn, 'W_cortex', None))
            self._draw_edges(cortex_acts, sy_cortex, col_cortex, node_r,
                             gru_acts,    sy_gru,    col_gru,    node_r,
                             getattr(sensory_nn, 'Wh', None))

        # 凡例・列ラベル
        leg = self.font_s.render("赤=+ 暗=0 青=-", True, (80, 100, 160))
        self.screen.blit(leg, (x + 4, y + h - 14))
        for lbl, lx, lc in [
            ("IN",  col_inp,    (80, 120, 220)),
            ("FF",  col_cortex, (90, 130, 220)),
            ("GRU", col_gru,    (100, 140, 220)),
            ("P",   col_pulse,  (200, 200, 80)),
        ]:
            lt = self.font_s.render(lbl, True, lc)
            self.screen.blit(lt, (lx - lt.get_width() // 2, y + h - 14))

    def _draw_bottleneck_panel(self, bottleneck, x, y, w, h):
        """中央パネル: BOTTLENECK（黄白系・上下2段）"""
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((20, 18, 5, 220))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (200, 180, 50), (x, y, w, h), 1, border_radius=4)

        title = self.font_s.render("BOTTLENECK  2bits / 5Hz", True, (220, 200, 80))
        self.screen.blit(title, (x + 4, y + 3))

        pulse  = bottleneck.get_current_pulse() if hasattr(bottleneck, 'get_current_pulse') else [0, 0]
        hist   = bottleneck.get_pulse_history()  if hasattr(bottleneck, 'get_pulse_history')  else []
        mode   = bottleneck.get_mode()            if hasattr(bottleneck, 'get_mode')            else 'listen'
        prog   = bottleneck.get_turn_progress()   if hasattr(bottleneck, 'get_turn_progress')   else 0.0
        phoneme = bottleneck.get_last_phoneme()   if hasattr(bottleneck, 'get_last_phoneme')    else ''

        # 上段: S→M（傾聴ターン）
        uy = y + 18
        sm_c = (100, 160, 255) if mode == 'listen' else (60, 60, 80)
        sm_lbl = self.font_s.render("S→M  ♀", True, sm_c)
        self.screen.blit(sm_lbl, (x + 6, uy))
        if mode == 'listen':
            pygame.draw.rect(self.screen, (30, 30, 50), (x + 6, uy + 14, w - 12, 5))
            pygame.draw.rect(self.screen, sm_c, (x + 6, uy + 14, int((w - 12) * prog), 5))
            for i, bit in enumerate(pulse[:2]):
                bx = x + 6 + i * 36
                bc = C_PULSE_ON if bit else C_PULSE_OFF
                pygame.draw.rect(self.screen, bc, (bx, uy + 22, 28, 28))
                pygame.draw.rect(self.screen, C_GRAY, (bx, uy + 22, 28, 28), 1)
                bt = self.font_m.render(str(bit), True, C_WHITE if bit else C_GRAY)
                self.screen.blit(bt, (bx + 14 - bt.get_width() // 2, uy + 28))
            if phoneme:
                pt = self.font_m.render(f"「{phoneme}」", True, (255, 220, 100))
                self.screen.blit(pt, (x + 90, uy + 22))

        # 区切り線
        mid_y = y + h // 2
        pygame.draw.line(self.screen, (100, 90, 30), (x + 4, mid_y), (x + w - 4, mid_y), 1)

        # 下段: M→S（発話ターン）
        ly = mid_y + 6
        ms_c = (100, 200, 120) if mode == 'speak' else (60, 80, 60)
        ms_lbl = self.font_s.render("M→S  ♂", True, ms_c)
        self.screen.blit(ms_lbl, (x + 6, ly))
        if mode == 'speak':
            pygame.draw.rect(self.screen, (30, 50, 30), (x + 6, ly + 14, w - 12, 5))
            pygame.draw.rect(self.screen, ms_c, (x + 6, ly + 14, int((w - 12) * prog), 5))
            for i, bit in enumerate(pulse[:2]):
                bx = x + 6 + i * 36
                bc = C_PULSE_ON if bit else C_PULSE_OFF
                pygame.draw.rect(self.screen, bc, (bx, ly + 22, 28, 28))
                pygame.draw.rect(self.screen, C_GRAY, (bx, ly + 22, 28, 28), 1)
                bt = self.font_m.render(str(bit), True, C_WHITE if bit else C_GRAY)
                self.screen.blit(bt, (bx + 14 - bt.get_width() // 2, ly + 28))
            if phoneme:
                pt = self.font_m.render(f"「{phoneme}」", True, (255, 220, 100))
                self.screen.blit(pt, (x + 90, ly + 22))

        # パルス履歴グリッド（下部）
        hy = y + h - 28
        for hi, hp in enumerate(hist[-16:]):
            for bi, bit in enumerate(hp[:2]):
                bx = x + 6 + hi * 16
                by = hy + bi * 10
                c = C_PULSE_ON if bit else C_PULSE_OFF
                pygame.draw.rect(self.screen, c, (bx, by, 12, 8))

    def _draw_motor_panel(self, genome, x, y, w, h):
        """右パネル: MOTOR NN（赤系）
        4列: 埋め込みFF(16) / 運動GRU(16) / 運動皮質FF(12) / OUTPUT(3)
        """
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((30, 5, 5, 220))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, (200, 60, 60), (x, y, w, h), 1, border_radius=4)

        title = self.font_s.render("MOTOR NN", True, (255, 100, 100))
        self.screen.blit(title, (x + 6, y + 4))

        embed_acts  = getattr(genome, 'last_embed_act',        [0.0] * 16)
        gru_acts    = getattr(genome, 'last_motor_gru',        [0.0] * 16)
        cortex_acts = getattr(genome, 'last_motor_cortex_act', [0.0] * 12)
        out_acts    = getattr(genome, 'last_output_act',       [0.5, 0.5, 0.0])
        out_labels  = ["Acc", "Str", "Brk"]

        node_r   = 4
        node_gap = 12
        row_top  = y + 18
        # 4列の水平位置
        col_embed  = x + 18
        col_gru    = x + int(w * 0.30)
        col_cortex = x + int(w * 0.54)
        col_out    = x + int(w * 0.76)
        bar_w      = w - (col_out - x) - 8

        def _bipolar_color(v):
            if v >= 0:
                t = min(1.0, v)
                return (int(40 + 215 * t), int(40 * (1-t)), 40)
            else:
                t = min(1.0, -v)
                return (40, int(40 * (1-t)), int(40 + 215 * t))

        def _draw_col(acts, cx, bipolar=True, border_c=(150,50,50)):
            n = len(acts)
            sy = row_top + max(0, (h - 28 - n * node_gap) // 2)
            for i, v in enumerate(acts):
                ny = sy + i * node_gap
                c = _bipolar_color(float(v))
                pygame.draw.circle(self.screen, c, (cx, ny), node_r)
                pygame.draw.circle(self.screen, border_c, (cx, ny), node_r, 1)
            return sy, n

        sy_embed,  n_embed  = _draw_col(embed_acts,  col_embed)
        sy_gru,    n_gru    = _draw_col(gru_acts,    col_gru)
        sy_cortex, n_cortex = _draw_col(cortex_acts, col_cortex)

        # OUTPUT列（バー + 数値）
        n_out = len(out_acts)
        sy_out = row_top + max(0, (h - 28 - n_out * 36) // 2)
        for i, (v, lbl) in enumerate(zip(out_acts, out_labels)):
            oy = sy_out + i * 36
            lt = self.font_s.render(lbl, True, (200, 100, 100))
            self.screen.blit(lt, (col_out - lt.get_width() - 2, oy + 2))
            pygame.draw.rect(self.screen, (50, 20, 20), (col_out, oy, bar_w, 14))
            pygame.draw.rect(self.screen, (220, 80, 80),
                             (col_out, oy, int(bar_w * max(0, min(1, v))), 14))
            vt = self.font_s.render(f"{v:.2f}", True, (200, 150, 150))
            self.screen.blit(vt, (col_out, oy + 16))

        # 層間エッジ（上位8本）
        motor_nn = getattr(genome, 'motor', None)
        if motor_nn is not None:
            self._draw_edges(embed_acts,  sy_embed,  col_embed,  node_r,
                             gru_acts,    sy_gru,    col_gru,    node_r,
                             getattr(motor_nn, 'Wh', None))
            self._draw_edges(gru_acts,    sy_gru,    col_gru,    node_r,
                             cortex_acts, sy_cortex, col_cortex, node_r,
                             getattr(motor_nn, 'W_cortex', None))
            self._draw_edges(cortex_acts, sy_cortex, col_cortex, node_r,
                             out_acts,    sy_out,    col_out,    node_r,
                             getattr(motor_nn, 'W_out', None),
                             node_gap_b=36)

        # 凡例・列ラベル
        leg = self.font_s.render("赤=+ 暗=0 青=-", True, (160, 80, 80))
        self.screen.blit(leg, (x + 4, y + h - 14))
        for lbl, lx, lc in [
            ("EMB",  col_embed,  (200, 80, 80)),
            ("GRU",  col_gru,    (210, 90, 90)),
            ("FF",   col_cortex, (220, 100, 100)),
            ("OUT",  col_out + bar_w // 2, (240, 120, 120)),
        ]:
            lt = self.font_s.render(lbl, True, lc)
            self.screen.blit(lt, (lx - lt.get_width() // 2, y + h - 14))

    def _draw_edges(self, acts_a, sy_a, cx_a, r_a,
                    acts_b, sy_b, cx_b, r_b,
                    W, max_draw: int = 8,
                    node_gap_a: int = 12, node_gap_b: int = 12):
        """2つの層間のエッジを描画する。
        重みの絶対値が大きい上位max_draw本のみ描画。
        正の重み: 緑 / 負の重み: 赤
        """
        if W is None:
            return
        try:
            import numpy as np
            W_np = W if hasattr(W, 'shape') else None
            if W_np is None:
                return
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

            for _, w, i, j in pairs[:max_draw]:
                iy = sy_a + i * node_gap_a
                jy = sy_b + j * node_gap_b
                alpha = min(180, int(abs(w) * 60))
                c = (20, alpha, 20) if w > 0 else (alpha, 20, 20)
                pygame.draw.line(self.screen, c,
                                 (cx_a + r_a, iy), (cx_b - r_b, jy), 1)
        except Exception:
            pass

    def _draw_flow_arrows(self, x, y, panel_w, panel_h, gap, mode):
        """パネル間の接続矢印。"""
        import time
        flash = (int(time.time() * 5) % 2 == 0)   # 点滅

        mid_y = y + panel_h // 2
        ax1 = x + panel_w
        ax2 = ax1 + gap
        ax3 = ax2 + panel_w
        ax4 = ax3 + gap

        # 左矢印（S→M: 傾聴ターン中は青く点滅）
        c1 = (100, 160, 255) if (mode == 'listen' and flash) else (50, 60, 80)
        pygame.draw.line(self.screen, c1, (ax1, mid_y), (ax2, mid_y), 2)
        pygame.draw.polygon(self.screen, c1, [
            (ax2, mid_y), (ax2 - 5, mid_y - 4), (ax2 - 5, mid_y + 4)])

        # 右矢印（M→S: 発話ターン中は赤く点滅）
        c2 = (100, 200, 120) if (mode == 'speak' and flash) else (50, 80, 50)
        pygame.draw.line(self.screen, c2, (ax3, mid_y), (ax4, mid_y), 2)
        pygame.draw.polygon(self.screen, c2, [
            (ax4, mid_y), (ax4 - 5, mid_y - 4), (ax4 - 5, mid_y + 4)])

    # ================================================================
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

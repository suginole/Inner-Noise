"""
profile_game.py — ゲームのボトルネックを特定するプロファイリングスクリプト
各処理を個別に計測し、フレームあたりの時間を出力する。
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import time
import pygame
pygame.init()
pygame.font.init()

from config import *
from game.field import Field
from game.car import Car
from game.ga_agent import GAAgent, GAGenome
from game.renderer import Renderer
from game.bottleneck import Bottleneck

# ---- セットアップ ----
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
renderer = Renderer(screen)
field = Field(terrain_seed=42, food_episode=0)
car = Car(*field.start_pos)
genome = GAGenome()
agent = GAAgent(car, field, genome)

N = 200   # 計測フレーム数

def measure(label, fn, n=N):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - t0) * 1000
    ms_per = elapsed / n
    fps_eq = 1000 / ms_per if ms_per > 0 else 9999
    print(f"  {label:<45} {ms_per:6.3f} ms/f  (~{fps_eq:5.0f} fps)")
    return ms_per

print(f"=== フレームごとの処理時間プロファイル ({N}フレーム平均) ===\n")

# ---- 地形描画 ----
cam = renderer.calc_camera(car.pos)
measure("draw_field (terrain blit only)",
    lambda: (screen.fill(C_BG), field.get_surface(), screen.blit(field.get_surface(), (-cam.x, -cam.y))))

# ---- 餌描画 ----
measure("draw_field (foods only)",
    lambda: [pygame.draw.circle(screen, (255,220,50), (int(fp.x - cam.x), int(fp.y - cam.y)), 6)
             for fp, _ in field.foods
             if -20 < int(fp.x - cam.x) < SCREEN_W + 20])

# ---- 視野コーン ----
measure("draw_vision_cone",
    lambda: renderer.draw_vision_cone(car, cam))

# ---- ミニマップ ----
measure("draw_minimap",
    lambda: renderer.draw_minimap(field, car.pos, field.goal_pos))

# ---- HUD ----
bn = Bottleneck()
measure("draw_hud_player",
    lambda: renderer.draw_hud_player(car, bn))

# ---- アクティベーションパネル ----
measure("draw_player_obs_panel",
    lambda: renderer.draw_player_obs_panel(car.get_observation(field), SCREEN_W-430, SCREEN_H-210))

# ---- get_observation (視野レイ含む) ----
measure("car.get_observation (with vision rays)",
    lambda: car.get_observation(field))

# ---- car.step (物理) ----
measure("car.step (physics)",
    lambda: car.step(0.5, 0.5, 0.0, field))

# ---- bottleneck.tick ----
measure("bottleneck.tick",
    lambda: (bn.push(car.get_observation(field)), bn.tick(1/60)))

# ---- draw_activation_panel ----
genome.forward(car.get_observation(field))
measure("draw_activation_panel",
    lambda: renderer.draw_activation_panel(genome, SCREEN_W-430, SCREEN_H-210))

# ---- pygame.display.flip ----
measure("pygame.display.flip",
    lambda: pygame.display.flip())

# ---- フルフレーム（プレイヤーモード想定） ----
print()
def full_player_frame():
    screen.fill(C_BG)
    cam2 = renderer.calc_camera(car.pos)
    renderer.draw_field(field, cam2)
    renderer.draw_vision_cone(car, cam2)
    car.draw(screen, cam2)
    renderer.draw_minimap(field, car.pos, field.goal_pos)
    renderer.draw_hud_player(car, bn)
    renderer.draw_player_obs_panel(car.get_observation(field), SCREEN_W-430, SCREEN_H-210)
    pygame.display.flip()

total = measure("=== FULL PLAYER FRAME ===", full_player_frame)
print(f"\n  理論FPS上限: {1000/total:.0f}  (目標: {FPS})")

# ---- フルフレーム（GAモード想定） ----
def full_ga_frame():
    screen.fill(C_BG)
    cam2 = renderer.calc_camera(car.pos)
    renderer.draw_field(field, cam2)
    renderer.draw_vision_cone(car, cam2)
    car.draw(screen, cam2)
    renderer.draw_minimap(field, car.pos, field.goal_pos)
    renderer.draw_hud_ga(car, type('GA', (), {
        'get_stats': lambda s: {'generation':1,'best':0,'avg':0,'worst':0},
        'pop_size': 30,
        'best_fitness_history': [0,1,2],
        'avg_fitness_history': [0,0.5,1],
    })(), bottleneck=bn, agent_idx=0, total=30)
    renderer.draw_activation_panel(genome, SCREEN_W-430, SCREEN_H-210)
    renderer.draw_fitness_graph(type('GA', (), {
        'best_fitness_history': list(range(50)),
        'avg_fitness_history': list(range(50)),
    })(), 20, 110)
    pygame.display.flip()

total_ga = measure("=== FULL GA FRAME ===", full_ga_frame)
print(f"\n  理論FPS上限: {1000/total_ga:.0f}  (目標: {FPS})")

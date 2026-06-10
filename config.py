"""
config.py — Sage-Brute 設定・定数
=============================================================
SAGE（賢者）: 弁別視野でキノコ12種を識別・パルスを送信
BRUTE（野人）: 視覚レイ+バイオーム+腐敗を感知・行動を出力
"""

# ---- ウィンドウ ----
SCREEN_W = 1280
SCREEN_H = 720
FPS      = 60
TITLE    = "Inner Noise — Sage & Brute"

# ---- ワールド ----
FIELD_SIZE     = 4000
WORLD_W        = FIELD_SIZE
WORLD_H        = FIELD_SIZE
TILE           = 8
TERRAIN_SEED   = 42

# ---- 地形生成 ----
TERRAIN_OCTAVES    = 6
TERRAIN_SCALE      = 0.0015
BIOME_THRESHOLDS   = (0.33, 0.66)   # 沼/平地/山の境界
MOUNTAIN_THRESHOLD = BIOME_THRESHOLDS[1]
VALLEY_THRESHOLD   = BIOME_THRESHOLDS[0]
PASS_WIDTH         = 200

# ---- 車体 ----
CAR_MAX_SPEED      = 4.0
CAR_ACCEL          = 0.18
CAR_BRAKE          = 0.25
CAR_FRICTION       = 0.08
CAR_TURN_SPEED     = 2.8
CAR_SLOPE_DRAG     = 0.6
CAR_FALL_DAMAGE    = 0.20
SLOPE_DAMAGE_THRESH= 0.45

# ---- エネルギー ----
MAX_ENERGY         = 100.0
INIT_ENERGY        = 100.0
ENERGY_DECAY       = 100.0 / 3000   # 3000fで死亡（約50秒）
ENERGY_NORMAL      = 8.0            # 普通種・新鮮
ENERGY_PREMIUM     = 20.0           # 栄養価高・新鮮
ENERGY_ROTTEN      = -40.0          # 腐敗（種類問わず）
ENERGY_TOXIC       = -30.0          # 中毒発症
ENERGY_GOAL        = 15.0           # ゴール到達
IDLE_SPEED_THRESH  = 0.3

# 旧定数との互換性エイリアス
ENERGY_MAX         = MAX_ENERGY
ENERGY_INIT        = INIT_ENERGY
ENERGY_DECAY_BASE  = ENERGY_DECAY
ENERGY_DECAY_CLIMB = 0.07
ENERGY_DECAY_IDLE  = 0.003
ENERGY_PER_FOOD    = ENERGY_NORMAL
ENERGY_PER_FOOD_HI = ENERGY_PREMIUM

# ---- 視野 ----
VISION_ANGLE_DEG   = 45.0
VISION_RANGE       = WORLD_W / 10
VISION_RAYS        = 5
FOCUS_RANGE        = 400.0

# ---- キノコ ----
MUSHROOM_DENSITY   = 0.015          # 高密度（旧値の約倍）
ROT_PROBABILITY    = 0.3            # 腐敗確率
TOXIC_COUNT        = 3              # 同一種連続摂取で中毒
HISTORY_LEN        = 5              # 摂取履歴保持数
MUSHROOM_RADIUS    = 12             # 当たり判定半径 (px)
FOOD_RADIUS        = MUSHROOM_RADIUS

# キノコ種類定義: (biome, grade, variant) → エネルギー値
MUSHROOM_SPECIES = {
    ('W', 'normal',  1): ENERGY_NORMAL,
    ('W', 'normal',  2): ENERGY_NORMAL,
    ('W', 'premium', 1): ENERGY_PREMIUM,
    ('W', 'premium', 2): ENERGY_PREMIUM,
    ('G', 'normal',  1): ENERGY_NORMAL,
    ('G', 'normal',  2): ENERGY_NORMAL,
    ('G', 'premium', 1): ENERGY_PREMIUM,
    ('G', 'premium', 2): ENERGY_PREMIUM,
    ('M', 'normal',  1): ENERGY_NORMAL,
    ('M', 'normal',  2): ENERGY_NORMAL,
    ('M', 'premium', 1): ENERGY_PREMIUM,
    ('M', 'premium', 2): ENERGY_PREMIUM,
}
MUSHROOM_SPECIES_LIST = list(MUSHROOM_SPECIES.keys())  # インデックス用
NUM_SPECIES = len(MUSHROOM_SPECIES_LIST)               # = 12

# キノコグリッド配置
FOOD_GRID_SPACING  = 200
FOOD_JITTER        = 80
FOOD_SEED          = 12345
FOOD_COUNT         = 200

# ---- ゴール ----
GOAL_RADIUS        = 60

# ---- 通信バス（一元管理） ----
PULSE_BITS            = 2
PULSE_GEN_INTERVAL    = 6             # 10Hz相当
PULSE_CONSUME_RATE    = 6             # 生成と同周期（リアルタイム同期）
TURN_FRAMES           = 480           # 8秒ターン
PULSE_TOTAL           = 80            # 80パルス/ターン

# 旧定数との互換性エイリアス
BN_HZ              = FPS / PULSE_GEN_INTERVAL
BN_TURN_SEC        = TURN_FRAMES / FPS
BN_PARAMS          = PULSE_BITS
BN_PULSES_PER_TURN = PULSE_TOTAL
PIPELINE_OFFSET    = PULSE_GEN_INTERVAL

# ---- GA ----
GA_POP_SIZE        = 50
GA_ELITE           = 4
ELITE_SIZE         = GA_ELITE
GA_MUTATION_RATE   = 0.1
GA_MUTATION_STD    = 0.3
MUTATE_RATE_INIT   = GA_MUTATION_RATE
MUTATE_STD_INIT    = GA_MUTATION_STD

# ---- SAGE NN次元 ----
MUSHROOM_ENC_DIM   = 6    # キノコ構造化エンコード次元
SAGE_OBS_DIM       = 11   # 弁別視野6 + ゴール角度1 + ゴール距離1 + エネルギー1 + 受信パルス2
SAGE_L3_OUT        = 24
SAGE_L3_NORMAL     = 12
SAGE_L3_BUF        = 12
SAGE_BUF_DIM       = 5
SAGE_MEM_DIM       = 12
SAGE_MEM_INHERIT   = 6
SAGE_BYPASS_OUT    = 16
SAGE_L1_IN         = SAGE_MEM_DIM + SAGE_BYPASS_OUT   # = 28
SAGE_L1_OUT        = 24
SAGE_ENCODE_DIM    = 2

# ---- BRUTE NN次元 ----
BRUTE_OBS_DIM      = 11   # 弁別視野6 + 視覚レイ5本 + 受信パルス2
BRUTE_L3_OUT       = 24
BRUTE_L3_NORMAL    = 12
BRUTE_L3_BUF       = 12
BRUTE_BUF_DIM      = 5
BRUTE_MEM_DIM      = 12
BRUTE_MEM_INHERIT  = 6
BRUTE_BYPASS_OUT   = 16
BRUTE_L1_IN        = BRUTE_MEM_DIM + BRUTE_BYPASS_OUT  # = 28
BRUTE_L1_OUT       = 24
BRUTE_ACTION_DIM   = 3
BRUTE_ENCODE_DIM   = 2

# ---- 報酬 ----
REWARD_GOAL        = ENERGY_GOAL
REWARD_FOOD        = ENERGY_NORMAL
REWARD_FOOD_HI     = ENERGY_PREMIUM
REWARD_GOAL_STEP   = 0.01
REWARD_MOVE        = 0.003
REWARD_SURVIVE     = 0.002
PENALTY_DEATH      = -500.0
PENALTY_FALL       = -10.0

# ---- 色定数 ----
C_BG           = ( 10,  12,  20)
C_WHITE        = (255, 255, 255)
C_GRAY         = (100, 100, 120)
C_CAR          = (200, 200, 255)
C_GOAL         = ( 50, 220, 150)
C_ENERGY_LO    = (220,  80,  80)
C_PULSE_ON     = (255, 200,  50)
C_PULSE_OFF    = ( 50,  50,  60)
C_FOOD         = (255, 220,  50)
C_FOOD_HI      = (255, 140,  20)
C_FOOD_ROT     = ( 80,  60,  20)

# バイオーム色（グラデーションなし・明確分割）
C_VALLEY       = (135, 206, 235)   # 沼: 水色
C_PLAIN        = (245, 240, 220)   # 平地: ミルク色
C_MOUNTAIN     = ( 34,  85,  34)   # 山: 深緑

# バイオーム別色ディクショナリ
BIOME_COLORS = {
    'W': (135, 206, 235),
    'G': (245, 240, 220),
    'M': ( 34,  85,  34),
}

# パルス配色（方向別）
PULSE_COLOR_S_TO_B = (255, 180,  50)   # 暖色・オレンジ系（SAGEが送信）
PULSE_COLOR_B_TO_S = ( 80, 180, 255)   # 寒色・青系（BRUTEが送信）
PULSE_COLOR_OFF    = ( 60,  60,  60)   # 消灯

# PHONEME TABLE（音素変換用）
PHONEME_TABLE = {0: 'う', 1: 'い', 2: 'お', 3: 'あ'}
PHONEME_VOWEL = PHONEME_TABLE

# 音声出力
AUDIO_SAMPLE_RATE   = 22050
AUDIO_FRAME_MS      = 100    # ms（10Hzパルスに同期）
AUDIO_FRAME_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_FRAME_MS / 1000)  # = 2205
PHONEME_FORMANTS = {
    'a': (400,  600),
    'i': (150, 1150),
    'u': (150,  400),
    'o': (250,  400),
}
PITCH_FACTOR_HIGH = 0.7    # S→B（SAGEが送信）高ピッチ
PITCH_FACTOR_LOW  = 0.35   # B→S（BRUTEが送信）低ピッチ

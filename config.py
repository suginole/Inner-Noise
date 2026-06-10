# =============================================================
# Inner Noise — Sage & Brute  設定・定数
# （audio-phoneme既存定数を維持して SAGE/BRUTE定数を追加）
# =============================================================

# ---- ウィンドウ ----
SCREEN_W = 1280
SCREEN_H = 720
FPS      = 60
TITLE    = "Inner Noise — Sage & Brute"

# ---- ワールド ----
WORLD_W  = 4000
WORLD_H  = 4000
TILE     = 8

# ---- 地形生成 ----
TERRAIN_OCTAVES    = 6
TERRAIN_SCALE      = 0.0005   # 旧値(0.0015)の1/3：バイオームをフィールド内に数個程度に絞る
MOUNTAIN_THRESHOLD = 0.55
VALLEY_THRESHOLD   = 0.35
PEAK_HEIGHT        = 1.0
PASS_WIDTH         = 120

# ---- 車の物理 ----
CAR_MAX_SPEED      = 4.0
CAR_ACCEL          = 0.18
CAR_BRAKE          = 0.25
CAR_FRICTION       = 0.08
CAR_TURN_SPEED     = 2.8
CAR_SLOPE_DRAG     = 0.6
CAR_FALL_DAMAGE    = 0.20
SLOPE_DAMAGE_THRESH= 0.45

# ---- エネルギー（旧・audio-phoneme互換） ----
ENERGY_MAX         = 0.5
ENERGY_INIT        = 0.5
ENERGY_DECAY_BASE  = 0.00150
ENERGY_DECAY_CLIMB = 0.07000
ENERGY_DECAY_IDLE  = 0.00300
ENERGY_PER_FOOD    = 0.25
ENERGY_PER_FOOD_HI = 0.55
IDLE_SPEED_THRESH  = 0.3

# ---- 視野 ----
VISION_ANGLE_DEG   = 45.0
VISION_RANGE       = WORLD_W / 10
VISION_RAYS        = 5
FOCUS_RANGE        = 400.0

# ---- 餌（旧・audio-phoneme互換） ----
FOOD_COUNT         = 340      # 旧値(200)の1.7倍
FOOD_RADIUS        = 12
FOOD_GRID_SPACING  = 200
FOOD_JITTER        = 80
FOOD_MOUNTAIN_THRESH = MOUNTAIN_THRESHOLD + 0.1
FOOD_SEED          = 12345

# ---- ゴール ----
GOAL_RADIUS        = 60

# ---- ボトルネック通信路（旧・audio-phoneme互換） ----
BN_PARAMS          = 2
BN_HZ              = 10
BN_TURN_SEC        = 2
BN_PULSES_PER_TURN = int(BN_HZ * BN_TURN_SEC)
TURN_FRAMES        = int(BN_TURN_SEC * FPS)
PULSE_TOTAL        = BN_PULSES_PER_TURN
PULSE_GEN_INTERVAL = FPS // BN_HZ
PULSE_CONSUME_RATE = PULSE_GEN_INTERVAL
PIPELINE_OFFSET    = PULSE_GEN_INTERVAL

# ---- RNNアーキテクチャ（旧・audio-phoneme互換） ----
SENSORY_INPUT_DIM  = 6 + VISION_RAYS + 1
L3_OUT_DIM         = 24
L3_NORMAL_DIM      = 12
L3_BUFFER_DIM      = 12
BUF_GRU_DIM        = 5
MEM_GRU_DIM        = 12
GRU_INHERIT_DIM    = 6
GRU_EPISODE_DIM    = 6
BYPASS_FF_DIM      = 16
L1_IN_DIM          = MEM_GRU_DIM + BYPASS_FF_DIM
L1_OUT_DIM         = 24
SENSORY_ENCODE_DIM = 2
MOTOR_OUTPUT_DIM   = 3
SENSORY_FF_DIM     = L3_OUT_DIM
SENSORY_GRU_DIM    = MEM_GRU_DIM
SENSORY_INTEG_DIM  = L1_OUT_DIM
SENSORY_CORTEX_DIM = L1_OUT_DIM
MOTOR_EMBED_DIM    = L3_OUT_DIM
MOTOR_GRU_DIM      = MEM_GRU_DIM
MOTOR_INTEG_DIM    = L1_OUT_DIM
MOTOR_CORTEX_DIM   = L1_OUT_DIM

# ---- 報酬（旧・audio-phoneme互換） ----
REWARD_GOAL        = 1000.0
REWARD_FOOD        = 5.0
REWARD_FOOD_HI     = 20.0
REWARD_GOAL_STEP   = 0.01
REWARD_MOVE        = 0.003
REWARD_SURVIVE     = 0.002
PENALTY_DEATH      = -500.0
PENALTY_FALL       = -10.0
COMPLEXITY_ALPHA   = 0.1

# ---- GA（旧・audio-phoneme互換） ----
GA_POP_SIZE        = 50
GA_ELITE           = 20
GA_MUTATION_RATE   = 0.15
GA_MUTATION_STD    = 0.3
GA_EPISODE_FRAMES  = 0

# ---- カラーパレット（旧・audio-phoneme互換） ----
C_BG           = (10,  12,  20)
C_MOUNTAIN     = (80,  70,  60)
C_VALLEY       = (30,  55,  35)
C_PLAIN        = (50,  75,  45)
C_ROAD         = (90,  85,  75)
C_FOOD         = (255, 220,  50)
C_GOAL         = (50,  220, 150)
C_CAR          = (220,  80,  60)
C_CAR_GA       = (80,  160, 220)
C_HUD_BG       = (10,  12,  20, 180)
C_HUD_TEXT     = (220, 220, 220)
C_PULSE_ON     = (255, 200,  50)
C_PULSE_OFF    = (50,   50,  60)
C_ENERGY_HI    = (80,  220, 100)
C_ENERGY_LO    = (220,  80,  60)
C_WHITE        = (255, 255, 255)
C_GRAY         = (120, 120, 120)
C_DARK         = (30,   30,  40)

# ---- 音素テーブル（audio-phoneme維持） ----
PHONEME_VOWEL = {
    0b00: 'u',
    0b01: 'i',
    0b10: 'o',
    0b11: 'a',
}
PHONEME_FORMANTS = {
    'a': (400,   600),
    'i': (150,  1150),
    'u': (150,   400),
    'o': (250,   400),
}
PHONEME_TABLE = {
    0b00: 'う',
    0b01: 'い',
    0b10: 'お',
    0b11: 'あ',
}
AUDIO_SAMPLE_RATE   = 22050
AUDIO_FRAME_MS      = 200
AUDIO_FRAME_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_FRAME_MS / 1000)
PITCH_FACTOR_HIGH   = 0.7
PITCH_FACTOR_LOW    = 0.35
VOWEL_F2F1_I        = 6.0
VOWEL_F1_A          = 650
VOWEL_F2_U          = 1200

# =============================================================
# ---- SAGE-BRUTE 追加定数 ----
# =============================================================

# ワールド（SAGE-BRUTE用エイリアス）
FIELD_SIZE   = 4000
TERRAIN_SEED = 42
BIOME_THRESHOLDS = (0.33, 0.66)

# エネルギー（SAGE-BRUTE用）
MAX_ENERGY   = 100.0
INIT_ENERGY  = 100.0
ENERGY_DECAY = 100.0 / 3000
ENERGY_NORMAL  = 8.0
ENERGY_PREMIUM = 20.0
ENERGY_ROTTEN  = -40.0
ENERGY_TOXIC   = -30.0
ENERGY_GOAL    = 15.0

# キノコ
MUSHROOM_DENSITY  = 0.0255   # 旧値(0.015)の1.7倍
ROT_PROBABILITY   = 0.3
TOXIC_COUNT       = 3
HISTORY_LEN       = 5
MUSHROOM_RADIUS   = 12

MUSHROOM_SPECIES = {
    ('W','normal', 1): 8.0,
    ('W','normal', 2): 8.0,
    ('W','premium',1): 20.0,
    ('W','premium',2): 20.0,
    ('G','normal', 1): 8.0,
    ('G','normal', 2): 8.0,
    ('G','premium',1): 20.0,
    ('G','premium',2): 20.0,
    ('M','normal', 1): 8.0,
    ('M','normal', 2): 8.0,
    ('M','premium',1): 20.0,
    ('M','premium',2): 20.0,
}
MUSHROOM_SPECIES_LIST = list(MUSHROOM_SPECIES.keys())
NUM_SPECIES = len(MUSHROOM_SPECIES_LIST)  # = 12

# 通信バス（SAGE-BRUTE用オーバーライド）
PULSE_BITS         = 2
# PULSE_GEN_INTERVAL, PULSE_CONSUME_RATE, TURN_FRAMES, PULSE_TOTAL は旧定数を流用

# GA（SAGE-BRUTE用）
ELITE_SIZE       = 4
MUTATE_RATE_INIT = 0.1
MUTATE_STD_INIT  = 0.3

# SAGE NN次元
MUSHROOM_ENC_DIM = 6
SAGE_OBS_DIM     = 11
SAGE_L3_OUT      = 24
SAGE_L3_NORMAL   = 12
SAGE_L3_BUF      = 12
SAGE_BUF_DIM     = 5
SAGE_MEM_DIM     = 12
SAGE_MEM_INHERIT = 6
SAGE_BYPASS_OUT  = 16
SAGE_L1_IN       = 28
SAGE_L1_OUT      = 24
SAGE_ENCODE_DIM  = 2

# BRUTE NN次元
BRUTE_OBS_DIM     = 11
BRUTE_L3_OUT      = 24
BRUTE_L3_NORMAL   = 12
BRUTE_L3_BUF      = 12
BRUTE_BUF_DIM     = 5
BRUTE_MEM_DIM     = 12
BRUTE_MEM_INHERIT = 6
BRUTE_BYPASS_OUT  = 16
BRUTE_L1_IN       = 28
BRUTE_L1_OUT      = 24
BRUTE_ACTION_DIM  = 3
BRUTE_ENCODE_DIM  = 2

# 報酬（SAGE-BRUTE用）
REWARD_GOAL_SB   = 15.0
REWARD_FOOD_SB   = 8.0
REWARD_FOOD_HI_SB = 20.0

# バイオーム色
BIOME_COLORS = {
    'W': (135, 206, 235),
    'G': (245, 240, 220),
    'M': ( 34,  85,  34),
}

# パルス配色
PULSE_COLOR_S_TO_B = (255, 180,  50)
PULSE_COLOR_B_TO_S = ( 80, 180, 255)
PULSE_COLOR_OFF    = ( 60,  60,  60)

# 色定数（SAGE-BRUTE用追加）
C_FOOD_ROT  = ( 80,  60,  20)
C_FOOD_HI   = (255, 140,  20)

# GPU高速化
COMPUTE_MODE    = "auto"  # "auto" | "numpy" | "torch_gpu"
GA_POP_SIZE_GPU = 3000    # モード3でのpop_size
GA_POP_SIZE_CPU = 50      # モード2でのpop_size

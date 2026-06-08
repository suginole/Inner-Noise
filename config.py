# =============================================================
# Blind Driving Survival — 設定・定数
# =============================================================

# ---- ウィンドウ ----
SCREEN_W = 1280
SCREEN_H = 720
FPS      = 60
TITLE    = "Blind Driving Survival"

# ---- ワールド ----
WORLD_W  = 4000   # ワールド全体の幅 (px)
WORLD_H  = 4000   # ワールド全体の高さ (px)
TILE     = 8      # 地形サンプリング解像度 (px)

# ---- 地形生成 ----
TERRAIN_OCTAVES    = 6
TERRAIN_SCALE      = 0.0015   # パーリンノイズのスケール
MOUNTAIN_THRESHOLD = 0.55     # この値以上を「山」とみなす
VALLEY_THRESHOLD   = 0.35     # この値以下を「谷」とみなす
PEAK_HEIGHT        = 1.0      # 最大標高（正規化）
PASS_WIDTH         = 120      # 峠（スリット）の幅 (px)

# ---- 車の物理 ----
CAR_MAX_SPEED      = 4.0      # px/frame
CAR_ACCEL          = 0.18
CAR_BRAKE          = 0.25
CAR_FRICTION       = 0.08
CAR_TURN_SPEED     = 2.8      # deg/frame
CAR_SLOPE_DRAG     = 0.6      # 登坂時の追加摩擦係数
CAR_FALL_DAMAGE    = 0.20     # 急勾配落下時のエネルギーダメージ
SLOPE_DAMAGE_THRESH= 0.45     # この勾配を超えると落下ダメージ

# ---- エネルギー（空腹） ----
ENERGY_MAX         = 1.0
ENERGY_DECAY_BASE  = 0.00150  # 毎フレームの基礎消費（×10）
ENERGY_DECAY_CLIMB = 0.00600  # 登坂中の追加消費（×10）
ENERGY_DECAY_IDLE  = 0.00300  # 停滞中の追加消費（×10）
ENERGY_PER_FOOD    = 0.25     # 餌1個で回復するエネルギー
IDLE_SPEED_THRESH  = 0.3      # この速度以下を「停滞」とみなす

# ---- 視野（感覚器官：餌探索） ----
VISION_ANGLE_DEG   = 45.0     # 視野角（片側）
VISION_RANGE       = WORLD_W / 10  # 視野距離 = マップ幅の1/10 = 400px
VISION_RAYS        = 5        # 視野内のレイ数（観測ベクトルの次元に影響）

# ---- 餌 ----
FOOD_COUNT         = 80       # フィールド上の餌の総数
FOOD_RADIUS        = 12       # 餌の当たり判定半径 (px)
FOOD_VALLEY_BIAS   = 0.85     # 谷に出現する確率

# ---- ゴール ----
GOAL_RADIUS        = 60       # ゴール判定半径 (px)

# ---- ボトルネック通信路 ----
BN_PARAMS          = 4        # パラメータ数（4 bits）
BN_HZ              = 5        # パルス周波数
BN_TURN_SEC        = 4        # 傾聴/発話ターン長 (秒)
BN_PULSES_PER_TURN = BN_HZ * BN_TURN_SEC   # = 20

# ---- 報酬 ----
REWARD_GOAL        = 1000.0
REWARD_FOOD        = 5.0
REWARD_GOAL_STEP   = 0.01     # ゴールに近づくごとの微小報酬
PENALTY_DEATH      = -500.0
PENALTY_FALL       = -10.0
COMPLEXITY_ALPHA   = 0.1      # NEATノード/エッジペナルティ係数

# ---- GA ----
GA_POP_SIZE        = 30       # 個体数
GA_GENOME_DIM      = 16       # シンプルGAのゲノム次元（重みベクトル）
GA_ELITE           = 4        # エリート個体数
GA_MUTATION_RATE   = 0.15
GA_MUTATION_STD    = 0.3
GA_EPISODE_FRAMES  = 60 * 30  # 1エピソードの最大フレーム数 (30秒)

# ---- カラーパレット ----
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

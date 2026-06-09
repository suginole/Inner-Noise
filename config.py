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
ENERGY_DECAY_BASE  = 0.00150  # 毎フレームの基礎消費
ENERGY_DECAY_CLIMB = 0.07000  # 登坂中の追加消費（勾配強度×この値）→ 旧値の1.75倍
ENERGY_DECAY_IDLE  = 0.00300  # 停滞中の追加消費
ENERGY_PER_FOOD    = 0.25     # 通常餌1個で回復するエネルギー
ENERGY_PER_FOOD_HI = 0.55     # 高級餌（山の上）の回復量
IDLE_SPEED_THRESH  = 0.3      # この速度以下を「停滞」とみなす

# ---- 視野（感覚器官：餌探索） ----
VISION_ANGLE_DEG   = 45.0     # 視野角（片側）
VISION_RANGE       = WORLD_W / 10  # 視野距離 = マップ幅の1/10 = 400px
VISION_RAYS        = 5        # 視野内のレイ数（観測ベクトルの次元に影響）
FOCUS_RANGE        = 400.0    # 弁別視野（視線中央線上）の距離 (px)

# ---- 餌 ----
FOOD_COUNT         = 80       # フィールド上の餌の総数
FOOD_RADIUS        = 12       # 餌の当たり判定半径 (px)
FOOD_VALLEY_BIAS   = 0.85     # 谷に出現する確率
FOOD_MOUNTAIN_THRESH = MOUNTAIN_THRESHOLD + 0.1  # この高さ以上に高級餌が出現
FOOD_SEED          = 12345    # 餌配置用の固定シード（地形と分離）
PASS_REWARD_RADIUS = 300      # 峰出口報酬エリアの半径 (px)
PASS_REWARD_COUNT  = 8        # 峰出口片側に配置する超高級餌の数

# ---- ゴール ----
GOAL_RADIUS        = 60       # ゴール判定半径 (px)

# ---- ボトルネック通信路 ----
BN_PARAMS          = 2        # パラメータ数（2 bits = 母音のみ）
BN_HZ              = 5        # パルス周波数
BN_TURN_SEC        = 4        # 傘聴/発話ターン長 (秒)
BN_PULSES_PER_TURN = BN_HZ * BN_TURN_SEC   # = 20

# ---- パイプライン型半双方向通信 ----
TURN_FRAMES        = 240      # 4秒（60fps基準）
PULSE_TOTAL        = 20       # 1ターンのパルス数
PULSE_CONSUME_RATE = 24       # 半速消化（24フレームごとに1パルス消化）
PIPELINE_OFFSET    = 120      # 半ターンずれ（10パルス分）

# ---- RNNボトルネックアーキテクチャ（継承・非継承分割設計） ----
SENSORY_INPUT_DIM  = 6 + VISION_RAYS + 1  # = 12（obsベクトル全次元）
SENSORY_FF_DIM     = 16       # 入力FFの出力次元
SENSORY_GRU_DIM    = 16       # 感覚 GRUの隠れ次元
SENSORY_INTEG_DIM  = 16       # 統合FFの出力次元
SENSORY_CORTEX_DIM = 16       # 互換性維持用（= SENSORY_INTEG_DIM）
GRU_INHERIT_DIM    = 8        # GRU隠れ状態の継承領域（8次元）
GRU_EPISODE_DIM    = 8        # GRU隠れ状態の非継承領域（8次元）
MOTOR_EMBED_DIM    = 16       # パルス埋め込みFFの出力次元
MOTOR_GRU_DIM      = 16       # 運動 GRUの隠れ次元
MOTOR_INTEG_DIM    = 16       # 統合FFの出力次元
MOTOR_CORTEX_DIM   = 12       # 運動皮質FFの出力次元
MOTOR_OUTPUT_DIM   = 3        # 出力FF（Accel/Steer/Brake）

# ---- 報酬 ----
REWARD_GOAL        = 1000.0
REWARD_FOOD        = 5.0
REWARD_FOOD_HI     = 20.0     # 高級餌（山の上）の報酬
REWARD_GOAL_STEP   = 0.01     # ゴールに近づくごとの微小報酬
REWARD_MOVE        = 0.003    # 毎フレームの前進報酬（速度に比例）
REWARD_SURVIVE     = 0.002    # 毎フレームの生存報酬
PENALTY_DEATH      = -500.0
PENALTY_FALL       = -10.0
COMPLEXITY_ALPHA   = 0.1      # NEATノード/エッジペナルティ係数

# ---- GA ----
GA_POP_SIZE        = 50       # 個体数
GA_ELITE           = 20       # エリート個体数
GA_MUTATION_RATE   = 0.15
GA_MUTATION_STD    = 0.3
GA_EPISODE_FRAMES  = 0        # 0 = 時間制限なし（全員餐死まで）

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

# ---- 音素テーブル（2bits = 母音のみ） ----
# 子音は使用しない。将来の変更はここを編集するだけでよい。

PHONEME_VOWEL = {
    0b00: 'u',   # う（F1低・F2低）
    0b01: 'i',   # い（F1低・F2高）
    0b10: 'o',   # お（F1中・F2低）
    0b11: 'a',   # あ（F1高・F2中）
}

# 母音フォルマント周波数 (Hz)
# 1オクターブ下げ実装（ピッチを人の耳に聴き取りやすい帯域に調整）
PHONEME_FORMANTS = {
    'a': (400,   600),   # あ
    'i': (150,  1150),   # い
    'u': (150,   400),   # う
    'o': (250,   400),   # お
}

# 2bits音素テーブル（表示用）
PHONEME_TABLE = {
    0b00: 'う',
    0b01: 'い',
    0b10: 'お',
    0b11: 'あ',
}

# 音声合成パラメータ
AUDIO_SAMPLE_RATE   = 22050   # Hz
AUDIO_FRAME_MS      = 200     # ms（5Hzパルスに同期）
AUDIO_FRAME_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_FRAME_MS / 1000)  # = 4410

# S↔M方向別ピッチ係数
# ベースピッチからのオクターブ調整：
#   1オクターブ上 = ×2.0
#   半オクターブ上 = ×√2 ≈ ×1.414
#   現在のベース（男性）: 0.35  → 1オクターブ上 = 0.35 × 2.0 = 0.70
#   現在のベース（女性）: 0.70  → 半オクターブ上 = 0.70 × 1.414 ≈ 0.99
PITCH_FACTOR_HIGH   = 0.99   # S→M（感覚→運動 / 傘聴ターン）← 女性：旧値×√2
PITCH_FACTOR_LOW    = 0.70   # M→S（運動→感覚 / 発話ターン）← 男性：旧値×2.0

# F1/F2ベース母音分類閾値
# 学術データ（Mokhtari & Tanaka 2000、日本語母音コーパス）に基づく
#
# 実測平均値（男性成人）:
#   い: F1=283Hz  F2=2353Hz  F2/F1=8.3
#   あ: F1=801Hz  F2=1159Hz  F2/F1=1.4
#   お: F1=503Hz  F2= 811Hz  F2/F1=1.6
#   う: F1=405Hz  F2=1550Hz  F2/F1=3.8
#
# 判定ツリー:
#   F2/F1 > 6.0  → い  (実測8.3、次点のう3.8と明確に分離)
#   F1 > 650Hz   → あ  (実測801Hz、次点のお503Hzと150Hz以上の差)
#   F2 > 1200Hz  → う  (実測1550Hz、お811Hzと700Hz以上の差)
#   else         → お
VOWEL_F2F1_I        = 6.0     # い の判定閾値（F2/F1比）
VOWEL_F1_A          = 650     # あ の判定閾値（F1 Hz）
VOWEL_F2_U          = 1200    # う の判定閾値（F2 Hz）← 旧値900Hzは誤り

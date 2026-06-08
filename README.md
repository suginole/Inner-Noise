# Blind Driving Survival

**RNN × GA（NEAT）ボトルネック通信アーキテクチャの実験シミュレーションゲーム**

エージェント（車）の「感覚系」と「運動系」が、5Hz・4bitsの極狭ボトルネック通信路のみで接続される。  
時系列パルスパターンによる「言語構造の創発（知性の共有）」を観察するための実験環境。

## 実行方法

```bash
pip install pygame numpy scipy
python main.py
```

## モード

- **PLAYER MODE** : キーボードで車を直接操作。フィールドの感触を確認する。
- **GA MODE** : シンプルな遺伝的アルゴリズムがリアルタイムで学習する様子を観察する。
- （将来）**RNN-BOTTLENECK MODE** : 2つの海馬RNN＋NEATボトルネックによる完全実装。

## プロジェクト構造

```
main.py              # エントリーポイント・モード選択
game/
  field.py           # 地形生成（山・谷・峠・餌出現）
  car.py             # 車の物理シミュレーション
  agent.py           # エージェント基底クラス（モジュール差し替え口）
  player_agent.py    # プレイヤー操作エージェント
  ga_agent.py        # シンプルGAエージェント
  bottleneck.py      # ボトルネック通信路（スタブ→将来RNN実装）
  renderer.py        # 描画システム
  hud.py             # HUD・情報表示
config.py            # 定数・パラメータ
```

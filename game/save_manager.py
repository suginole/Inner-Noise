"""
save_manager.py — SQLiteセーブ・ロード管理

DB スキーマ:
  models テーブルに全個体の重みベクトル（pickle）を保存する。
  ネットワーク構造メタデータで互換性チェックを行う。

保存対象:
  - 全個体の重みベクトル（種情報含む）
  - エリート個体の重みベクトル
  - 適応的突然変異率の現在値
  - 世代数カウンタ

保存しないもの:
  - GRUの隠れ状態（エピソード内オンラインのため不要）
"""
import sqlite3
import pickle
import json
import os
from datetime import datetime, timezone


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "saves.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """DBを初期化してテーブルを作成する。"""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_at     TEXT,
                terrain_seed INTEGER,
                generation   INTEGER,
                best_fitness REAL,
                avg_fitness  REAL,
                goal_count   INTEGER,
                genome_data  BLOB,
                net_meta     TEXT
            )
        """)
        conn.commit()


def _build_net_meta(ga) -> dict:
    """ネットワーク構造メタデータを構築する（新GRU分割アーキテクチャ対応）。"""
    from game.ga_agent import GAGenome, OBS_DIM
    from game.rnn_bottleneck import SensoryNN, MotorNN
    from config import (
        SENSORY_INPUT_DIM, SENSORY_FF_DIM, SENSORY_GRU_DIM, SENSORY_INTEG_DIM,
        MOTOR_EMBED_DIM, MOTOR_GRU_DIM, MOTOR_INTEG_DIM, MOTOR_CORTEX_DIM,
        MOTOR_OUTPUT_DIM, BN_PARAMS, GRU_INHERIT_DIM, GRU_EPISODE_DIM,
    )
    return {
        "obs_dim":            OBS_DIM,
        "sensory_input_dim":  SENSORY_INPUT_DIM,
        "sensory_ff_dim":     SENSORY_FF_DIM,
        "sensory_gru_dim":    SENSORY_GRU_DIM,
        "sensory_integ_dim":  SENSORY_INTEG_DIM,
        "motor_embed_dim":    MOTOR_EMBED_DIM,
        "motor_gru_dim":      MOTOR_GRU_DIM,
        "motor_integ_dim":    MOTOR_INTEG_DIM,
        "motor_cortex_dim":   MOTOR_CORTEX_DIM,
        "motor_output_dim":   MOTOR_OUTPUT_DIM,
        "bn_params":          BN_PARAMS,
        "gru_inherit_dim":    GRU_INHERIT_DIM,
        "gru_episode_dim":    GRU_EPISODE_DIM,
        "sensory_flat_size":  SensoryNN.flat_size(),
        "motor_flat_size":    MotorNN.flat_size(),
        "genome_flat_size":   GAGenome.total_flat_size(),
        "pop_size":           ga.pop_size,
    }


def _check_compat(meta: dict) -> tuple[bool, str]:
    """現在の構造とメタデータの互換性をチェックする（新GRU分割アーキテクチャ対応）。"""
    from game.ga_agent import GAGenome, OBS_DIM
    from game.rnn_bottleneck import SensoryNN, MotorNN
    from config import (
        SENSORY_INPUT_DIM, SENSORY_FF_DIM, SENSORY_GRU_DIM, SENSORY_INTEG_DIM,
        MOTOR_EMBED_DIM, MOTOR_GRU_DIM, MOTOR_INTEG_DIM, MOTOR_CORTEX_DIM,
        MOTOR_OUTPUT_DIM, BN_PARAMS, GRU_INHERIT_DIM, GRU_EPISODE_DIM,
    )
    errors = []

    # --- 旧アーキテクチャ（H1/H2形式）との互換性チェック ---
    # 旧形式のメタデータが保存されている場合は非互換と判定する
    if "h1" in meta or "h2" in meta:
        return False, "旧アーキテクチャ形式 (H1/H2) のデータは現在のGRU分割設計と互換性がありません"

    # --- 新アーキテクチャの互換性チェック ---
    # obs_dim は必須チェック
    if meta.get("obs_dim") != OBS_DIM:
        errors.append(f"obs_dim: saved={meta.get('obs_dim')} current={OBS_DIM}")

    # ゲノムフラットサイズが記録されている場合はそれを優先チェック
    if "genome_flat_size" in meta:
        current_flat = GAGenome.total_flat_size()
        if meta["genome_flat_size"] != current_flat:
            errors.append(
                f"genome_flat_size: saved={meta['genome_flat_size']} current={current_flat}"
            )
    else:
        # 旧形式の新アーキテクチャメタデータ（個別次元チェック）
        checks = [
            ("sensory_input_dim",  SENSORY_INPUT_DIM),
            ("sensory_ff_dim",     SENSORY_FF_DIM),
            ("sensory_gru_dim",    SENSORY_GRU_DIM),
            ("sensory_integ_dim",  SENSORY_INTEG_DIM),
            ("motor_embed_dim",    MOTOR_EMBED_DIM),
            ("motor_gru_dim",      MOTOR_GRU_DIM),
            ("motor_integ_dim",    MOTOR_INTEG_DIM),
            ("motor_cortex_dim",   MOTOR_CORTEX_DIM),
            ("motor_output_dim",   MOTOR_OUTPUT_DIM),
            ("bn_params",          BN_PARAMS),
            ("gru_inherit_dim",    GRU_INHERIT_DIM),
            ("gru_episode_dim",    GRU_EPISODE_DIM),
        ]
        for key, current_val in checks:
            if key in meta and meta[key] != current_val:
                errors.append(f"{key}: saved={meta[key]} current={current_val}")

    if errors:
        return False, "NN structure mismatch: " + ", ".join(errors)
    return True, "OK"


def save_model(ga, terrain_seed: int, goal_count: int,
               label: str = "") -> int:
    """
    GAの現在状態をDBに保存する。
    Returns: 保存されたレコードのID
    """
    init_db()
    stats = ga.get_stats()

    # 保存データを構築
    data = {
        "population": [g.flat().tolist() for g in ga.population],
        "species_ids": [g.species_id for g in ga.population],
        "mut_rate":   ga._mut_rate,
        "mut_std":    ga._mut_std,
        "generation": ga.generation,
        "best_fitness_history": ga.best_fitness_history,
        "avg_fitness_history":  ga.avg_fitness_history,
        "species_count_history": ga.species_count_history,
    }
    blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    meta = json.dumps(_build_net_meta(ga))
    now  = datetime.now(timezone.utc).isoformat()

    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO models
              (saved_at, terrain_seed, generation, best_fitness,
               avg_fitness, goal_count, genome_data, net_meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, terrain_seed, ga.generation,
              stats["best"], stats["avg"],
              goal_count, blob, meta))
        conn.commit()
        return cur.lastrowid


def load_model(record_id: int, ga):
    """
    DBからモデルをロードしてGAオブジェクトを復元する。
    互換性チェックに失敗した場合は ValueError を送出する。
    """
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM models WHERE id = ?", (record_id,)
        ).fetchone()

    if row is None:
        raise ValueError(f"Record id={record_id} not found.")

    meta = json.loads(row["net_meta"])
    ok, msg = _check_compat(meta)
    if not ok:
        raise ValueError(f"Incompatible model: {msg}")

    data = pickle.loads(row["genome_data"])

    from game.ga_agent import GAGenome
    import numpy as np

    # 個体群を復元
    new_pop = []
    for flat_list in data["population"]:
        g = GAGenome.from_flat(np.array(flat_list, dtype=np.float32))
        new_pop.append(g)

    ga.population = new_pop
    ga.generation = data["generation"]
    ga._mut_rate  = data["mut_rate"]
    ga._mut_std   = data["mut_std"]
    ga.best_fitness_history  = data.get("best_fitness_history", [])
    ga.avg_fitness_history   = data.get("avg_fitness_history", [])
    ga.species_count_history = data.get("species_count_history", [])

    return row["terrain_seed"]


def list_models() -> list[dict]:
    """保存済みモデルの一覧を返す（新しい順）。"""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT id, saved_at, terrain_seed, generation,
                   best_fitness, avg_fitness, goal_count, net_meta
            FROM models
            ORDER BY id DESC
        """).fetchall()
    result = []
    for row in rows:
        meta = json.loads(row["net_meta"]) if row["net_meta"] else {}
        ok, _ = _check_compat(meta)
        result.append({
            "id":           row["id"],
            "saved_at":     row["saved_at"],
            "terrain_seed": row["terrain_seed"],
            "generation":   row["generation"],
            "best_fitness": row["best_fitness"],
            "avg_fitness":  row["avg_fitness"],
            "goal_count":   row["goal_count"],
            "compatible":   ok,
            "pop_size":     meta.get("pop_size", "?"),
        })
    return result


def delete_model(record_id: int):
    """指定IDのモデルを削除する。"""
    init_db()
    with _get_conn() as conn:
        conn.execute("DELETE FROM models WHERE id = ?", (record_id,))
        conn.commit()

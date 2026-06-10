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
    """ネットワーク構造メタデータを構築する（Sage-Brute版）。"""
    from game.ga_agent import GAGenome
    return {
        "arch":              "sage-brute",
        "sage_obs_dim":      SAGE_OBS_DIM,
        "brute_obs_dim":     BRUTE_OBS_DIM,
        "genome_flat_size":  GAGenome.total_flat_size(),
        "pop_size":          ga.pop_size,
    }


def _check_compat(meta: dict) -> tuple[bool, str]:
    """互換性チェック: genome_flat_sizeの一致のみで判定。不一致でもクラッシュしない。"""
    from game.ga_agent import GAGenome
    saved_size = meta.get("genome_flat_size", 0)
    current_size = GAGenome.total_flat_size()
    if saved_size != current_size:
        return False, f"genome_flat_size: saved={saved_size} current={current_size}"
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
        "species_ids": [getattr(g, 'species_id', -1) for g in ga.population],
        "mut_rate":   getattr(ga, '_mut_rate', ga.mut_rate),
        "mut_std":    getattr(ga, '_mut_std', ga.mut_std),
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
    ga.mut_rate   = data["mut_rate"]
    ga.mut_std    = data["mut_std"]
    ga._mut_rate  = ga.mut_rate
    ga._mut_std   = ga.mut_std
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


# ---- SAGE-BRUTE 互換チェック追加 ----
def check_compatible(genome_flat_size: int) -> bool:
    """ゲノムサイズが現在のアーキテクチャと一致するか確認する。
    不一致でもクラッシュさせない。compatible=False を返すだけ。
    """
    try:
        from game.ga_agent import GAGenome
        g = GAGenome()
        return len(g.flat()) == genome_flat_size
    except Exception:
        return False

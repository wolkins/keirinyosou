"""評価基盤: ウォークフォワードCV + 多軸指標"""
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.common.database import Odds, Race

from .features import FEATURE_COLUMNS, build_features_for_race


def evaluate_predictions(valid_df: pd.DataFrame, session: Session | None = None) -> dict:
    """多軸評価指標を算出

    Args:
        valid_df: race_id, pred_score, finish_position, car_number を含むDF
        session: オッズ取得用セッション (ROIシミュレーション用)

    Returns:
        dict: ndcg3, mrr, top1_hit_rate, top3_exact_rate, roi_simulation, avg_payout
    """
    race_ids = valid_df["race_id"].unique()
    n_races = len(race_ids)
    if n_races == 0:
        return {}

    ndcg3_list = []
    mrr_list = []
    top1_hits = 0
    top3_exact = 0
    total_bet = 0
    total_payout = 0.0

    for rid in race_ids:
        race_data = valid_df[valid_df["race_id"] == rid].copy()
        if race_data.empty or "finish_position" not in race_data.columns:
            continue

        # 予測順にソート
        race_data = race_data.sort_values("pred_score", ascending=False).reset_index(drop=True)
        positions = race_data["finish_position"].values

        # --- NDCG@3 ---
        ndcg3_list.append(_ndcg_at_k(positions, k=3))

        # --- MRR (1着のランク逆数) ---
        rank_of_first = np.where(positions == 1)[0]
        if len(rank_of_first) > 0:
            mrr_list.append(1.0 / (rank_of_first[0] + 1))
        else:
            mrr_list.append(0.0)

        # --- Top1 hit rate ---
        if positions[0] == 1:
            top1_hits += 1

        # --- Top3 exact match ---
        pred_top3 = set(race_data.head(3)["car_number"].values)
        actual_top3 = set(
            race_data.nsmallest(3, "finish_position")["car_number"].values
        )
        if pred_top3 == actual_top3:
            top3_exact += 1

        # --- ROIシミュレーション (単勝上位1点100円) ---
        if session is not None:
            top1_car = int(race_data.iloc[0]["car_number"])
            total_bet += 100
            odds_row = (
                session.query(Odds)
                .filter_by(race_id=int(rid), bet_type="win", combination=str(top1_car))
                .first()
            )
            if positions[0] == 1 and odds_row:
                total_payout += odds_row.odds_value * 100

    results = {
        "ndcg3": float(np.mean(ndcg3_list)) if ndcg3_list else 0.0,
        "mrr": float(np.mean(mrr_list)) if mrr_list else 0.0,
        "top1_hit_rate": top1_hits / n_races if n_races > 0 else 0.0,
        "top3_exact_rate": top3_exact / n_races if n_races > 0 else 0.0,
    }

    if session is not None and total_bet > 0:
        results["roi_simulation"] = total_payout / total_bet
        # 的中時平均配当
        hit_payouts = []
        for rid in race_ids:
            race_data = valid_df[valid_df["race_id"] == rid].sort_values(
                "pred_score", ascending=False
            )
            if race_data.empty:
                continue
            if race_data.iloc[0]["finish_position"] == 1:
                top1_car = int(race_data.iloc[0]["car_number"])
                odds_row = (
                    session.query(Odds)
                    .filter_by(race_id=int(rid), bet_type="win", combination=str(top1_car))
                    .first()
                )
                if odds_row:
                    hit_payouts.append(odds_row.odds_value * 100)
        results["avg_payout"] = float(np.mean(hit_payouts)) if hit_payouts else 0.0
    else:
        results["roi_simulation"] = 0.0
        results["avg_payout"] = 0.0

    return results


def _ndcg_at_k(positions: np.ndarray, k: int = 3) -> float:
    """NDCG@k を計算

    relevance = max_pos + 1 - finish_position (1着が最高)
    """
    max_pos = positions.max() if len(positions) > 0 else 9
    relevances = np.maximum(0, max_pos + 1 - positions)

    # DCG@k
    top_k = relevances[:k]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(top_k))

    # ideal DCG@k
    ideal_rels = np.sort(relevances)[::-1][:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_by_group(results_df: pd.DataFrame, group_key: str, session: Session | None = None) -> dict:
    """層別評価: group_key (例: 'grade', 'track_length') でグループ分けして評価

    Args:
        results_df: race_id, pred_score, finish_position, car_number, + group_key列を含むDF
        group_key: グループ分けのカラム名
        session: オッズ取得用セッション

    Returns:
        dict: {group_value: evaluate_predictions結果, ...}
    """
    if group_key not in results_df.columns:
        return {"error": f"カラム '{group_key}' が見つかりません"}

    grouped = {}
    for gval, gdf in results_df.groupby(group_key):
        grouped[gval] = evaluate_predictions(gdf, session=session)
    return grouped


def walk_forward_cv(
    session: Session,
    n_splits: int = 4,
    gap_days: int = 7,
    min_races: int = 50,
) -> dict:
    """ウォークフォワードCV

    Args:
        session: DBセッション
        n_splits: fold数
        gap_days: 学習期間と検証期間のギャップ日数
        min_races: 最低レース数

    Returns:
        dict: folds (各foldの指標リスト), mean (平均), std (標準偏差)
    """
    from datetime import timedelta

    try:
        import lightgbm as lgb
        has_lgb = True
    except (ImportError, OSError):
        has_lgb = False
        from sklearn.ensemble import GradientBoostingRegressor

    from src.common.config import DECAY_HALF_LIFE_DAYS

    races = (
        session.query(Race)
        .filter(Race.status == "finished")
        .order_by(Race.race_date)
        .all()
    )
    if len(races) < min_races:
        return {"error": f"レース数不足 ({len(races)}/{min_races})"}

    # 特徴量構築
    all_dfs = []
    race_date_map = {}
    for race in races:
        df = build_features_for_race(session, race)
        if not df.empty and "finish_position" in df.columns:
            df["race_id"] = race.id
            df["race_date"] = race.race_date
            all_dfs.append(df)
            race_date_map[race.id] = race.race_date

    if not all_dfs:
        return {"error": "有効なデータがありません"}

    full_df = pd.concat(all_dfs, ignore_index=True)
    max_pos = full_df["finish_position"].max()
    full_df["label"] = (max_pos + 1 - full_df["finish_position"]).clip(lower=0)

    # レースを日付順にユニーク取得
    unique_race_ids = full_df.drop_duplicates("race_id").sort_values("race_date")["race_id"].values
    n_total = len(unique_race_ids)

    # 検証期間サイズ (約20%)
    valid_size = max(1, n_total // (n_splits + 4))  # 各foldの検証サイズ
    # fold分割: 累積学習 + gap + 検証
    fold_results = []

    for fold_idx in range(n_splits):
        # 検証開始位置: 全体を均等に分割
        valid_start = n_total - (n_splits - fold_idx) * valid_size
        valid_end = valid_start + valid_size
        if valid_start < min_races // 2:
            continue

        valid_race_ids = set(unique_race_ids[valid_start:valid_end])
        # gap: 検証期間の最小日付 - gap_days より前を学習に使う
        valid_dates = [race_date_map[rid] for rid in valid_race_ids if rid in race_date_map]
        if not valid_dates:
            continue
        gap_cutoff = min(valid_dates) - timedelta(days=gap_days)

        train_race_ids = set(
            rid for rid in unique_race_ids[:valid_start]
            if race_date_map.get(rid, gap_cutoff) <= gap_cutoff
        )
        if len(train_race_ids) < 10:
            continue

        train_df = full_df[full_df["race_id"].isin(train_race_ids)].sort_values("race_id")
        valid_df = full_df[full_df["race_id"].isin(valid_race_ids)].sort_values("race_id")

        available_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]
        X_train = train_df[available_cols].fillna(0)
        y_train = train_df["label"]

        # 時間減衰重み
        ref_date = max(race_date_map[rid] for rid in train_race_ids)
        w_train = train_df["race_id"].map(
            lambda rid: np.exp(
                -np.log(2) * (ref_date - race_date_map.get(rid, ref_date)).days
                / DECAY_HALF_LIFE_DAYS
            )
        )
        group_train = train_df.groupby("race_id").size().tolist()

        X_valid = valid_df[available_cols].fillna(0)
        y_valid = valid_df["label"]
        group_valid = valid_df.groupby("race_id").size().tolist()

        # モデル学習
        if has_lgb:
            ranker = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                ndcg_eval_at=[1, 3],
                learning_rate=0.03,
                num_leaves=31,
                min_data_in_leaf=max(30, len(train_df) // 80),
                feature_fraction=0.7,
                bagging_fraction=0.8,
                bagging_freq=5,
                lambda_l1=0.1,
                lambda_l2=1.0,
                n_estimators=1500,
                verbose=-1,
            )
            callbacks = [lgb.early_stopping(50, verbose=False)]
            ranker.fit(
                X_train, y_train, group=group_train,
                sample_weight=w_train,
                eval_set=[(X_valid, y_valid)],
                eval_group=[group_valid],
                callbacks=callbacks,
            )
        else:
            ranker = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
            )
            ranker.fit(X_train, y_train, sample_weight=w_train)

        # 予測・評価
        valid_df = valid_df.copy()
        valid_df["pred_score"] = ranker.predict(X_valid)

        fold_metrics = evaluate_predictions(valid_df, session=session)
        fold_metrics["fold"] = fold_idx
        fold_metrics["n_train_races"] = len(train_race_ids)
        fold_metrics["n_valid_races"] = len(valid_race_ids)
        fold_results.append(fold_metrics)

    if not fold_results:
        return {"error": "有効なfoldが作成できませんでした"}

    # 平均・標準偏差
    metric_keys = ["ndcg3", "mrr", "top1_hit_rate", "top3_exact_rate", "roi_simulation", "avg_payout"]
    mean_metrics = {}
    std_metrics = {}
    for key in metric_keys:
        vals = [f[key] for f in fold_results if key in f]
        if vals:
            mean_metrics[key] = float(np.mean(vals))
            std_metrics[key] = float(np.std(vals))

    return {
        "n_splits": len(fold_results),
        "folds": fold_results,
        "mean": mean_metrics,
        "std": std_metrics,
    }

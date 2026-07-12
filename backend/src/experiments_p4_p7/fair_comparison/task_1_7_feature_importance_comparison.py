"""
Task 1.7 -- Cross-Architecture Feature Importance Comparison.

Runs sklearn.inspection.permutation_importance on the P6 coarse sub-model
(S3, barlow) using the IDENTICAL parameters the notebook used for the P3
champion (n_repeats=30, random_state=42, scoring='accuracy' -- see
notebooks/gen_nb_new.py's P3 XAI branch), on the saved, already-scaled
Xtest/ytest -- no raw data, no retraining. Compares the resulting top-5
against P3's existing T6_permutation_importance.csv.
"""
import os

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from _common import (
    P6_CHAMPION_SUBJECT, EMOTIV_CH, FEAT_PER_CH, FEAT_SUBNAMES, T_TABLES_DIR,
    p6_submodel_paths, check_exists, load_pickle, save_json,
)

T6_P3_PATH = os.path.join(T_TABLES_DIR, "T6_permutation_importance.csv")


def build_feature_names(feature_group, n_features, channels):
    fpc = FEAT_PER_CH.get(feature_group)
    if fpc and n_features % fpc == 0 and n_features // fpc == len(channels):
        subnames = FEAT_SUBNAMES[feature_group]
        return [f"{ch}_{sn}" for ch in channels for sn in subnames]
    return [f"feat_{i}" for i in range(n_features)]


def main():
    model_p, scaler_p, xtest_p, ytest_p = p6_submodel_paths(P6_CHAMPION_SUBJECT, "coarse")
    missing = check_exists(model_p, xtest_p, ytest_p)
    if missing:
        result = {"status": "DATA_NOT_AVAILABLE", "missing_paths": missing}
        save_json(result, "p6_coarse_permutation_importance_s3.json")
        print(f"[TASK 1.7] MISSING: {missing}")
        return result

    model = load_pickle(model_p)
    X = np.load(xtest_p)
    y = np.load(ytest_p)
    feature_group = "barlow"  # winning_coarse_feature_group for S3, confirmed constant across Stage B
    feat_names = build_feature_names(feature_group, X.shape[1], EMOTIV_CH)

    print(f"[TASK 1.7] Running permutation_importance on P6 coarse/{P6_CHAMPION_SUBJECT} "
          f"({X.shape[0]} samples, {X.shape[1]} features, n_repeats=30)...")
    perm = permutation_importance(model, X, y, n_repeats=30, random_state=42, scoring='accuracy')

    p6_table = (pd.DataFrame({
        'Feature': feat_names,
        'Mean Importance': perm.importances_mean.round(6),
        'Std Importance': perm.importances_std.round(6),
    }).sort_values('Mean Importance', ascending=False).reset_index(drop=True))
    p6_table['Rank'] = range(1, len(p6_table) + 1)

    p6_top5 = p6_table.head(5).to_dict(orient='records')

    comparison = {"p6_coarse_top5": p6_top5}
    if os.path.exists(T6_P3_PATH):
        p3_table = pd.read_csv(T6_P3_PATH)
        p3_top5 = p3_table.head(5).to_dict(orient='records')
        comparison["p3_champion_top5"] = p3_top5

        p3_channels = {f.split('_')[0] for f in p3_table.head(5)['Feature']}
        p6_channels = {f.split('_')[0] for f in p6_table.head(5)['Feature']}
        comparison["overlapping_channels_top5"] = sorted(p3_channels & p6_channels)
        comparison["p3_only_channels_top5"] = sorted(p3_channels - p6_channels)
        comparison["p6_only_channels_top5"] = sorted(p6_channels - p3_channels)
    else:
        comparison["p3_champion_top5"] = None
        comparison["note"] = f"P3 reference table not found at {T6_P3_PATH}"

    result = {
        "status": "OK",
        "description": (
            "Permutation importance (n_repeats=30, random_state=42, scoring='accuracy') on the "
            "P6 coarse sub-model, same methodology as the existing P3 champion table (T6), for a "
            "cross-architecture comparison of which channels/features dominate."
        ),
        "subject_id": P6_CHAMPION_SUBJECT,
        "feature_group": feature_group,
        "n_test_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "full_table": p6_table.to_dict(orient='records'),
        "comparison": comparison,
    }
    save_json(result, "p6_coarse_permutation_importance_s3.json")

    print(f"[TASK 1.7] P6 coarse top-5: {[r['Feature'] for r in p6_top5]}")
    if comparison.get("p3_champion_top5") is not None:
        print(f"[TASK 1.7] P3 champion top-5: {[r['Feature'] for r in comparison['p3_champion_top5']]}")
        print(f"[TASK 1.7] Overlapping channels in top-5: {comparison['overlapping_channels_top5']}")
    return result


if __name__ == "__main__":
    main()

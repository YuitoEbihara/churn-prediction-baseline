"""Telco Customer Churn — LightGBM ベースラインモデル"""

from __future__ import annotations

import pathlib

import kagglehub
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

OUTPUT_DIR = pathlib.Path("outputs")


# ---------------------------------------------------------------------------
# 1. データ取得
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Kaggle から Telco Customer Churn データをダウンロードして読み込む"""
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    csv_path = pathlib.Path(path) / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(csv_path)
    return df


# ---------------------------------------------------------------------------
# 2. 前処理
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """前処理を実施する"""
    df = df.copy()
    df = df.drop(columns=["customerID"])

    # ---- ターゲット ----
    target = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["Churn"])

    # ---- 数値型の修正 ----
    # TotalCharges の空白エラーを回避してゼロ埋めし、数値型に変換
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # ---- カテゴリ変数を category 型に変換 (LightGBM用) ----
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")

    return df, target


# ---------------------------------------------------------------------------
# 3. 学習
# ---------------------------------------------------------------------------
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> lgb.LGBMClassifier:
    """LightGBM を学習する"""
    model = lgb.LGBMClassifier(
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    return model


# ---------------------------------------------------------------------------
# 4. 評価 & ログ出力
# ---------------------------------------------------------------------------
def evaluate_and_export(
    model: lgb.LGBMClassifier,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    feature_names: list[str],
    split_name: str = "test",
) -> None:
    """評価指標を表示し、予測ログと feature importance を CSV に書き出す"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    # ---- コンソール出力 ----
    print("=" * 60)
    print(f"Baseline Model Evaluation on {split_name.upper()} set")
    print("=" * 60)
    print(f"Accuracy : {accuracy_score(y_eval, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_eval, y_proba):.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_eval, y_pred, target_names=["No Churn", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_eval, y_pred))
    print("=" * 60)

    # ---- 予測ログ CSV ----
    log_df = X_eval.copy()
    log_df["y_true"] = y_eval.values
    log_df["y_pred"] = y_pred
    log_df["y_proba"] = y_proba
    log_df["correct"] = (y_eval.values == y_pred).astype(int)
    log_path = OUTPUT_DIR / f"{split_name}_prediction_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n予測ログを保存しました: {log_path}")

    # ---- Feature Importance CSV ----
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    imp_path = OUTPUT_DIR / "feature_importance.csv"
    importance_df.to_csv(imp_path, index=False)
    print(f"Feature Importance を保存しました: {imp_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading data...")
    df = load_data()
    print(f"  rows: {len(df)}, cols: {len(df.columns)}")

    print("Preprocessing...")
    X, y = preprocess(df)

    # Train 60%, Valid 20%, Test 20%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"  train: {len(X_train)}, valid: {len(X_valid)}, test: {len(X_test)}")
    print(f"  Churn rate (train): {y_train.mean():.2%}")

    print("Training baseline LightGBM model...")
    model = train_model(X_train, y_train, X_valid, y_valid)

    print("\n--- Validation Data Evaluation ---")
    evaluate_and_export(model, X_valid, y_valid, list(X.columns), split_name="valid")

    print("\n--- Test Data Evaluation ---")
    evaluate_and_export(model, X_test, y_test, list(X.columns), split_name="test")


if __name__ == "__main__":
    main()

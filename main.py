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
from sklearn.preprocessing import LabelEncoder

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

    # ---- カテゴリ変数を LabelEncoding ----
    label_encoders: dict[str, LabelEncoder] = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, target


# ---------------------------------------------------------------------------
# 3. 学習
# ---------------------------------------------------------------------------
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> lgb.LGBMClassifier:
    """LightGBM を学習する"""
    model = lgb.LGBMClassifier(
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 4. 評価 & ログ出力
# ---------------------------------------------------------------------------
def evaluate_and_export(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> None:
    """評価指標を表示し、予測ログと feature importance を CSV に書き出す"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ---- コンソール出力 ----
    print("=" * 60)
    print("Baseline Model Evaluation")
    print("=" * 60)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 60)

    # ---- 予測ログ CSV ----
    log_df = X_test.copy()
    log_df["y_true"] = y_test.values
    log_df["y_pred"] = y_pred
    log_df["y_proba"] = y_proba
    log_df["correct"] = (y_test.values == y_pred).astype(int)
    log_path = OUTPUT_DIR / "prediction_log.csv"
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  train: {len(X_train)}, test: {len(X_test)}")
    print(f"  Churn rate (train): {y_train.mean():.2%}")

    print("Training baseline LightGBM model...")
    model = train_model(X_train, y_train)

    evaluate_and_export(model, X_test, y_test, list(X.columns))


if __name__ == "__main__":
    main()

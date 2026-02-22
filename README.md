# churn-prediction-baseline

Telco Customer Churn データセット（Kaggle 公開・約 7,000 行 × 21 特徴量）に対する LightGBM ベースラインモデルです。

---

## セットアップ

```bash
uv sync
```

> **macOS の場合**: LightGBM が OpenMP を必要とするため `brew install libomp` を先に実行してください。

## 実行

```bash
uv run python main.py
```

`outputs/` に以下のファイルが出力されます:

| ファイル | 内容 |
|---|---|
| `{valid,test}_prediction_log.csv` | 各データ行ごとの特徴量値 + 正解ラベル + 予測ラベル + 予測確率 + 正誤フラグ |
| `feature_importance.csv` | LightGBM の feature importance（split ベース） |

## ベースライン結果 (Test Data)

```
Accuracy : 0.7850
ROC-AUC  : 0.8268
```

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "Advertising.csv"
OUTPUT_DIR = ROOT / "outputs"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_FILE = ARTIFACT_DIR / "sales_model.joblib"

DATA_URL = (
    "https://raw.githubusercontent.com/justmarkham/"
    "DAT8/master/data/Advertising.csv"
)

FEATURES = ["tv", "radio", "newspaper"]
TARGET = "sales"


def load_data():
    """Load the local CSV or download it if it isn't present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        print(f"Loading local data from {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
    else:
        print("Downloading the Advertising dataset...")
        try:
            df = pd.read_csv(DATA_URL)
            df.to_csv(DATA_FILE, index=False)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download the dataset. Download the assignment CSV "
                f"and save it as {DATA_FILE}, then run this script again."
            ) from exc

    # Normalize column names and remove an optional CSV index column.
    df.columns = [str(column).strip().lower() for column in df.columns]
    df = df.loc[
        :,
        [not column.startswith("unnamed") for column in df.columns]
    ]

    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"CSV is missing columns: {missing_columns}. "
            f"Expected: {required_columns}"
        )

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    rows_before = len(df)
    df = df.dropna(subset=required_columns).copy()
    rows_dropped = rows_before - len(df)

    if len(df) < 20:
        raise ValueError("Not enough valid rows to train and evaluate a model.")

    print(f"Valid rows: {len(df)}; rows dropped due to missing values: {rows_dropped}")
    return df


def regression_metrics(actual, predicted):
    """Return standard regression metrics as regular Python floats."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def save_plots(df, X_test, y_test, predictions, model):
    """Save EDA, evaluation, and feature-importance charts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Correlation heatmap
    columns = FEATURES + [TARGET]
    correlation = df[columns].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)

    for row in range(len(columns)):
        for col in range(len(columns)):
            ax.text(
                col,
                row,
                f"{correlation.iloc[row, col]:.2f}",
                ha="center",
                va="center",
            )

    ax.set_title("Feature and Sales Correlations")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    # Actual sales vs. predicted sales
    actual = np.asarray(y_test)
    predicted = np.asarray(predictions)

    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual, predicted, alpha=0.75)
    ax.plot([low, high], [low, high], "--", color="gray")
    ax.set_xlabel("Actual sales (thousands of units)")
    ax.set_ylabel("Predicted sales (thousands of units)")
    ax.set_title("Actual vs. Predicted Sales")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    # Permutation importance: how much model error increases when a feature is shuffled.
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    importance = pd.Series(
        result.importances_mean,
        index=FEATURES
    ).sort_values()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(importance.index, importance.values, color="#4c78a8")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Increase in RMSE when feature is shuffled")
    ax.set_title("Permutation Feature Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)


def main():
    df = load_data()

    print("\nDataset summary:")
    print(df[FEATURES + [TARGET]].describe().round(2))

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # Compare models using cross-validation on training data only.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, estimator in candidates.items():
        scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        )
        fold_rmse = -scores
        cv_results[name] = {
            "mean_rmse": float(fold_rmse.mean()),
            "std_rmse": float(fold_rmse.std()),
        }

    best_name = min(
        cv_results,
        key=lambda name: cv_results[name]["mean_rmse"],
    )
    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)

    test_predictions = best_model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_predictions)

    # A mean-only model is a useful simple baseline.
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = regression_metrics(y_test, baseline_predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_file = X_test.copy()
    predictions_file["actual_sales"] = y_test
    predictions_file["predicted_sales"] = test_predictions
    predictions_file.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    save_plots(df, X_test, y_test, test_predictions, best_model)

    metrics = {
        "selected_model": best_name,
        "cv_rmse": cv_results[best_name]["mean_rmse"],
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_metrics,
        "n_samples": int(len(df)),
        "test_samples": int(len(X_test)),
    }

    feature_ranges = {
        feature: {
            "min": float(df[feature].min()),
            "max": float(df[feature].max()),
        }
        for feature in FEATURES
    }

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": FEATURES,
        "feature_ranges": feature_ranges,
        "metrics": metrics,
    }

    joblib.dump(bundle, MODEL_FILE)
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"\nSelected model: {best_name}")
    print("Cross-validation results:")
    print(json.dumps(cv_results, indent=2))
    print("\nHoldout test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("\nMean baseline metrics:")
    print(json.dumps(baseline_metrics, indent=2))
    print(f"\nSaved model to: {MODEL_FILE}")
    print(f"Saved charts and predictions to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
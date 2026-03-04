"""
CryptoPulse — Model Training (Enhanced)
Trains an XGBoost classifier with hyperparameter tuning via
RandomizedSearchCV using TimeSeriesSplit cross-validation.
"""
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBClassifier

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.processing import FeatureEngineer
from ml.evaluate import ModelEvaluator
from core.utils import save_model
from config.settings import MODELS_DIR, TEST_SIZE
from core.logger import logging

# ── Hyperparameter search space ──────────────────────────────────────────
PARAM_DISTRIBUTIONS = {
    "n_estimators": [300, 500, 800, 1000, 1500],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5, 7, 10],
    "gamma": [0, 0.05, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.05, 0.1, 0.5],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
}


class ModelTrainer:
    """Trains an XGBoost classifier for crypto price direction prediction."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = None

    def train(self, tune_hyperparams: bool = True):
        """
        Full training pipeline:
        1. Load & engineer features
        2. Split (time-series order preserved)
        3. Scale features
        4. Optionally tune hyperparameters with TimeSeriesSplit CV
        5. Train XGBClassifier with best params + early stopping
        6. Feature selection pass (drop low-importance features)
        7. Retrain on selected features
        8. Evaluate & save model
        """
        print("\n📊 Loading and engineering features...")
        fe = FeatureEngineer()
        df = fe.prepare_features()
        self.feature_names = fe.get_feature_columns(df)

        print(f"   Features: {len(self.feature_names)} columns, {len(df)} rows")

        # Split data
        X_train, X_test, y_train, y_test = fe.split_data(df, test_size=TEST_SIZE)
        print(f"   Train: {len(X_train)} rows | Test: {len(X_test)} rows")

        # Scale features
        print("\n⚙️  Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Class balance weight
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        scale_pos_weight = n_neg / max(n_pos, 1)
        print(f"   Class balance — UP: {n_pos}, DOWN: {n_neg}, "
              f"scale_pos_weight: {scale_pos_weight:.2f}")

        # ── Hyperparameter Tuning ────────────────────────────────────────
        if tune_hyperparams and len(X_train) >= 200:
            print("\n🔍 Tuning hyperparameters (RandomizedSearchCV + TimeSeriesSplit)...")
            best_params = self._tune_hyperparams(
                X_train_scaled, y_train, scale_pos_weight
            )
        else:
            print("\n⚡ Using default hyperparameters (not enough data for tuning)...")
            best_params = {
                "n_estimators": 800,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_weight": 5,
                "gamma": 0.1,
                "reg_alpha": 0.05,
                "reg_lambda": 1.5,
            }

        # ── Train with best params ───────────────────────────────────────
        print(f"\n🤖 Training XGBoost with tuned params...")
        self.model = XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=50,
            tree_method="hist",
        )

        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )
        print(f"   Best iteration: {self.model.best_iteration}")

        # ── Feature Selection Pass ───────────────────────────────────────
        print("\n✂️  Feature selection — removing low-importance features...")
        X_train_sel, X_test_sel, selected_features = self._select_features(
            X_train, X_test, X_train_scaled, threshold_percentile=15
        )

        if len(selected_features) < len(self.feature_names):
            print(f"   Reduced: {len(self.feature_names)} → {len(selected_features)} features")
            self.feature_names = selected_features

            # Re-fit scaler on selected features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_sel)
            X_test_scaled = self.scaler.transform(X_test_sel)

            # Retrain on selected features
            print("\n🤖 Retraining on selected features...")
            self.model = XGBClassifier(
                **best_params,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                early_stopping_rounds=50,
                tree_method="hist",
            )
            self.model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False,
            )
            print(f"   Best iteration: {self.model.best_iteration}")
        else:
            print("   All features retained.")

        # ── Evaluate ─────────────────────────────────────────────────────
        print("\n📈 Evaluating model...")
        evaluator = ModelEvaluator()
        self.metrics = evaluator.evaluate(
            self.model, X_test_scaled, y_test, self.feature_names
        )
        evaluator.print_report(self.metrics)

        # ── Save ─────────────────────────────────────────────────────────
        print("\n💾 Saving model artifacts...")
        self.metrics["best_params"] = best_params
        save_model(
            model=self.model,
            scaler=self.scaler,
            feature_names=self.feature_names,
            metrics=self.metrics,
            models_dir=MODELS_DIR,
        )
        print(f"   Saved to: {MODELS_DIR}")

        return self.metrics

    def _tune_hyperparams(
        self, X_train_scaled, y_train, scale_pos_weight
    ) -> dict:
        """
        Run RandomizedSearchCV with TimeSeriesSplit cross-validation.
        Returns the best parameter dict.
        """
        tscv = TimeSeriesSplit(n_splits=5)

        base_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            tree_method="hist",
        )

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=40,
            scoring="f1",
            cv=tscv,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(X_train_scaled, y_train)

        best_params = search.best_params_
        print(f"   Best CV F1: {search.best_score_:.4f}")
        print(f"   Best params: { {k: v for k, v in sorted(best_params.items())} }")

        return best_params

    def _select_features(
        self, X_train_raw, X_test_raw, X_train_scaled, threshold_percentile=15
    ):
        """
        Remove features below the given percentile of importance.
        Returns filtered X_train, X_test (unscaled), and selected feature names.
        """
        importances = self.model.feature_importances_
        threshold = np.percentile(importances, threshold_percentile)

        selected_mask = importances > threshold
        selected_features = [
            f for f, keep in zip(self.feature_names, selected_mask) if keep
        ]

        X_train_sel = X_train_raw[selected_features]
        X_test_sel = X_test_raw[selected_features]

        return X_train_sel, X_test_sel, selected_features


def main():
    print("=" * 50)
    print("  CryptoPulse — Enhanced Model Training")
    print("=" * 50)

    try:
        trainer = ModelTrainer()
        metrics = trainer.train(tune_hyperparams=True)
        print("\n🎉 Training complete!")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1 Score: {metrics['f1_score']:.4f}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

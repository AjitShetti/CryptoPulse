"""
CryptoPulse — Model Evaluation
Computes classification metrics and generates evaluation reports.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from core.logger import logging


class ModelEvaluator:
    """Evaluates a trained classification model."""

    def evaluate(
        self, model, X_test, y_test, feature_names: list = None
    ) -> dict:
        """
        Run predictions on test set and compute metrics.

        Returns:
            Dict with accuracy, precision, recall, f1_score,
            classification_report, confusion_matrix, and feature_importances.
        """
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["DOWN", "UP"], output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "test_samples": int(len(y_test)),
            "class_distribution": {
                "DOWN": int((y_test == 0).sum()),
                "UP": int((y_test == 1).sum()),
            },
        }

        # Feature importances
        if hasattr(model, "feature_importances_") and feature_names:
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:10]
            metrics["top_features"] = [
                {"name": feature_names[i], "importance": float(importances[i])}
                for i in top_indices
            ]

        logging.info(
            f"Evaluation: acc={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1_score']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}"
        )
        return metrics

    @staticmethod
    def print_report(metrics: dict):
        """Pretty-print evaluation metrics."""
        print("\n" + "─" * 40)
        print("  MODEL EVALUATION REPORT")
        print("─" * 40)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  Test Size: {metrics['test_samples']} samples")
        print(f"  Classes:   DOWN={metrics['class_distribution']['DOWN']}, "
              f"UP={metrics['class_distribution']['UP']}")

        cm = metrics["confusion_matrix"]
        print(f"\n  Confusion Matrix:")
        print(f"              Pred DOWN  Pred UP")
        print(f"  Actual DOWN    {cm[0][0]:>5}    {cm[0][1]:>5}")
        print(f"  Actual UP      {cm[1][0]:>5}    {cm[1][1]:>5}")

        if "top_features" in metrics:
            print(f"\n  Top 10 Features:")
            for i, feat in enumerate(metrics["top_features"], 1):
                bar = "█" * int(feat["importance"] * 50)
                print(f"    {i:>2}. {feat['name']:<25} {feat['importance']:.4f} {bar}")
        print("─" * 40)

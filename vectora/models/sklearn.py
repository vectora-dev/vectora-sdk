from __future__ import annotations

from datetime import datetime, timezone
import sys
import threading
from typing import Any
from uuid import UUID

import numpy as np

from vectora.compliance.trace import generate_trace_id
from vectora.exceptions import VectoraConfigError
from vectora.explain.shap import SHAPWrapper

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
except Exception:  # pragma: no cover - dependency contract handles this at install time
    accuracy_score = None
    f1_score = None
    precision_score = None
    recall_score = None


class SklearnConnector:
    def __init__(self, client: Any, model_id: str, model: Any | None = None) -> None:
        self.client = client
        self.model = model
        self.model_id = self._validate_model_id(model_id)
        self._shap = SHAPWrapper()

    def predict(self, X: Any, y_true: Any | None = None, model: Any | None = None) -> Any:
        model_to_use = model or self.model
        if model_to_use is None:
            raise VectoraConfigError(
                "Provide a fitted sklearn-compatible model when creating SklearnConnector or calling predict()."
            )

        predictions = model_to_use.predict(X)
        payload = self._build_payload(model_to_use, X, predictions, y_true=y_true)

        thread = threading.Thread(
            target=self._send_payload,
            args=(payload,),
            daemon=True,
        )
        thread.start()

        return predictions

    def _build_payload(
        self,
        model: Any,
        X: Any,
        predictions: Any,
        y_true: Any | None = None,
    ) -> dict[str, Any]:
        feature_names = self._feature_names(X)
        metrics = self._compute_metrics(y_true, predictions)
        feature_distributions = self._compute_distributions(X, feature_names)
        shap_values = self._shap.compute(model, X, feature_names)

        return {
            "trace_id": generate_trace_id(),
            "model_id": self.model_id,
            "metrics": metrics,
            "feature_distributions": feature_distributions,
            "shap_values": shap_values,
            "sample_count": int(self._row_count(X)),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    def _compute_metrics(self, y_true: Any | None, predictions: Any) -> dict[str, float]:
        if y_true is None:
            return {}

        if accuracy_score is None or f1_score is None or precision_score is None or recall_score is None:
            return {}

        y_true_array = np.asarray(y_true)
        predictions_array = np.asarray(predictions)
        average = "binary" if len(np.unique(y_true_array)) <= 2 else "weighted"

        return {
            "accuracy": float(accuracy_score(y_true_array, predictions_array)),
            "f1": float(f1_score(y_true_array, predictions_array, average=average, zero_division=0)),
            "precision": float(
                precision_score(y_true_array, predictions_array, average=average, zero_division=0)
            ),
            "recall": float(recall_score(y_true_array, predictions_array, average=average, zero_division=0)),
        }

    def _compute_distributions(self, X: Any, feature_names: list[str]) -> dict[str, dict[str, float]]:
        array = self._as_2d_array(X)
        distributions: dict[str, dict[str, float]] = {}

        for index, feature_name in enumerate(feature_names):
            column = array[:, index].astype(float)
            distributions[feature_name] = {
                "mean": float(np.mean(column)),
                "std": float(np.std(column)),
                "p25": float(np.percentile(column, 25)),
                "p50": float(np.percentile(column, 50)),
                "p75": float(np.percentile(column, 75)),
            }

        return distributions

    def _send_payload(self, payload: dict[str, Any]) -> None:
        trace_id = payload.get("trace_id", "unknown-trace")
        try:
            self.client._post("/api/ingest/metrics", payload)
        except Exception as exc:
            print(
                f"[vectora] failed to send metrics for trace_id={trace_id}: {exc}",
                file=sys.stderr,
            )

    def _feature_names(self, X: Any) -> list[str]:
        if hasattr(X, "columns"):
            return [str(column) for column in X.columns]

        array = self._as_2d_array(X)
        return [f"feature_{index}" for index in range(array.shape[1])]

    def _as_2d_array(self, X: Any) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            array = X.to_numpy()
        else:
            array = np.asarray(X)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise VectoraConfigError("X must be a 2D array-like object for Vectora monitoring.")

        return array

    def _row_count(self, X: Any) -> int:
        return int(self._as_2d_array(X).shape[0])

    def _validate_model_id(self, model_id: str) -> str:
        try:
            return str(UUID(model_id))
        except (ValueError, TypeError) as exc:
            raise VectoraConfigError("model_id must be a valid UUID string.") from exc

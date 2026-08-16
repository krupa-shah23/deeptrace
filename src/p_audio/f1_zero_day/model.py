import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class OneClassSVMDetector:
    """
    Zero-Day Novelty Detector using One-Class SVM trained ONLY on genuine human speech features.
    Learns the boundary of real human speech distribution in feature space.
    """

    def __init__(self, kernel: str = "rbf", nu: float = 0.1, gamma: str = "scale"):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.svm = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.is_fitted = False
        self.feature_names = []
        self.metadata = {}

    def fit(self, X: np.ndarray, feature_names: Optional[list] = None, metadata: Optional[Dict[str, Any]] = None):
        """Fits StandardScaler and One-Class SVM on genuine real speech feature matrix X."""
        if X.ndim != 2 or len(X) == 0:
            raise ValueError(f"Invalid input matrix X shape: {X.shape}. Must be 2D non-empty array.")

        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled)
        self.is_fitted = True
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.metadata = metadata or {}
        self.metadata.update({
            "num_training_samples": len(X),
            "feature_dim": X.shape[1],
            "kernel": self.kernel,
            "nu": self.nu,
            "gamma": self.gamma
        })

    def predict_score(self, feature_vector: np.ndarray) -> Tuple[float, float]:
        """
        Computes One-Class SVM decision score and normalized consistency score.
        - decision_score: Raw distance from decision boundary (>= 0 inside real speech, < 0 outside).
        - consistency_score: Sigmoid-mapped normalized score between 0.0 and 1.0.
        """
        if not self.is_fitted:
            raise RuntimeError("OneClassSVMDetector is not fitted. Train or load a model artifact first.")

        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        feature_vector_scaled = self.scaler.transform(feature_vector)
        raw_score = float(self.svm.decision_function(feature_vector_scaled)[0])

        # Handle RBF kernel asymptotic saturation (where K(x, xi) -> 0 and decision_function clamps to -offset_)
        if hasattr(self.svm, "offset_") and hasattr(self.svm, "support_vectors_") and len(self.svm.support_vectors_) > 0:
            offset = float(self.svm.offset_[0])
            if abs(raw_score + offset) < 1e-4:
                sv_dists = np.linalg.norm(self.svm.support_vectors_ - feature_vector_scaled, axis=1)
                min_dist = float(np.min(sv_dists))
                sv_radii = np.linalg.norm(self.svm.support_vectors_, axis=1)
                r_sv = float(np.mean(sv_radii)) if len(sv_radii) > 0 and np.mean(sv_radii) > 1e-6 else 1.0
                raw_score = float(-offset - (min_dist / r_sv))

        # Sigmoid mapping for smooth score between 0.0 and 1.0
        # Positive raw_score -> > 0.5 (consistent with real speech)
        # Negative raw_score -> < 0.5 (anomaly signal / synthetic speech characteristic)
        scale_factor = 0.1 if raw_score < -1.0 else 2.0
        consistency_score = float(1.0 / (1.0 + np.exp(-scale_factor * raw_score)))

        return raw_score, consistency_score

    def save(self, model_path: Path):
        """Saves fitted scaler, SVM model, and metadata to joblib file."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "scaler": self.scaler,
            "svm": self.svm,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "is_fitted": self.is_fitted
        }
        joblib.dump(artifact, model_path)
        print(f"[F1 Model] Saved One-Class SVM artifact to {model_path}")

    @classmethod
    def load(cls, model_path: Path) -> "OneClassSVMDetector":
        """Loads One-Class SVM detector from joblib file."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

        artifact = joblib.load(model_path)
        instance = cls()
        instance.scaler = artifact["scaler"]
        instance.svm = artifact["svm"]
        instance.feature_names = artifact.get("feature_names", [])
        instance.metadata = artifact.get("metadata", {})
        instance.is_fitted = artifact.get("is_fitted", True)
        return instance

"""
Uncertainty estimation via ensembles and MC Dropout.

These methods produce uncertainty estimates by aggregating predictions
from multiple models or stochastic forward passes.
"""

from typing import List, Tuple

import numpy as np


class DeepEnsemblePredictor:
    """
    Aggregate predictions from an ensemble of independently trained models.
    Uncertainty = variance across ensemble members.
    """

    def __init__(self, models: list):
        self.models = models

    def predict_proba(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            mean_probs: Mean predicted probability across ensemble
            epistemic_uncertainty: Variance across ensemble members
        """
        all_probs = np.array([m.predict_proba(X) for m in self.models])
        mean_probs = all_probs.mean(axis=0)
        uncertainty = all_probs.var(axis=0)
        return mean_probs, uncertainty

    def predict_with_decomposition(
        self, X
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose total uncertainty into aleatoric and epistemic components.

        Total uncertainty ~= aleatoric + epistemic
        - Aleatoric: mean of per-model variance (irreducible)
        - Epistemic: variance of per-model means (reducible with more data)
        """
        all_probs = np.array([m.predict_proba(X) for m in self.models])
        mean_probs = all_probs.mean(axis=0)

        # Epistemic: variance of means
        epistemic = all_probs.var(axis=0)

        # Aleatoric: mean of per-model Bernoulli variance
        aleatoric = np.mean(all_probs * (1 - all_probs), axis=0)

        return mean_probs, aleatoric, epistemic


class MCDropoutPredictor:
    """
    MC Dropout uncertainty estimation.
    Requires a PyTorch model with dropout layers.
    """

    def __init__(self, model, n_forward_passes: int = 50):
        self.model = model
        self.n_forward_passes = n_forward_passes

    def predict_proba(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run multiple stochastic forward passes with dropout enabled.

        Returns:
            mean_probs: Mean predicted probability
            uncertainty: Variance across forward passes
        """
        import torch

        self.model.model.train()  # Enable dropout

        X_scaled = self.model.scaler.transform(X)
        xb = torch.FloatTensor(X_scaled).to(self.model.device)

        all_probs = []
        with torch.no_grad():
            for _ in range(self.n_forward_passes):
                logits = self.model.model(xb).squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        all_probs = np.array(all_probs)
        mean_probs = all_probs.mean(axis=0)
        uncertainty = all_probs.var(axis=0)

        self.model.model.eval()

        return mean_probs, uncertainty

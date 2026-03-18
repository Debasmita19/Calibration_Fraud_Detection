"""
Baseline fraud detection models.

Implements a unified interface for training and predicting with:
- XGBoost
- LightGBM
- MLP (PyTorch deep tabular)
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class BaseModel:
    """Abstract base for all fraud detection models."""

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        """Return P(fraud) for each sample."""
        raise NotImplementedError

    def evaluate(self, X, y) -> dict:
        """Compute standard classification metrics."""
        probs = self.predict_proba(X)
        return {
            "auroc": roc_auc_score(y, probs),
            "auprc": average_precision_score(y, probs),
        }


class XGBoostFraud(BaseModel):
    def __init__(self, params: dict):
        import xgboost as xgb
        self.params = params.copy()
        self.model = None
        self._xgb = xgb

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from src.utils.helpers import get_class_weight

        params = self.params.copy()
        if params.get("scale_pos_weight") == "auto":
            params["scale_pos_weight"] = get_class_weight(y_train)

        early_stopping = params.pop("early_stopping_rounds", 50)
        n_estimators = params.pop("n_estimators", 500)

        self.model = self._xgb.XGBClassifier(
            n_estimators=n_estimators,
            use_label_encoder=False,
            **params,
        )

        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=100,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class LightGBMFraud(BaseModel):
    def __init__(self, params: dict):
        import lightgbm as lgb
        self.params = params.copy()
        self.model = None
        self._lgb = lgb

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        params = self.params.copy()
        early_stopping = params.pop("early_stopping_rounds", 50)
        n_estimators = params.pop("n_estimators", 500)

        callbacks = [self._lgb.log_evaluation(100)]
        if early_stopping:
            callbacks.append(self._lgb.early_stopping(early_stopping))

        self.model = self._lgb.LGBMClassifier(
            n_estimators=n_estimators,
            **params,
        )

        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class MLPFraud(BaseModel):
    """Simple deep tabular model using PyTorch."""

    def __init__(self, params: dict):
        self.params = params.copy()
        self.model = None
        self.scaler = None
        self.device = None

    def _build_model(self, input_dim: int):
        import torch
        import torch.nn as nn

        hidden_dims = self.params.get("hidden_dims", [256, 128, 64])
        dropout = self.params.get("dropout", 0.3)

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        return nn.Sequential(*layers)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        X_t = torch.FloatTensor(X_scaled)
        y_t = torch.FloatTensor(y_train)
        dataset = TensorDataset(X_t, y_t)

        batch_size = self.params.get("batch_size", 2048)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = self._build_model(X_train.shape[1]).to(self.device)

        # Weighted BCE for class imbalance
        pos_weight = torch.tensor(
            [np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)]
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.get("learning_rate", 1e-3),
            weight_decay=self.params.get("weight_decay", 1e-4),
        )

        epochs = self.params.get("epochs", 100)
        patience = self.params.get("early_stopping_patience", 10)
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb).squeeze()
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if X_val is not None and (epoch + 1) % 5 == 0:
                val_loss = self._val_loss(X_val, y_val, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.model.load_state_dict(best_state)
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

        return self

    def _val_loss(self, X_val, y_val, criterion):
        import torch

        self.model.eval()
        X_scaled = self.scaler.transform(X_val)
        with torch.no_grad():
            xb = torch.FloatTensor(X_scaled).to(self.device)
            yb = torch.FloatTensor(y_val).to(self.device)
            logits = self.model(xb).squeeze()
            return criterion(logits, yb).item()

    def predict_proba(self, X) -> np.ndarray:
        import torch

        self.model.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            xb = torch.FloatTensor(X_scaled).to(self.device)
            logits = self.model(xb).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


def get_model(name: str, params: dict) -> BaseModel:
    """Factory function to create a model by name."""
    models = {
        "xgboost": XGBoostFraud,
        "lightgbm": LightGBMFraud,
        "mlp": MLPFraud,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")
    return models[name](params)

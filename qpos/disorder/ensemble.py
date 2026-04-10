import numpy as np
import warnings
import pandas as pd

class DisorderNetV6:
    def __init__(self, config=None):
        self.config = config or {}
        try:
            from lightgbm import LGBMClassifier
            from xgboost import XGBClassifier
            self.lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63, random_state=42)
            self.xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
            self.models_available = True
        except ImportError:
            self.models_available = False
            
        self.is_fitted = False
        self.feature_names_ = None
        
    def fit(self, sequences, labels):
        if not self.models_available:
            warnings.warn("LGBM/XGBoost unavailable. Install with qpos[ml].")
            return
            
        from .features import extract_features
        X_list = [extract_features(seq) for seq in sequences]
        X = np.vstack(X_list)
        y = np.concatenate(labels)
        
        self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=self.feature_names_)
        
        self.lgbm.fit(X_df, y)
        self.xgb.fit(X_df, y)
        self.is_fitted = True
        
    def predict(self, sequence: str) -> np.ndarray:
        if not self.is_fitted or not self.models_available:
            return self._heuristic_predict(sequence)
            
        from .features import extract_features
        X = extract_features(sequence)
        
        X_df = pd.DataFrame(X, columns=self.feature_names_)
        p_lgbm = self.lgbm.predict_proba(X_df)[:, 1]
        p_xgb = self.xgb.predict_proba(X_df)[:, 1]
        return (p_lgbm + p_xgb) / 2.0
        
    def _heuristic_predict(self, sequence: str) -> np.ndarray:
        promoters_disorder = {'R', 'K', 'E', 'S', 'P', 'Q', 'A', 'G'}
        promoters_order = {'W', 'Y', 'F', 'I', 'L', 'V', 'C', 'N'}
        scores = []
        for res in sequence:
            if res in promoters_disorder:
                scores.append(0.7)
            elif res in promoters_order:
                scores.append(0.2)
            else:
                scores.append(0.45)
        
        w = 11
        half = w // 2
        padded = np.pad(scores, (half, half), mode='edge')
        cumsum = np.cumsum(np.insert(padded, 0, 0))
        moving_avg = (cumsum[w:] - cumsum[:-w]) / float(w)
        return moving_avg

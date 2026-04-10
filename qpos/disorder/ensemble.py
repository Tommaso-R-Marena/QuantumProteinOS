import numpy as np
import warnings
import pandas as pd
from sklearn.decomposition import PCA
from qpos.disorder.features import DisorderFeatureExtractor

class DisorderNetV6:
    def __init__(self, config=None):
        self.config = config or {}
        self.use_esm = self.config.get('use_esm', True)
        self.feature_extractor = DisorderFeatureExtractor(use_esm=self.use_esm)
        
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
        """Fit the ensemble model on protein sequences and disorder labels."""
        if not self.models_available:
            warnings.warn("LGBM/XGBoost unavailable. Install with qpos[ml].")
            return
            
        # 1. Fit PCA on ESM embeddings if available
        if self.feature_extractor.esm_available:
            print("Extracting ESM embeddings for PCA fitting...")
            all_embeddings = []
            for seq in sequences:
                try:
                    emb = self.feature_extractor.esm.get_embeddings(seq)
                    if hasattr(emb, 'cpu'):
                        emb = emb.cpu().numpy()
                    all_embeddings.append(emb)
                except Exception as e:
                    print(f"Warning: Failed to extract ESM for sequence: {e}")
            
            if all_embeddings:
                X_esm = np.vstack(all_embeddings)
                print(f"Fitting PCA on {X_esm.shape[0]} residue embeddings...")
                self.feature_extractor.pca = PCA(n_components=20, random_state=42)
                self.feature_extractor.pca.fit(X_esm)
        
        # 2. Extract 406-dimensional features
        print("Extracting full 406-dim feature set...")
        X_list = [self.feature_extractor.extract_features(seq) for seq in sequences]
        X = np.vstack(X_list)
        y = np.concatenate(labels)
        
        # 3. Fit classifiers
        self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=self.feature_names_)
        
        print("Training LightGBM and XGBoost ensemble...")
        self.lgbm.fit(X_df, y)
        self.xgb.fit(X_df, y)
        self.is_fitted = True
        
        # TODO: Add joblib.dump(self, path) here for model persistence in production
        
    def predict(self, sequence: str) -> np.ndarray:
        """Predict per-residue disorder probability."""
        if not self.is_fitted or not self.models_available:
            return self._heuristic_predict(sequence)
            
        X = self.feature_extractor.extract_features(sequence)
        X_df = pd.DataFrame(X, columns=self.feature_names_)
        
        p_lgbm = self.lgbm.predict_proba(X_df)[:, 1]
        p_xgb = self.xgb.predict_proba(X_df)[:, 1]
        return (p_lgbm + p_xgb) / 2.0
        
    def _heuristic_predict(self, sequence: str) -> np.ndarray:
        """Fallback heuristic based on amino acid propensities."""
        # Top disorder/order promoters (Uversky/Dunker)
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
        
        # Sliding average window 11
        w = 11
        half = w // 2
        padded = np.pad(scores, (half, half), mode='edge')
        cumsum = np.cumsum(np.insert(padded, 0, 0))
        moving_avg = (cumsum[w:] - cumsum[:-w]) / float(w)
        return moving_avg

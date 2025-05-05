import xgboost as xgb
import numpy as np
from collections import Counter
import torch

class XGBRegWrapper:
    def __init__(self, xgb_params=None):
        if xgb_params is None:
            xgb_params = {}
        self.xgb_object = xgb.XGBRegressor(**xgb_params)
    def fit(self, X, y):
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        self.xgb_object.fit(X, y)
    def predict(self, X):
        X = X.cpu().numpy()
        return self.xgb_object.predict(X)

class XGBClfWrapper:
    def __init__(self, xgb_params=None):
        if xgb_params is None:
            xgb_params = {}
        self.xgb_object = xgb.XGBClassifier(**xgb_params)
        self.translation_dict = None
        self.retranslation_dict = None
    def fit(self, X, y):
        # Convert one-hot to class labels if needed
        if y.ndim == 2:
            y = y.argmax(-1)
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        unique_labels = np.unique(y)
        self.translation_dict = {l: i for i, l in enumerate(unique_labels)}
        self.retranslation_dict = {i: l for i, l in enumerate(unique_labels)}
        y_enc = np.vectorize(self.translation_dict.get)(y)
        class_counts = Counter(y_enc)
        class_weights = {i: min(class_counts.values()) / class_counts[i] for i in class_counts.keys()}
        class_weights_arr = np.vectorize(class_weights.get)(y_enc)
        self.xgb_object.fit(X, y_enc, sample_weight=class_weights_arr)
    def predict(self, X):
        X = X.cpu().numpy()
        y_pred = self.xgb_object.predict(X)
        return np.vectorize(self.retranslation_dict.get)(y_pred)

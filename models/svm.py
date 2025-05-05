from sklearn.svm import SVC, SVR
import numpy as np

class SVMRegWrapper:
    def __init__(self, svm_params=None):
        if svm_params is None:
            svm_params = {}
        self.svm_object = SVR(**svm_params)
    def fit(self, X, y):
        X = X.cpu().numpy()
        y = y.squeeze(-1).cpu().numpy()
        self.svm_object.fit(X, y)
    def predict(self, X):
        X = X.cpu().numpy()
        return self.svm_object.predict(X)

class SVMClfWrapper:
    def __init__(self, svm_params=None):
        if svm_params is None:
            svm_params = {}
        self.svm_object = SVC(**svm_params)
        self.translation_dict = None
        self.retranslation_dict = None
    def fit(self, X, y):
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        if y.ndim == 2:
            y = y.argmax(-1)
        unique_labels = np.unique(y)
        self.translation_dict = {l: i for i, l in enumerate(unique_labels)}
        self.retranslation_dict = {i: l for i, l in enumerate(unique_labels)}
        y_enc = np.vectorize(self.translation_dict.get)(y)
        self.svm_object.fit(X, y_enc)
    def predict(self, X):
        X = X.cpu().numpy()
        y_pred = self.svm_object.predict(X)
        return np.vectorize(self.retranslation_dict.get)(y_pred)

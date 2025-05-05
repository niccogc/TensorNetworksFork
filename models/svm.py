from sklearn.svm import SVC, SVR
import numpy as np

class SVMRegWrapper:
    def __init__(self, svm_params=None):
        if svm_params is None:
            svm_params = {}
        self.svm_object = SVR(**svm_params)
    def fit(self, X, y):
        self.svm_object.fit(X, y)
    def predict(self, X):
        return self.svm_object.predict(X)

class SVMClfWrapper:
    def __init__(self, svm_params=None):
        if svm_params is None:
            svm_params = {}
        self.svm_object = SVC(**svm_params)
        self.translation_dict = None
        self.retranslation_dict = None
    def fit(self, X, y):
        if y.ndim == 2:
            y = y.argmax(-1)
        self.translation_dict = {l: i for i, l in enumerate(np.unique(y))}
        self.retranslation_dict = {i: l for i, l in enumerate(np.unique(y))}
        y_enc = np.vectorize(self.translation_dict.get)(y)
        self.svm_object.fit(X, y_enc)
    def predict(self, X):
        y_pred = self.svm_object.predict(X)
        return np.vectorize(self.retranslation_dict.get)(y_pred)

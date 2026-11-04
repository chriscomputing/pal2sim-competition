import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sktime.pipeline import make_pipeline
from sktime.transformations.panel.rocket import MiniRocketMultivariate


class MiniRocketClassifier:
    def __init__(self):
        minirocket = MiniRocketMultivariate(num_kernels=1000, random_state=42)

        sgd_logic = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            class_weight='balanced',
            max_iter=1000,
            tol=1e-3,
            random_state=42
        )

        self.rocket_pipeline = make_pipeline(minirocket, StandardScaler(with_mean=False), sgd_logic)

    def train(self, train, val):
        X_train, y_train = train
        X_val, y_val = val
        self.rocket_pipeline.fit(X_train, y_train)
        return self.rocket_pipeline.predict(X_train).to_numpy()

    def predict(self, X: np.ndarray):
        return self.rocket_pipeline.predict(X).to_numpy()

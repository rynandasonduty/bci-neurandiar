import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class ClassicalClassifier:
    def __init__(self, model_type='svm', C=10, n_estimators=100):
        """
        Classical ML classifier wrapper supporting SVM (RBF kernel) and Random Forest.

        StandardScaler is intentionally excluded from this pipeline; normalisation
        is handled centrally by data_utils.py to prevent double standardisation.
        """
        self.model_type = model_type.lower()

        if self.model_type == 'svm':
            # RBF kernel is appropriate for the non-linear EEG feature space
            base_model = SVC(kernel='rbf', C=C, probability=True, random_state=42)
        elif self.model_type == 'rf':
            base_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        else:
            raise ValueError("model_type must be 'svm' or 'rf'")

        self.pipeline = Pipeline([
            ('classifier', base_model)
        ])

    def train(self, X_train, y_train):
        print(f"[INFO] Training {self.model_type.upper()} classifier...")
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_val, y_val):
        y_pred = self.pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        return acc

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)

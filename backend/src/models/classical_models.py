import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class ClassicalClassifier:
    def __init__(self, model_type='svm', C=10, n_estimators=100):
        """
        Inisialisasi Model ML Klasik (SVM atau Random Forest).
        """
        self.model_type = model_type.lower()
        
        if self.model_type == 'svm':
            # Menggunakan RBF kernel karena sinyal EEG sangat non-linear
            base_model = SVC(kernel='rbf', C=C, probability=True, random_state=42)
        elif self.model_type == 'rf':
            base_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        else:
            raise ValueError("Model type harus 'svm' atau 'rf'")
            
        # [PERBAIKAN KRITIS #2] StandardScaler dihapus dari Pipeline.
        # Normalisasi sudah ditangani secara terpusat oleh data_utils.py 
        # agar tidak terjadi double normalization (z-score dari z-score).
        self.pipeline = Pipeline([
            ('classifier', base_model)
        ])

    def train(self, X_train, y_train):
        print(f"[*] Melatih model {self.model_type.upper()}...")
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
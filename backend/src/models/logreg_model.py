import numpy as np
import os
import sys
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment

# 10 target word classes mapped to integer labels (0–9)
WORD_CLASSES = {
    "MAKAN": 0, "MINUM": 1, "BERAK": 2, "PIPIS": 3, "MANDI": 4,
    "BOSAN": 5, "LELAH": 6, "SAKIT": 7, "TIDUR": 8, "SAYANG": 9
}

# Inverse mapping for converting integer predictions back to word strings
REVERSE_WORD_CLASSES = {v: k for k, v in WORD_CLASSES.items()}

class WordAssembler:
    def __init__(self, exp_id=None, pilar="P1_Global", filename=None):
        """
        Logistic Regression word assembler for combining syllable-slot probabilities.

        If exp_id is provided, model artefacts are stored under the corresponding
        Golden Standard experiment directory for the given paradigm (`pilar`,
        default 'P1_Global' for backward compatibility). If None, a simulation
        mode directory is used to prevent errors when the assembler is called
        without experiment context.

        Args:
            exp_id (str or None): Experiment identifier (e.g., 'E5_Data_Augmentation').
            pilar (str): Paradigm label ('P1_Global', 'P2_EEGNet', or 'P3_SVM').
            filename (str or None): Override for the model artefact filename.
                Defaults to 'logreg_assembler_{exp_id}.pkl' if not provided.
        """
        if exp_id:
            paths = setup_experiment(exp_id, pilar=pilar)
            self.model_dir = paths["weights"]
            self.model_path = os.path.join(self.model_dir, filename or f"logreg_assembler_{exp_id}.pkl")
        else:
            # Default/simulation mode: prevents FileNotFoundError when called without context
            self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'weights', 'E_Sim'))
            self.model_path = os.path.join(self.model_dir, "logreg_assembler_sim.pkl")
            os.makedirs(self.model_dir, exist_ok=True)

        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self._is_loaded = False  # Set to True only after a successful load_model() call

    def train(self, X_train, y_train):
        """
        Fit the Logistic Regression model on the training split.

        Args:
            X_train (np.ndarray): Feature matrix of shape (N, 38) — 19 slot-1 probs + 19 slot-2 probs.
            y_train (np.ndarray): Integer word labels in the range [0, 9].

        Returns:
            float: Training set accuracy (reported as a sanity check only).
        """
        print("[INFO] Training Logistic Regression word assembler...")

        self.model.fit(X_train, y_train)

        y_pred_train = self.model.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)

        print(f"[INFO] Word assembler training accuracy: {acc_train * 100:.2f}%")

        return acc_train

    def save_model(self):
        """Serialise the fitted model to disk."""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[INFO] Word assembler model saved to: {self.model_path}")

    def load_model(self):
        """Load a previously serialised model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Train the assembler first.")

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        self._is_loaded = True
        print(f"[INFO] Word assembler model loaded from: {self.model_path}")

    def assemble_word(self, prob_slot1, prob_slot2):
        """
        Real-time inference: decode a word from two syllable-slot probability vectors.

        Args:
            prob_slot1 (np.ndarray): EEGNet output probabilities for slot 1, shape (19,).
            prob_slot2 (np.ndarray): EEGNet output probabilities for slot 2, shape (19,).

        Returns:
            str: Decoded word string from WORD_CLASSES.
        """
        combined_probs = np.concatenate((prob_slot1, prob_slot2))
        X_input = combined_probs.reshape(1, -1)
        pred_idx = self.model.predict(X_input)[0]
        pred_word = REVERSE_WORD_CLASSES[pred_idx]
        return pred_word

    def assemble_word_with_confidence(self, prob_slot1, prob_slot2):
        """
        Same as `assemble_word()`, but additionally returns the assembler's own
        class-probability confidence, so callers can report a meaningful
        word-level confidence instead of the raw slot-classifier's max probability.

        Args:
            prob_slot1 (np.ndarray): Slot 1 syllable probabilities, shape (19,).
            prob_slot2 (np.ndarray): Slot 2 syllable probabilities, shape (19,).

        Returns:
            tuple: (pred_word (str), confidence (float in [0, 1])).
        """
        combined_probs = np.concatenate((prob_slot1, prob_slot2))
        X_input = combined_probs.reshape(1, -1)
        proba = self.model.predict_proba(X_input)[0]
        # np.argmax(proba) is a positional index into self.model.classes_, not
        # necessarily the class label itself — map explicitly to avoid relying
        # on classes_ happening to equal [0..9] in sorted order.
        pred_pos = int(np.argmax(proba))
        pred_label = int(self.model.classes_[pred_pos])
        pred_word = REVERSE_WORD_CLASSES[pred_label]
        confidence = float(proba[pred_pos])
        return pred_word, confidence

if __name__ == "__main__":
    print("=" * 50)
    print(" WORD ASSEMBLER SIMULATION (DRY-RUN) ")
    print("=" * 50)

    assembler = WordAssembler()

    print("[INFO] Generating 1000 simulated syllable probability samples...")
    X_dummy_probs = np.random.rand(1000, 38)
    y_dummy_words = np.random.randint(0, 10, 1000)

    assembler.train(X_dummy_probs, y_dummy_words)
    assembler.save_model()

    print("\n" + "=" * 50)
    print(" REAL-TIME INFERENCE SIMULATION ")
    print("=" * 50)

    dummy_p1 = np.random.rand(19)
    dummy_p2 = np.random.rand(19)

    assembler.load_model()
    predicted = assembler.assemble_word(dummy_p1, dummy_p2)
    print(f"[INFO] Decoded word: {predicted}")

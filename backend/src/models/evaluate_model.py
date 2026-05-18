import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment
from models.eegnet_model import EEGNetClassifier 

SYLLABLE_NAMES = [
    "MA", "KAN", "MI", "NUM", "BE", "RAK", "PI", "PIS", "MAN", "DI", 
    "BO", "SAN", "LE", "LAH", "SA", "KIT", "TI", "DUR", "YANG"
]

WORD_NAMES = [
    "MAKAN", "MINUM", "BERAK", "PIPIS", "MANDI", 
    "BOSAN", "LELAH", "SAKIT", "TIDUR", "SAYANG"
]

def plot_confusion_matrix(y_true, y_pred, classes, title, filepath=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
        print(f"[INFO] {title} saved to: {filepath}")
    plt.close()

def evaluate_system(exp_id="E0_Baseline"):
    """
    Evaluate the end-to-end BCI decoding system on the held-out test set for a given experiment.

    Stage 1 evaluates syllable-level accuracy using the EEGNet classifier.
    Stage 2 evaluates word-level accuracy using the logistic regression word assembler.

    Args:
        exp_id (str): Experiment identifier (e.g., 'E0_Baseline').
    """
    print(f"[INFO] Starting system evaluation for experiment: {exp_id}")

    paths = setup_experiment(exp_id)
    weights_dir = paths["weights"]
    processed_dir = paths["processed_data"]

    # 1. Load trained models
    print("[INFO] Loading EEGNet classifier and word assembler...")
    try:
        eegnet_path = os.path.join(weights_dir, f"eegnet_trained_{exp_id}.h5")
        eegnet = load_model(eegnet_path)

        logreg_path = os.path.join(weights_dir, f"logreg_assembler_{exp_id}.pkl")
        with open(logreg_path, 'rb') as f:
            word_assembler = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load models for '{exp_id}': {e}")
        return

    # 2. Load held-out test sets
    print("[INFO] Loading test set arrays from experiment directory...")
    try:
        X_eeg = np.load(os.path.join(processed_dir, "X_test.npy"))
        y_syl = np.load(os.path.join(processed_dir, "y_test.npy"))

        X_word = np.load(os.path.join(processed_dir, "X_word_test.npy"))
        y_word = np.load(os.path.join(processed_dir, "y_word_test.npy"))
    except Exception as e:
        print(f"[ERROR] Failed to load test data. Ensure train_pipeline and build_logreg_dataset have been run. {e}")
        return

    # ---------------------------------------------------------
    # Stage 1: EEGNet syllable classification
    # ---------------------------------------------------------
    print("\n[INFO] Stage 1: Syllable-level evaluation (EEGNet).")

    start_time = time.perf_counter()
    prob_syl = eegnet.predict(X_eeg, verbose=0)
    end_time = time.perf_counter()

    y_pred_syl = np.argmax(prob_syl, axis=1)
    acc_syllable = accuracy_score(y_syl, y_pred_syl)
    avg_lat_eeg = ((end_time - start_time) / len(X_eeg)) * 1000

    print(f"  Syllable samples : {len(X_eeg)}")
    print(f"  Syllable accuracy: {acc_syllable * 100:.2f}%")
    print(f"  Mean latency     : {avg_lat_eeg:.2f} ms/sample")

    plot_confusion_matrix(y_syl, y_pred_syl, SYLLABLE_NAMES,
                          f"Syllable Confusion Matrix ({exp_id})", filepath=None)

    # ---------------------------------------------------------
    # Stage 2: Logistic regression word assembly
    # ---------------------------------------------------------
    print("\n[INFO] Stage 2: Word-level evaluation (Logistic Regression).")

    start_total = time.perf_counter()
    y_pred_word = word_assembler.predict(X_word)
    end_total = time.perf_counter()

    acc_word = accuracy_score(y_word, y_pred_word)
    avg_lat_word = ((end_total - start_total) / len(X_word)) * 1000

    print(f"  Word samples     : {len(X_word)}")
    print(f"  Word accuracy    : {acc_word * 100:.2f}%")
    print(f"  Mean latency     : {avg_lat_word:.2f} ms/sample")

    plot_confusion_matrix(y_word, y_pred_word, WORD_NAMES,
                          f"Word Confusion Matrix ({exp_id})", filepath=None)

    print(f"\n[INFO] Evaluation complete for experiment: {exp_id}")

if __name__ == "__main__":
    evaluate_system(exp_id="E0_Baseline")
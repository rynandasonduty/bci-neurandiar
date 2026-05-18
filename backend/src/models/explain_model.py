import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import shap
import tensorflow as tf

# Required for SHAP GradientExplainer compatibility with TensorFlow
tf.experimental.numpy.experimental_enable_numpy_behavior()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment

def run_explainability(exp_id="E0_Baseline"):
    """
    Compute and visualize SHAP feature importance for the EEGNet model of a given experiment.

    Uses GradientExplainer with the scaled test set as both background reference and
    test inputs. The resulting mean absolute SHAP values are displayed as a
    channel-by-time heatmap.

    Args:
        exp_id (str): Experiment identifier (e.g., 'E0_Baseline').
    """
    print(f"[INFO] Starting SHAP explainability analysis for experiment: {exp_id}")

    paths = setup_experiment(exp_id)
    weights_dir = paths["weights"]
    processed_dir = paths["processed_data"]

    print("[INFO] Loading EEGNet model and test tensor...")
    model_path = os.path.join(weights_dir, f"eegnet_trained_{exp_id}.h5")
    if not os.path.exists(model_path):
        print("[ERROR] Model file not found.")
        return

    model = load_model(model_path)

    data_path = os.path.join(processed_dir, "X_test.npy")
    if not os.path.exists(data_path):
        print("[ERROR] X_test.npy not found.")
        return

    X = np.load(data_path)

    # Use first 50 samples as SHAP background reference; next 50 as test inputs.
    # A minimum of 50 test samples is required for reliable mean |SHAP| estimates.
    n_background = min(50, len(X) // 2)
    n_test       = min(50, len(X) - n_background)
    background   = X[:n_background]
    test_samples = X[n_background : n_background + n_test]
    print(f"[INFO] SHAP configuration — background: {n_background} samples, test: {n_test} samples.")

    print("[INFO] Initializing SHAP GradientExplainer...")
    explainer = shap.GradientExplainer(model, background)

    print("[INFO] Computing SHAP values...")
    try:
        shap_values = explainer.shap_values(test_samples)

        shap_array = np.array(shap_values)

        # Reduce to 2D (Channels x Time) by averaging over class and sample dimensions
        if isinstance(shap_values, list):
            # Shape: (19 classes, 3 samples, 14 channels, T timesteps, 1 depth)
            shap_mean = np.mean(np.abs(shap_array), axis=(0, 1, 4))
        else:
            shap_abs = np.abs(shap_array)
            shap_mean = np.mean(shap_abs, axis=tuple([i for i in range(shap_abs.ndim) if i not in [1, 2]]))

        # Ensure orientation: rows = channels, columns = time
        n_channels = X.shape[1]
        if shap_mean.shape[0] != n_channels:
            shap_mean = shap_mean.T

        plt.figure(figsize=(12, 6))
        sns.heatmap(shap_mean, cmap="viridis", cbar_kws={'label': 'Mean |SHAP value|'})
        plt.title(f"SHAP Feature Importance (Channels x Time) — {exp_id}")
        plt.ylabel("EEG Channels")
        plt.xlabel("Time Samples")

        plt.tight_layout()
        plt.show()
        plt.close()

    except Exception as e:
        print(f"[ERROR] SHAP computation failed: {e}")

    print(f"[INFO] SHAP analysis complete for experiment: {exp_id}")

if __name__ == "__main__":
    run_explainability()
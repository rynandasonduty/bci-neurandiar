import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.extract_eeg_features import EEGFeatureExtractor
from models.classical_models import ClassicalClassifier
from sklearn.preprocessing import StandardScaler

def calibrate_new_user(base_model_path, X_new_3d, y_new, new_subject_id, save_dir,
                       champion_type="eegnet", feat_group="all"):
    """
    Personalise a pretrained model to a new user via transfer learning or fast retraining.

    For EEGNet models, all layers except the final Dense and softmax are frozen and
    the model is fine-tuned for 15 epochs on the new user's data. For SVM models,
    features are extracted from scratch and a new classifier is fitted directly.

    Args:
        base_model_path (str): Path to the champion model artefact (.h5 or .pkl).
        X_new_3d (np.ndarray): New user EEG data, shape (N, 14, 256, 1).
        y_new (np.ndarray): Ground-truth labels for the new user's data.
        new_subject_id (str): Unique identifier for the new user (e.g., 'S_New_01').
        save_dir (str): Directory in which to save the personalised model artefact.
        champion_type (str): Model architecture — 'eegnet' or 'svm'.
        feat_group (str): Feature group configuration for SVM (e.g., 'time', 'all').

    Returns:
        tuple: (save_path, champion_type) — path to the saved personalised model.
    """
    print(f"[INFO] Starting calibration for new user: {new_subject_id}...")
    os.makedirs(save_dir, exist_ok=True)

    # =====================================================================
    # BRANCH 1: DEEP LEARNING (EEGNet) — Transfer Learning
    # =====================================================================
    if champion_type.lower() == "eegnet":
        print("[INFO] Architecture: EEGNet. Executing transfer learning protocol.")

        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Base EEGNet model not found at: {base_model_path}")

        model = load_model(base_model_path)

        # Freeze all layers except the final classification head
        for layer in model.layers:
            if 'dense' not in layer.name.lower() and 'softmax' not in layer.name.lower():
                layer.trainable = False

        # Recompile with a low learning rate to preserve pretrained representations
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print(f"[INFO] Fine-tuning classification head on {len(X_new_3d)} new user samples...")
        model.fit(X_new_3d, y_new, epochs=15, batch_size=2, verbose=1)

        save_path = os.path.join(save_dir, f"calibrated_EEGNet_{new_subject_id}.h5")
        model.save(save_path)
        print(f"[INFO] Personalised EEGNet model saved to: {save_path}")

        return save_path, "eegnet"

    # =====================================================================
    # BRANCH 2: CLASSICAL ML (SVM) — Fast Retraining
    # =====================================================================
    elif champion_type.lower() == "svm":
        print("[INFO] Architecture: SVM. Executing fast retraining protocol.")

        # Remove the depth dimension: (N, 14, 256, 1) → (N, 14, 256)
        X_new_2d_raw = np.squeeze(X_new_3d, axis=-1)

        print(f"[INFO] Extracting '{feat_group}' features from {len(X_new_2d_raw)} samples...")
        extractor = EEGFeatureExtractor(fs=256)
        groups = None if feat_group == 'all' else [feat_group]

        X_features = extractor.transform(X_new_2d_raw, groups=groups)
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit a new scaler exclusively on the new user's data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)

        scaler_path = os.path.join(save_dir, f"calibrated_scaler_{new_subject_id}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        print("[INFO] Training personalised SVM classifier...")
        svm_model = ClassicalClassifier(model_type='svm', C=10)
        svm_model.train(X_scaled, y_new)

        save_path = os.path.join(save_dir, f"calibrated_SVM_{new_subject_id}.pkl")
        svm_model.save_model(save_path)

        print(f"[INFO] Personalised SVM model saved to: {save_path}")
        print(f"[INFO] Personalised scaler saved to: {scaler_path}")

        return save_path, "svm"

    else:
        raise ValueError("champion_type must be 'eegnet' or 'svm'")

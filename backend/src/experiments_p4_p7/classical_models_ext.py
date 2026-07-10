"""
backend/src/experiments_p4_p7/classical_models_ext.py

P7 coarse sub-model improvement, Varian A (class-weight balanced). The coarse
stage's vowel groups are unbalanced (A={MA,MAN,SA}=3, I={MI,PI,TI}=3,
E={BE,LE}=2, O={BO}=1), which is a plausible source of the coarse stage's
bottleneck accuracy in the P7 pipeline.

`models/classical_models.py` (ClassicalClassifier) is one of the isolation-
protected files and must never be modified. WeightedClassicalClassifier
subclasses it instead, overriding ONLY __init__ to add class_weight='balanced'
to the SVC constructor -- train/evaluate/save_model/load_model are inherited
unchanged from the original.
"""
import os
import sys

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.classical_models import ClassicalClassifier


class WeightedClassicalClassifier(ClassicalClassifier):
    """Identical to ClassicalClassifier, except the SVC uses
    class_weight='balanced' to counter the coarse stage's unbalanced
    vowel-group class sizes. SVM-only (class_weight is SVC-specific)."""

    def __init__(self, model_type='svm', C=10, n_estimators=100):
        if model_type.lower() != 'svm':
            raise ValueError(
                "WeightedClassicalClassifier only supports model_type='svm' "
                "(class_weight='balanced' is an SVC-specific parameter)."
            )
        self.model_type = model_type.lower()
        base_model = SVC(kernel='rbf', C=C, probability=True, random_state=42, class_weight='balanced')
        self.pipeline = Pipeline([('classifier', base_model)])

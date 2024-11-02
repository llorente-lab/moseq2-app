import cv2
import h5py
import pickle
import joblib
import numpy as np
from collections import defaultdict
from typing import Union
from pathlib import Path
from sklearn.svm import SVC
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from moseq2_extract.extract.proc import clean_frames
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer


@dataclass
class CleanParameters:
    prefilter_space: tuple = (3,)
    prefilter_time: Optional[tuple] = None
    strel_tail: Tuple[int, int] = (9, 9)
    strel_min: Tuple[int, int] = (5, 5)
    iters_tail: Optional[int] = 1
    iters_min: Optional[int] = None
    height_threshold: int = 10


def create_training_dataset(
    data_index_path: str,
    clean_parameters: Optional[CleanParameters] = None,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[str, dict]:
    """Create training dataset while holding out validation frames.
    Returns path to training data and dictionary of validation frames."""
    np.random.seed(random_state)
    data_index_path = Path(data_index_path)
    out_path = data_index_path.with_name("training_data.npz")
    if clean_parameters is None:
        clean_parameters = CleanParameters()

    # load trainingdata index
    with open(data_index_path, "rb") as f:
        session_paths, data_index = pickle.load(f)

    # Dictionary to store validation frames for each session
    validation_frames = {}
    training_index = defaultdict(list)

    # For each session, hold out some frames for validation
    for session_name, ranges in data_index.items():
        n_ranges = len(ranges)
        n_val = max(
            1, int(n_ranges * validation_size)
        )  # At least 1 range for validation

        # Randomly select ranges for validation
        val_indices = np.random.choice(n_ranges, n_val, replace=False)
        val_ranges = [ranges[i] for i in val_indices]
        train_ranges = [r for i, r in enumerate(ranges) if i not in val_indices]

        # Store validation ranges
        validation_frames[session_name] = val_ranges
        # Store training ranges
        training_index[session_name] = train_ranges

    # Process training frames as before
    frames = []
    for k, v in training_index.items():
        with h5py.File(session_paths[k], "r") as h5f:
            for left, _slice in v:
                frames_subset = h5f["frames"][_slice]
                if left:
                    frames_subset = np.rot90(frames_subset, 2, axes=(1, 2))
                frames.append(frames_subset)

    frames = np.concatenate(frames, axis=0)
    frames = np.concatenate((frames, np.rot90(frames, 2, axes=(1, 2))), axis=0)

    flipped = np.zeros((len(frames),), dtype=np.uint8)
    flipped[len(frames) // 2 :] = 1

    # Add augmentations as before
    shifts = np.random.randint(-5, 5, size=(len(frames), 2))
    shifted_frames = np.array(
        [np.roll(f, tuple(s), axis=(0, 1)) for f, s in zip(frames, shifts)]
    ).astype(np.uint8)

    cleaned_frames = clean_frames(
        np.where(frames > clean_parameters.height_threshold, frames, 0),
        clean_parameters.prefilter_space,
        clean_parameters.prefilter_time,
        strel_tail=cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, clean_parameters.strel_tail
        ),
        iters_tail=clean_parameters.iters_tail,
        strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, clean_parameters.strel_min),
        iters_min=clean_parameters.iters_min,
    )

    frames = np.concatenate((frames, shifted_frames, cleaned_frames), axis=0)
    flipped = np.concatenate((flipped, flipped, flipped), axis=0)

    # Save training data
    np.savez(out_path, frames=frames, flipped=flipped)

    # Save validation info
    val_path = data_index_path.with_name("validation_ranges.pkl")
    with open(val_path, "wb") as f:
        pickle.dump((session_paths, validation_frames), f)

    return out_path, validation_frames.resolve()


def flatten(array: np.ndarray) -> np.ndarray:
    return array.reshape(len(array), -1)


def batch_apply_pca(frames: np.ndarray, pca: PCA, batch_size: int = 1000) -> np.ndarray:
    output = []
    if len(frames) < batch_size:
        return pca.transform(flatten(frames)).astype(np.float32)

    for arr in np.array_split(frames, len(frames) // batch_size):
        output.append(pca.transform(flatten(arr)).astype(np.float32))
    return np.concatenate(output, axis=0).astype(np.float32)


def train_classifier(
    data_path: str,
    classifier: str = "SVM",
    n_components: int = 20,
    test_size: float = 0.1,
    random_state: int = 0,
    # Random Forest parameters
    rf_n_estimators: int = 150,
    rf_max_depth: Optional[int] = None,
    rf_min_samples_split: int = 2,
    rf_min_samples_leaf: int = 1,
    rf_max_features: Union[str, int, float] = "sqrt",
    # SVM parameters
    svm_C: float = 1.0,
    svm_kernel: str = "rbf",
    svm_gamma: Union[str, float] = "scale",
    svm_class_weight: Optional[Union[dict, str]] = None,
    # Cross validation parameters
    cv_splits: int = 4,
):
    """Train a classifier to predict the orientation of a mouse.

    Parameters:
        data_path (str): Path to the training data numpy file.
        classifier (str): Classifier to use. Either 'SVM' or 'RF'.
        n_components (int): Number of components to keep in PCA.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Random state for reproducibility.

        # Random Forest Parameters
        rf_n_estimators (int): Number of trees in random forest.
        rf_max_depth (Optional[int]): Maximum depth of the tree.
        rf_min_samples_split (int): Minimum samples required to split an internal node.
        rf_min_samples_leaf (int): Minimum samples required to be at a leaf node.
        rf_max_features (Union[str, int, float]): Number of features to consider for best split.

        # SVM Parameters
        svm_C (float): Regularization parameter.
        svm_kernel (str): Kernel type to be used in the algorithm.
        svm_gamma (Union[str, float]): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        svm_class_weight (Optional[Union[dict, str]]): Class weights.

        # Cross Validation Parameters
        cv_splits (int): Number of folds for cross-validation.

    Returns:
        sklearn.pipeline.Pipeline: Trained classifier pipeline
    """
    data = np.load(data_path)
    frames = data["frames"]
    flipped = data["flipped"]

    print("Fitting PCA")
    pca = PCA(n_components=n_components)
    pca.fit(flatten(frames[-len(frames) // 3 :]))

    # Create the classifier based on input parameters
    if classifier == "RF":
        model = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            max_features=rf_max_features,
            random_state=random_state,
        )
    else:  # SVM
        model = SVC(
            C=svm_C,
            kernel=svm_kernel,
            gamma=svm_gamma,
            probability=True,
            class_weight=svm_class_weight,
            random_state=random_state,
        )

    pipeline = make_pipeline(
        FunctionTransformer(batch_apply_pca, kw_args={"pca": pca}, validate=False),
        StandardScaler(),
        model,
    )

    print("Running cross-validation")
    accuracy = cross_val_score(
        pipeline,
        frames,
        flipped,
        cv=KFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
    )
    print(f"Held-out model accuracy: {accuracy.mean():.3f} ± {accuracy.std():.3f}")

    print("Final fitting step")
    return pipeline.fit(frames, flipped)


def save_classifier(clf_pipeline, out_path: str):
    joblib.dump(clf_pipeline, out_path)
    print(f"Classifier saved to {out_path}")

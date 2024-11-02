import unittest
from unittest import mock
from unittest.mock import MagicMock, mock_open, patch
from pathlib import Path
import tempfile
import os
import numpy as np
import h5py
import pickle
import joblib
import pytest

from moseq2_app.flip.train import *
from moseq2_app.flip.widget import *


# ----------------------------
# Test for extraction_complete
# ----------------------------
def test_extraction_complete_success():
    mock_yaml_content = "complete: true\nother_key: value"
    with patch("pathlib.Path.read_text", return_value=mock_yaml_content):
        file_path = Path("dummy.yaml")
        assert extraction_complete(file_path) is True


def test_extraction_complete_failure():
    mock_yaml_content = "other_key: value"
    with patch("pathlib.Path.read_text", return_value=mock_yaml_content):
        file_path = Path("dummy.yaml")
        assert extraction_complete(file_path) is False


def test_extraction_complete_invalid_yaml():
    mock_yaml_content = ":::invalid_yaml:::"
    with patch("pathlib.Path.read_text", return_value=mock_yaml_content):
        file_path = Path("dummy.yaml")
        assert extraction_complete(file_path) is False


# ----------------------------
# Test for find_extractions
# ----------------------------


def test_find_extractions_no_duplicates():
    mock_files = [
        Path("session1/file1.h5"),
        Path("session2/file2.h5"),
        Path("session3/file3.h5"),
    ]
    with patch("pathlib.Path.glob", return_value=mock_files):
        with patch("__main__._extraction_complete", return_value=True):
            data_path = "dummy_data_path"
            result = find_extractions(data_path)
            expected = {f.name: f for f in mock_files}
            assert result == expected


def test_find_extractions_with_duplicates():
    mock_files = [
        Path("session1/file1.h5"),
        Path("session2/file1.h5"),
        Path("session3/file2.h5"),
    ]
    with patch("pathlib.Path.glob", return_value=mock_files):
        with patch("__main__._extraction_complete", return_value=True):
            data_path = "dummy_data_path"
            result = find_extractions(data_path)
            expected = {
                "session1/file1.h5": Path("session1/file1.h5"),
                "session2/file1.h5": Path("session2/file1.h5"),
                "file2.h5": Path("session3/file2.h5"),
            }
            assert result == expected


def test_find_extractions_incomplete_files():
    mock_files = [
        Path("session1/file1.h5"),
        Path("session2/file2.h5"),
        Path("session3/file3.h5"),
    ]

    # Assume only file1 and file3 are complete
    def extraction_complete_side_effect(path):
        return path.stem != "file2"

    with patch("pathlib.Path.glob", return_value=mock_files):
        with patch(
            "__main__._extraction_complete", side_effect=extraction_complete_side_effect
        ):
            data_path = "dummy_data_path"
            result = find_extractions(data_path)
            expected = {
                f.name: f
                for f in [Path("session1/file1.h5"), Path("session3/file3.h5")]
            }
            assert result == expected


# ----------------------------
# Test for CleanParameters
# ----------------------------
def test_clean_parameters_defaults():
    params = CleanParameters()
    assert params.prefilter_space == (3,)
    assert params.prefilter_time is None
    assert params.strel_tail == (9, 9)
    assert params.strel_min == (5, 5)
    assert params.iters_tail == 1
    assert params.iters_min is None
    assert params.height_threshold == 10


def test_clean_parameters_custom():
    custom_params = CleanParameters(
        prefilter_space=(5,),
        prefilter_time=(2,),
        strel_tail=(7, 7),
        strel_min=(3, 3),
        iters_tail=2,
        iters_min=1,
        height_threshold=15,
    )
    assert custom_params.prefilter_space == (5,)
    assert custom_params.prefilter_time == (2,)
    assert custom_params.strel_tail == (7, 7)
    assert custom_params.strel_min == (3, 3)
    assert custom_params.iters_tail == 2
    assert custom_params.iters_min == 1
    assert custom_params.height_threshold == 15


# ----------------------------
# Test for flatten
# ----------------------------
def test_flatten():
    array = np.array([[1, 2], [3, 4]])
    flattened = flatten(array)
    expected = np.array([[1, 2], [3, 4]]).reshape(2, -1)
    np.testing.assert_array_equal(flattened, expected)


def test_flatten_high_dimensional():
    array = np.random.rand(10, 20, 30)
    flattened = flatten(array)
    expected = array.reshape(10, -1)
    np.testing.assert_array_equal(flattened, expected)


# ----------------------------
# Test for batch_apply_pca
# ----------------------------
def test_batch_apply_pca_small_batch():
    frames = np.random.rand(500, 10, 10)
    pca = mock.MagicMock()
    transformed = np.random.rand(500, 5)
    pca.transform.return_value = transformed
    result = batch_apply_pca(frames, pca, batch_size=1000)
    pca.transform.assert_called_once()
    np.testing.assert_array_equal(result, transformed)


def test_batch_apply_pca_large_batch():
    frames = np.random.rand(3000, 10, 10)
    pca = mock.MagicMock()
    transformed_part = np.random.rand(1000, 5)
    pca.transform.side_effect = [transformed_part, transformed_part, transformed_part]
    result = batch_apply_pca(frames, pca, batch_size=1000)
    assert len(pca.transform.call_args_list) == 3
    assert result.shape == (3000, 5)


# ----------------------------
# Test for train_classifier
# ----------------------------
def test_train_classifier_svm(mocker):
    # Mock data
    frames = np.random.rand(100, 64, 64)
    flipped = np.random.randint(0, 2, size=100)
    np.savez = mocker.patch("numpy.savez")
    mock_load = mocker.patch(
        "numpy.load", return_value={"frames": frames, "flipped": flipped}
    )

    # Mock PCA
    with patch("flip_module.PCA") as mock_pca:
        mock_pca_instance = mock_pca.return_value
        mock_pca_instance.transform.side_effect = lambda x: x  # Identity
        mock_pca_instance.fit.return_value = None

        # Mock classifier
        with patch("flip_module.SVC") as mock_svc:
            mock_svc_instance = mock_svc.return_value
            mock_svc_instance.fit.return_value = mock_svc_instance
            mock_svc_instance.predict.return_value = flipped

            # Mock cross_val_score
            with patch(
                "flip_module.cross_val_score",
                return_value=np.array([0.9, 0.92, 0.88, 0.91]),
            ):
                pipeline = train_classifier(
                    data_path="dummy_training_data.npz",
                    classifier="SVM",
                    n_components=10,
                    cv_splits=4,
                )
                assert pipeline is not None
                mock_load.assert_called_once_with("dummy_training_data.npz")
                mock_pca_instance.fit.assert_called_once()
                mock_svc.assert_called_once()
                mock_svc_instance.fit.assert_called_once()
                mock_svc_instance.predict.assert_called()


def test_train_classifier_rf(mocker):
    # Mock data
    frames = np.random.rand(200, 32, 32)
    flipped = np.random.randint(0, 2, size=200)
    np.savez = mocker.patch("numpy.savez")
    mock_load = mocker.patch(
        "numpy.load", return_value={"frames": frames, "flipped": flipped}
    )

    # Mock PCA
    with patch("flip_module.PCA") as mock_pca:
        mock_pca_instance = mock_pca.return_value
        mock_pca_instance.transform.side_effect = lambda x: x  # Identity
        mock_pca_instance.fit.return_value = None

        # Mock RandomForestClassifier
        with patch("flip_module.RandomForestClassifier") as mock_rf:
            mock_rf_instance = mock_rf.return_value
            mock_rf_instance.fit.return_value = mock_rf_instance
            mock_rf_instance.predict.return_value = flipped

            # Mock cross_val_score
            with patch(
                "flip_module.cross_val_score",
                return_value=np.array([0.85, 0.87, 0.83, 0.86]),
            ):
                pipeline = train_classifier(
                    data_path="dummy_training_data.npz",
                    classifier="RF",
                    n_components=15,
                    cv_splits=4,
                    rf_n_estimators=100,
                )
                assert pipeline is not None
                mock_load.assert_called_once_with("dummy_training_data.npz")
                mock_pca_instance.fit.assert_called_once()
                mock_rf.assert_called_once_with(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    random_state=0,
                )
                mock_rf_instance.fit.assert_called_once()
                mock_rf_instance.predict.assert_called()


# ----------------------------
# Test for save_classifier
# ----------------------------
def test_save_classifier(mocker):
    clf_pipeline = mock.MagicMock()
    mock_joblib_dump = mocker.patch("joblib.dump")
    out_path = "classifier.pkl"
    save_classifier(clf_pipeline, out_path)
    mock_joblib_dump.assert_called_once_with(clf_pipeline, out_path)


# ----------------------------
# Test for FlipClassifierWidget
# ----------------------------
def test_flip_classifier_widget_initialization(mocker):
    # Mock _find_extractions to return a dummy sessions dict
    mock_sessions = {
        "session1.h5": Path("session1.h5"),
        "session2.h5": Path("session2.h5"),
    }
    mocker.patch("__main__._find_extractions", return_value=mock_sessions)

    # Mock Path.mkdir to prevent actual directory creation
    with patch.object(Path, "mkdir") as mock_mkdir:
        widget = FlipClassifierWidget(data_path="dummy_data_path")
        assert widget.data_path == Path("dummy_data_path")
        assert widget.flip_dir == Path("dummy_data_path") / "flip_classifier"
        assert (
            widget.training_data_path
            == Path("dummy_data_path") / "flip_classifier" / "training_data"
        )
        assert (
            widget.model_path == Path("dummy_data_path") / "flip_classifier" / "models"
        )
        mock_mkdir.assert_called()
        assert widget.sessions == mock_sessions
        assert widget.curr_total_selected_frames == 0
        assert widget.selected_frame_ranges_dict == defaultdict(list)


def test_flip_classifier_widget_start_stop_range(mocker):
    mock_sessions = {"session1.h5": Path("session1.h5")}
    mocker.patch("__main__._find_extractions", return_value=mock_sessions)
    with patch.object(Path, "mkdir"):
        widget = FlipClassifierWidget(data_path="dummy_data_path")

    # Mock frame_num_slider value
    widget.frame_num_slider.value = 10

    # Start range
    widget.start_stop_frame_range(event=None)
    assert widget.start_button.name == "Cancel Select"
    assert widget.start_button.button_type == "danger"
    assert widget.face_left_button.visible is True
    assert widget.face_right_button.visible is True
    assert widget.facing_info.visible is True
    assert widget.start == 10

    # Cancel range
    widget.start_stop_frame_range(event=None)
    assert widget.start_button.name == "Start Range"
    assert widget.start_button.button_type == "primary"
    assert widget.face_left_button.visible is False
    assert widget.face_right_button.visible is False
    assert widget.facing_info.visible is False


def test_flip_classifier_widget_facing_callback(mocker):
    mock_sessions = {"session1.h5": Path("session1.h5")}
    mocker.patch("__main__._find_extractions", return_value=mock_sessions)
    with patch.object(Path, "mkdir"):
        widget = FlipClassifierWidget(data_path="dummy_data_path")

    widget.start = 5
    widget.frame_num_slider.value = 15
    widget.selected_frame_ranges_dict = defaultdict(list)

    # Simulate facing left
    widget.facing_range_callback(event=None, left=True)
    assert widget.selected_frame_ranges_dict["session1.h5"] == [(True, range(5, 15))]
    assert widget.curr_total_selected_frames == 10
    assert widget.selected_ranges.options == ["L - range(5, 15) - session1.h5"]

    # Simulate facing right
    widget.start = 20
    widget.frame_num_slider.value = 25
    widget.facing_range_callback(event=None, left=False)
    assert widget.selected_frame_ranges_dict["session1.h5"] == [
        (True, range(5, 15)),
        (False, range(20, 25)),
    ]
    assert widget.curr_total_selected_frames == 15
    assert widget.selected_ranges.options == [
        "L - range(5, 15) - session1.h5",
        "R - range(20, 25) - session1.h5",
    ]


def test_flip_classifier_widget_delete_selection():
    mock_sessions = {"session1.h5": Path("session1.h5")}
    with patch("__main__._find_extractions", return_value=mock_sessions):
        with patch.object(Path, "mkdir"):
            widget = FlipClassifierWidget(data_path="dummy_data_path")

    # Setup selected ranges
    widget.selected_frame_ranges_dict["session1.h5"] = [
        (True, range(5, 15)),
        (False, range(20, 25)),
    ]
    widget.curr_total_selected_frames = 15
    widget.selected_ranges.options = [
        "L - range(5, 15) - session1.h5",
        "R - range(20, 25) - session1.h5",
    ]

    # Select the first range to delete
    widget.selected_ranges.value = ["L - range(5, 15) - session1.h5"]
    widget.on_delete_selection_clicked(event=None)

    assert widget.selected_frame_ranges_dict["session1.h5"] == [(False, range(20, 25))]
    assert widget.curr_total_selected_frames == 10
    assert widget.selected_ranges.options == ["R - range(20, 25) - session1.h5"]


# ----------------------------
# Test for DisplayWidget
# ----------------------------
def test_display_widget_initialization(mocker):
    # Mock joblib.load to return a dummy classifier
    mock_classifier = mock.MagicMock()
    mocker.patch("joblib.load", return_value=mock_classifier)

    # Mock pickle.load to return dummy session_paths and validation_ranges
    mock_session_paths = {"session1.h5": "path/to/session1.h5"}
    mock_validation_ranges = {"session1.h5": [(True, range(0, 10))]}
    mocker.patch(
        "builtins.open",
        mock_open(read_data=pickle.dumps((mock_session_paths, mock_validation_ranges))),
    )

    with patch("h5py.File") as mock_h5:
        # Mock h5py File object
        mock_file = mock.MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file["frames"].shape = (100, 64, 64)
        mock_file["frames"].__getitem__.return_value = np.random.rand(10, 64, 64)
        mock_h5.return_value = mock_file

        widget = DisplayWidget(
            data_path="dummy_data_path",
            classifier_path="dummy_classifier.pkl",
            validation_ranges_path="dummy_validation.pkl",
        )

        assert widget.data_path == Path("dummy_data_path")
        mock_classifier.predict.assert_not_called()  # Not called during initialization
        assert widget.session_select.options == ["session1.h5"]
        assert widget.session_select.value == "session1.h5"
        mock_h5.assert_called_with("path/to/session1.h5", mode="r")


def test_display_widget_load_session(mocker):
    # Mock joblib.load to return a dummy classifier
    mock_classifier = mock.MagicMock()
    mock_classifier.predict.return_value = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    mocker.patch("joblib.load", return_value=mock_classifier)

    # Mock pickle.load to return dummy session_paths and validation_ranges
    mock_session_paths = {"session1.h5": "path/to/session1.h5"}
    mock_validation_ranges = {"session1.h5": [(True, range(0, 10))]}
    mocker.patch(
        "builtins.open",
        mock_open(read_data=pickle.dumps((mock_session_paths, mock_validation_ranges))),
    )

    with patch("h5py.File") as mock_h5:
        # Mock h5py File object
        mock_file = mock.MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file["frames"].shape = (100, 64, 64)
        mock_file["frames"].__getitem__.return_value = np.random.rand(10, 64, 64)
        mock_h5.return_value = mock_file

        widget = DisplayWidget(
            data_path="dummy_data_path",
            classifier_path="dummy_classifier.pkl",
            validation_ranges_path="dummy_validation.pkl",
        )

        # Check if frames are loaded and corrected_frames are set
        assert hasattr(widget, "real_frames")
        assert hasattr(widget, "corrected_frames")
        assert widget.real_frames.shape == (10, 64, 64)
        assert widget.corrected_frames.shape == (10, 64, 64)
        mock_classifier.predict.assert_called_once_with(widget.real_frames)


# ----------------------------
# Run tests if this script is executed
# ----------------------------
if __name__ == "__main__":
    pytest.main([__file__])

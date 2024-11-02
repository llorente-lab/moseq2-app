import h5py
import pickle
import panel as pn
import holoviews as hv
import ruamel.yaml as yaml
from pathlib import Path
import joblib
from collections import defaultdict
import numpy as np


def extraction_complete(file_path: Path):
    try:
        config = yaml.safe_load(file_path.read_text())
        return config["complete"]
    except:
        return False
        print("Please check your config file, the complete key was not found")


def find_extractions(data_path: str):
    files = Path(data_path).glob("**/*.h5")
    files = sorted(f for f in files if extraction_complete(f.with_suffix(".yaml")))
    if len(set([f.name for f in files])) < len(files):
        files = {f.parent.name + "/" + f.name: f for f in files}
    else:
        files = {f.name: f for f in files}
    return files


class FlipClassifierWidget:
    def __init__(self, data_path: str, flip_path: str = "flip_classifier"):
        # ==== Setup directory structure
        self.data_path = Path(data_path)
        self.flip_dir = self.data_path / flip_path
        self.flip_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_path = self.flip_dir / "training_data"
        self.model_path = self.flip_dir / "models"
        self.training_data_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_path.resolve()
        # ====

        self.sessions = find_extractions(data_path)
        # self.selected_frame_ranges_dict = {k: [] for k in self.path_dict}
        self.selected_frame_ranges_dict = defaultdict(list)
        self.curr_total_selected_frames = 0

        self.session_select_dropdown = pn.widgets.Select(
            options=list(self.sessions), name="Session", value=list(self.sessions)[1]
        )
        self.frame_num_slider = pn.widgets.IntSlider(
            name="Current Frame", start=0, end=1000, step=1, value=1
        )
        self.start_button = pn.widgets.Button(name="Start Range", button_type="primary")
        self.face_left_button = pn.widgets.Button(
            name="Facing Left", button_type="success", width=140, visible=False
        )
        self.face_right_button = pn.widgets.Button(
            name="Facing Right", button_type="success", width=140, visible=False
        )
        self.selected_ranges = pn.widgets.MultiSelect(
            name="Selected Ranges", options=[]
        )
        self.delete_selection_button = pn.widgets.Button(
            name="Delete Selection", button_type="danger"
        )
        self.curr_total_label = pn.pane.Markdown(
            f"Current Total Selected Frames: {self.curr_total_selected_frames}"
        )

        self.facing_info = pn.pane.Markdown(
            "To finish selection, click on direction the animal is facing",
            visible=False,
        )
        self.facing_row = pn.Row(self.face_left_button, self.face_right_button)
        self.range_box = pn.Column(
            self.curr_total_label,
            pn.pane.Markdown("#### Selected Correct Frame Ranges"),
            self.selected_ranges,
            self.delete_selection_button,
        )

        self.forward_button = pn.widgets.Button(
            name="Forward", button_type="primary", width=142
        )
        self.backward_button = pn.widgets.Button(
            name="Backward", button_type="primary", width=142
        )

        self.frame_advancer_row = pn.Row(self.backward_button, self.forward_button)

        self.widgets = pn.Column(
            self.session_select_dropdown,
            self.frame_num_slider,
            self.start_button,
            self.frame_advancer_row,
            self.facing_info,
            self.facing_row,
            self.range_box,
            width=325,
        )

        self.frame_display = hv.DynamicMap(
            self.display_frame, streams=[self.frame_num_slider.param.value]
        ).opts(
            frame_width=400, frame_height=400, aspect="equal", xlim=(0, 1), ylim=(0, 1)
        )

        self.start_button.on_click(self.start_stop_frame_range)
        self.face_left_button.on_click(
            lambda event: self.facing_range_callback(event, True)
        )
        self.face_right_button.on_click(
            lambda event: self.facing_range_callback(event, False)
        )
        self.delete_selection_button.on_click(self.on_delete_selection_clicked)

        self.forward_button.on_click(self.advance_frame)
        self.backward_button.on_click(self.rewind_frame)

        self.session_select_dropdown.param.watch(self.changed_selected_session, "value")
        self.session_select_dropdown.value = list(self.sessions)[0]

    def display_frame(self, value):
        if hasattr(self, "frames"):
            frame = self.frames[value]
        else:
            frame = None
        # set bounds for the image
        return hv.Image(frame, bounds=(0, 0, 1, 1)).opts(cmap="cubehelix")

    def advance_frame(self, event):
        if self.frame_num_slider.value < self.frame_num_slider.end:
            self.frame_num_slider.value += 1

    def rewind_frame(self, event):
        if self.frame_num_slider.value > 0:
            self.frame_num_slider.value -= 1

    def start_stop_frame_range(self, event):
        if self.start_button.name == "Start Range":
            self.start = self.frame_num_slider.value
            self.start_button.name = "Cancel Select"
            self.start_button.button_type = "danger"
            self.face_left_button.visible = True
            self.face_right_button.visible = True
            self.facing_info.visible = True
            self.facing_info.object = f"To finish selection, click on direction the animal is facing. Start: {self.start}"
        else:
            self.start_button.name = "Start Range"
            self.start_button.button_type = "primary"
            self.face_left_button.visible = False
            self.face_right_button.visible = False
            self.facing_info.visible = False

    def facing_range_callback(self, event, left):
        self.stop = self.frame_num_slider.value
        if self.stop > self.start:
            self.update_state_on_selected_range(left)
            self.face_left_button.visible = False
            self.face_right_button.visible = False
            self.facing_info.visible = False
            self.start_button.name = "Start Range"
            self.start_button.button_type = "primary"

    def update_state_on_selected_range(self, left):
        selected_range = range(self.start, self.stop)

        beginning = "L" if left else "R"
        display_selected_range = (
            f"{beginning} - {selected_range} - {self.session_select_dropdown.value}"
        )

        self.curr_total_selected_frames += len(selected_range)
        self.curr_total_label.object = (
            f"Current Total Selected Frames: {self.curr_total_selected_frames}"
        )

        self.selected_frame_ranges_dict[self.session_select_dropdown.value].append(
            (left, selected_range)
        )

        self.selected_ranges.options = self.selected_ranges.options + [
            display_selected_range
        ]
        self.selected_ranges.value = []

        self.save_frame_ranges()

    def on_delete_selection_clicked(self, event):
        selected_range = self.selected_ranges.value
        if selected_range:
            vals = selected_range[0].split(" - ")
            delete_key = vals[2]
            direction = vals[0] == "L"
            range_to_delete = eval(vals[1])

            to_drop = (direction, range_to_delete)
            self.selected_frame_ranges_dict[delete_key].remove(to_drop)

            self.curr_total_selected_frames -= len(range_to_delete)
            self.curr_total_label.object = (
                f"Current Total Selected Frames: {self.curr_total_selected_frames}"
            )

            l = list(self.selected_ranges.options)
            l.remove(selected_range[0])

            self.selected_ranges.options = l
            self.selected_ranges.value = []

    def changed_selected_session(self, event):
        if self.start_button.name == "Cancel Select":
            self.start_button.name = "Start Range"
            self.start_button.button_type = "primary"

        self.frame_num_slider.value = 1
        self.widgets.loading = True

        with h5py.File(self.sessions[event.new], mode="r") as f:
            self.frame_num_slider.end = f["frames"].shape[0] - 1
            self.frames = f["frames"][()]

        self.widgets.loading = False

        self.frame_num_slider.value = 0

    def show(self):
        return pn.Row(self.widgets, self.frame_display)

    def save_frame_ranges(self):
        with open(self.train_file, "wb") as f:
            pickle.dump((self.sessions, dict(self.selected_frame_ranges_dict)), f)

    @property
    def train_file(self):
        return str(self.training_data_path.resolve() / "flip-training-frame-ranges.p")


class DisplayWidget:
    def __init__(
        self, data_path: str, classifier_path: str, validation_ranges_path: str
    ):
        """
        Widget to display original and classifier-corrected validation frames.

        Parameters:
            data_path: Path to directory containing h5 files
            classifier_path: Path to saved classifier
            validation_ranges_path: Path to saved validation ranges pickle file
        """
        # Load paths and data
        self.data_path = Path(data_path)
        self.classifier = joblib.load(classifier_path)

        with open(validation_ranges_path, "rb") as f:
            self.session_paths, self.validation_ranges = pickle.load(f)

        # Get sessions with validation data
        self.sessions = {
            k: v for k, v in self.session_paths.items() if k in self.validation_ranges
        }

        # Initialize widget state
        self.current_frame = 0

        # Create widgets
        self.session_select = pn.widgets.Select(
            name="Session", options=list(self.sessions.keys())
        )
        self.frame_slider = pn.widgets.IntSlider(
            name="Frame", start=0, end=1000, step=1, value=0
        )
        self.forward_button = pn.widgets.Button(
            name="Forward", button_type="primary", width=100
        )
        self.backward_button = pn.widgets.Button(
            name="Backward", button_type="primary", width=100
        )

        # Create image displays
        self.real_image = hv.DynamicMap(self.get_real_image).opts(
            cmap="gray", width=400, height=400, title="Original Frame"
        )
        self.corrected_image = hv.DynamicMap(self.get_corrected_image).opts(
            cmap="gray", width=400, height=400, title="Classifier Corrected"
        )

        # Layout
        self.navigation = pn.Row(
            self.backward_button, self.frame_slider, self.forward_button
        )
        self.images = pn.Row(self.real_image, self.corrected_image)

        self.widgets = pn.Column(
            pn.pane.Markdown("## Validation Frame Viewer"),
            self.session_select,
            self.navigation,
            self.images,
            width=850,
        )

        # Connect callbacks
        self.forward_button.on_click(self.advance_frame)
        self.backward_button.on_click(self.rewind_frame)
        self.frame_slider.param.watch(self.slider_moved, "value")
        self.session_select.param.watch(self.session_changed, "value")

        # Load initial session
        if self.sessions:
            self.session_select.value = list(self.sessions.keys())[0]
            self.load_session(self.session_select.value)

    def get_real_image(self, frame_num):
        """Display the original validation frame"""
        if hasattr(self, "real_frames") and 0 <= frame_num < len(self.real_frames):
            frame = self.real_frames[frame_num]
            return hv.Image(frame, bounds=(0, 0, 1, 1))
        return hv.Text(0.5, 0.5, "No Frame Available")

    def get_corrected_image(self, frame_num):
        """Display the classifier-corrected frame"""
        if hasattr(self, "corrected_frames") and 0 <= frame_num < len(
            self.corrected_frames
        ):
            frame = self.corrected_frames[frame_num]
            return hv.Image(frame, bounds=(0, 0, 1, 1))
        return hv.Text(0.5, 0.5, "No Frame Available")

    def load_session(self, session_name: str):
        """Load validation frames for a session and apply classifier"""
        self.widgets.loading = True

        with h5py.File(self.sessions[session_name], "r") as f:
            frames = f["frames"][()]

            # Extract validation frames
            val_frames = []
            for left, _slice in self.validation_ranges[session_name]:
                val_frames.append(frames[_slice])

            if val_frames:
                self.real_frames = np.concatenate(val_frames)

                # Get classifier predictions
                predictions = self.classifier.predict(self.real_frames)

                # Apply corrections based on predictions
                self.corrected_frames = self.real_frames.copy()
                self.corrected_frames[predictions == 1] = np.rot90(
                    self.corrected_frames[predictions == 1], 2, axes=(1, 2)
                )
            else:
                self.real_frames = np.array([])
                self.corrected_frames = np.array([])

        # Update slider
        if len(self.real_frames) > 0:
            self.frame_slider.end = len(self.real_frames) - 1
            self.frame_slider.value = 0
            self.current_frame = 0

        self.widgets.loading = False

    def advance_frame(self, event):
        """Move to next frame"""
        if self.current_frame < self.frame_slider.end:
            self.current_frame += 1
            self.frame_slider.value = self.current_frame

    def rewind_frame(self, event):
        """Move to previous frame"""
        if self.current_frame > self.frame_slider.start:
            self.current_frame -= 1
            self.frame_slider.value = self.current_frame

    def slider_moved(self, event):
        """Handle slider movement"""
        self.current_frame = event.new

    def session_changed(self, event):
        """Handle session selection"""
        self.load_session(event.new)

    def show(self):
        """Display the widget"""
        return self.widgets

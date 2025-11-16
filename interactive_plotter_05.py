import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sarpy.io.complex.sicd import SICDReader
from sarpy.visualization.remap import Density
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
from IPython.display import display, clear_output
import ipywidgets as widgets

TARGET_SPACING = 0.15  # meters

# Default configuration variables
NITF_PATH = ""
JSON_REPORT_PATH = ""
CLASS_MAP = {}

# Visualization options
SHOW_CONFIDENCE_SCORES = True
SHOW_CLASS_LABELS = True
SHOW_ERROR_TYPES = True
SHOW_UNMATCHED_DETECTIONS = True
SHOW_UNMATCHED_ANNOTATIONS = True
SHOW_CONFUSED_DETECTIONS = True

# Colors for different types of detections/annotations
COLORS = {
    'matched_detections': 'lime',
    'unmatched_detections': 'yellow',
    'unmatched_annotations': 'cyan',
    'confused_detections': 'magenta'
}

# Line styles for different types
LINE_STYLES = {
    'matched_detections': '-',
    'unmatched_detections': '--',
    'unmatched_annotations': ':',
    'confused_detections': '-.'
}

# Global state
KEYS_TO_VISUALIZE = ['matched_detections', 'unmatched_detections', 'unmatched_annotations', 'confused_detections']

# Global variables for observation tracking
current_detections = {}
current_image_stem = None
observations = []


def setup_jupyter_interactivity():
    """
    Helper function to set up plotting in Jupyter environments.

    Prefer the ipympl "widget" backend, which is compatible with JupyterLab.
    If that is not available, fall back to the static "inline" backend.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except Exception:
        ip = None

    if ip is None:
        print("Not running inside IPython/Jupyter; using default matplotlib backend.")
        return

    try:
        import ipympl  # ensure the widget backend is installed
        ip.run_line_magic("matplotlib", "widget")
        print("Using interactive 'widget' backend. You can zoom, pan, and interact with plots.")
        print("This backend works in JupyterLab and classic Notebook.")
    except Exception as e:
        print(f"Could not enable 'widget' backend ({e!r}); falling back to 'inline'.")
        ip.run_line_magic("matplotlib", "inline")
        print("Using 'inline' backend (static plots). Interactivity will be limited.")


def plot_from_matching_report(ax, report_data, keys_to_plot, width_scale_factor, height_scale_factor, image_stem_to_match, class_map):
    """
    Plots bounding boxes from a matching report JSON file onto a given matplotlib axis.

    Parameters:
        ax: matplotlib axis to plot on.
        report_data: Loaded JSON object from the matching report.
        keys_to_plot: List of keys in the JSON that contain detection/annotation data.
        width_scale_factor: Scaling factor for width (columns).
        height_scale_factor: Scaling factor for height (rows).
        image_stem_to_match: Stem used to filter which detections belong to the current image.
        class_map: Dictionary mapping class IDs to human-readable labels.
    """
    global current_detections
    current_detections = {}

    num_plotted = 0
    for key in keys_to_plot:
        if key not in report_data:
            print(f"Warning: key '{key}' not found in report data.")
            continue

        entries = report_data[key]
        for idx, det in enumerate(entries):
            # Check image stem
            img_path = det.get('image_path', '')
            if image_stem_to_match not in img_path:
                continue

            # Detection/annotation bounding box
            bbox = det.get('bbox', None)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            # Scale the bounding box to display coordinates
            x1_scaled = x1 * width_scale_factor
            x2_scaled = x2 * width_scale_factor
            y1_scaled = y1 * height_scale_factor
            y2_scaled = y2 * height_scale_factor

            width = x2_scaled - x1_scaled
            height = y2_scaled - y1_scaled

            color = COLORS.get(key, 'white')
            linestyle = LINE_STYLES.get(key, '-')

            rect = patches.Rectangle(
                (x1_scaled, y1_scaled),
                width, height,
                linewidth=1.5,
                edgecolor=color,
                facecolor='none',
                linestyle=linestyle,
                alpha=0.8
            )
            ax.add_patch(rect)

            # Get class label text
            cls_id = det.get('class_id', None)
            cls_label = class_map.get(cls_id, str(cls_id)) if cls_id is not None else "Unknown"

            # Build text label
            label_parts = []
            if SHOW_CLASS_LABELS:
                label_parts.append(cls_label)

            if SHOW_CONFIDENCE_SCORES and 'score' in det:
                label_parts.append(f"{det['score']:.2f}")

            if SHOW_ERROR_TYPES and 'error_type' in det:
                label_parts.append(det['error_type'])

            if label_parts:
                ax.text(
                    x1_scaled,
                    y1_scaled - 5,
                    " | ".join(label_parts),
                    fontsize=8,
                    color=color,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1)
                )

            # Save detection info for interactive queries
            det_id = f"{key}_{idx}"
            current_detections[det_id] = {
                'type': key,
                'original_bbox': bbox,
                'scaled_bbox': (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
                'class_label': cls_label,
                'score': det.get('score', None),
                'error_type': det.get('error_type', None),
                'image_path': img_path
            }
            num_plotted += 1

    print(f"Plotted {num_plotted} annotations/detections for image '{image_stem_to_match}'.")


def read_nitf_and_resample(nitf_path, target_spacing=TARGET_SPACING):
    """
    Reads a NITF file via SARPy, remaps the data to approximate the target pixel spacing,
    and returns the remapped image as a NumPy array.
    """
    print(f"Reading and preparing image from NITF: {nitf_path}")
    reader = SICDReader(nitf_path)
    sicd = reader.get_sicds_as_tuple()[0]

    # Original pixel spacing (row, col) – go through Grid
    row_ss = abs(sicd.Grid.Row.SS)
    col_ss = abs(sicd.Grid.Col.SS)
    print(f"Original pixel spacing (Row, Col): ({row_ss:.4f}m, {col_ss:.4f}m)")

    # Compute scale factor to hit desired ~target_spacing (averaging row/col sizes)
    avg_ss = 0.5 * (row_ss + col_ss)
    if avg_ss == 0:
        print("Warning: average sample spacing is zero or invalid; skipping resampling.")
        density = Density(reader[:, :])
        remapped = density()
    else:
        scale_factor = avg_ss / target_spacing
        print(f"Using scale factor ~ {scale_factor:.3f} to achieve ~{target_spacing} m spacing.")
        density = Density(reader[:, :], pixel_size=target_spacing)
        remapped = density()

    # Convert to magnitude image
    img = np.abs(remapped)
    return img, row_ss, col_ss


def create_interactive_visualization(
    nitf_path,
    json_report_path,
    keys_to_visualize,
    class_map=None,
    target_spacing=TARGET_SPACING,
    figsize=(12, 12)
):
    """
    High-level convenience function that:
      1) Reads the NITF file and rescales to target spacing
      2) Loads the JSON detections/matching report
      3) Plots the image + bounding boxes
      4) Enables interactive clicking on detections
    """
    global current_image_stem
    current_image_stem = Path(nitf_path).stem

    # Load report JSON
    try:
        with open(json_report_path, 'r') as f:
            report_json = json.load(f)
    except Exception as e:
        print(f"Error loading JSON report: {e}")
        return None

    # NITF read + approximate resampling
    full_res_img, row_ss, col_ss = read_nitf_and_resample(nitf_path, target_spacing)

    original_height, original_width = full_res_img.shape[:2]

    # We assume the Density remap approximates target_spacing; compute scale factors
    width_scale_factor = 1.0
    height_scale_factor = 1.0

    print("Computing scale factors for bounding boxes...")
    print(f"Original dimensions: ({original_height}, {original_width})")

    # We might want to fit the image into a typical screen size while maintaining aspect ratio
    max_display_width = 1600
    max_display_height = 1600

    height, width = full_res_img.shape[:2]
    width_scale = max_display_width / width
    height_scale = max_display_height / height
    overall_scale = min(width_scale, height_scale, 1.0)

    new_width = int(width * overall_scale)
    new_height = int(height * overall_scale)
    print(f"Resizing image from ({height}, {width}) to ({new_width}, {new_height})")
    img = cv2.resize(full_res_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    del full_res_img
    print(f"Display image shape: {img.shape}")

    # Create figure and axis with interactive features
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap='gray')

    # Plot the annotations/detections with enhanced styling
    plot_from_matching_report(ax, report_json, keys_to_visualize, width_scale_factor, height_scale_factor, current_image_stem, class_map)

    # Set up click handler for detailed inspection
    fig.canvas.mpl_connect('button_press_event', on_detection_click)

    # Set title and formatting
    ax.set_title(f"Detections and Annotations for {current_image_stem}", fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.show()
    return fig


def on_detection_click(event):
    """Handle mouse click events on the plot"""
    if event.inaxes is None:
        return

    click_x, click_y = event.xdata, event.ydata

    # Find which detection was clicked
    for det_id, det_info in current_detections.items():
        bbox = det_info['scaled_bbox']
        if bbox[0] <= click_x <= bbox[2] and bbox[1] <= click_y <= bbox[3]:
            show_detection_details(det_id, det_info)
            break


def show_detection_details(det_id, det_info):
    """Show detailed information about clicked detection"""
    clear_output(wait=True)

    print("=" * 60)
    print(f"DETECTION DETAILS - ID: {det_id}")
    print("=" * 60)
    print(f"Type: {det_info['type'].replace('_', ' ').title()}")
    print(f"Bounding Box: {det_info['original_bbox']}")

    if det_info['class_label'] is not None:
        print(f"Class: {det_info['class_label']}")

    if det_info['score'] is not None:
        print(f"Score: {det_info['score']:.4f}")

    if det_info['error_type'] is not None:
        print(f"Error Type: {det_info['error_type']}")

    print(f"Image Path: {det_info['image_path']}")
    print("=" * 60)

    # Create widgets for user input
    error_category = widgets.Dropdown(
        options=['Localization Error', 'Classification Error', 'Spurious Detection', 'Missed Detection', 'Other'],
        description='Error Category:',
        style={'description_width': 'initial'}
    )

    severity = widgets.Dropdown(
        options=['Low', 'Medium', 'High', 'Critical'],
        description='Severity:',
        style={'description_width': 'initial'}
    )

    notes = widgets.Textarea(
        value='',
        placeholder='Add any additional notes here...',
        description='Notes:',
        layout=widgets.Layout(width='100%', height='100px'),
        style={'description_width': 'initial'}
    )

    save_button = widgets.Button(description='Save Observation', button_style='success')
    cancel_button = widgets.Button(description='Cancel', button_style='warning')

    button_box = widgets.HBox([save_button, cancel_button])

    def on_save_clicked(b):
        add_observation(det_id, det_info, error_category.value, severity.value, notes.value)
        clear_output(wait=True)
        print("✅ Observation saved.")
        show_current_statistics()

    def on_cancel_clicked(b):
        clear_output(wait=True)
        print("❌ Observation canceled.")
        show_current_statistics()

    save_button.on_click(on_save_clicked)
    cancel_button.on_click(on_cancel_clicked)

    display(error_category, severity, notes, button_box)


def add_observation(det_id, det_info, error_category, severity, notes):
    """Add an observation to the global list"""
    global observations
    observation = {
        'timestamp': datetime.utcnow().isoformat(),
        'image_stem': current_image_stem,
        'detection_id': det_id,
        'type': det_info['type'],
        'bbox': det_info['original_bbox'],
        'class_label': det_info['class_label'],
        'score': det_info['score'],
        'error_type': det_info['error_type'],
        'error_category': error_category,
        'severity': severity,
        'notes': notes
    }
    observations.append(observation)


def save_observations_to_csv(output_path):
    """Save all observations to a CSV file"""
    if not observations:
        print("No observations to save.")
        return

    df = pd.DataFrame(observations)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(observations)} observations to {output_path}")


def load_existing_observations(csv_path=None):
    """Load existing observations from a CSV file"""
    global observations
    if not csv_path:
        print("No CSV path provided; starting with empty observations.")
        return

    if not os.path.exists(csv_path):
        print(f"No existing CSV found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    observations = df.to_dict('records')
    print(f"Loaded {len(observations)} existing observations from {csv_path}")


def show_current_statistics():
    """Display summary statistics of the current observations"""
    if not observations:
        print("No observations recorded yet.")
        return

    df = pd.DataFrame(observations)
    print("📊 CURRENT OBSERVATION STATISTICS")
    print(f"Total observations: {len(df)}\n")

    print("By error category:")
    error_counts = df['error_category'].value_counts().head(5)
    for error, count in error_counts.items():
        print(f"  {error}: {count}")

    print("\n🔥 Severity Distribution:")
    severity_counts = df['severity'].value_counts()
    for severity, count in severity_counts.items():
        print(f"  {severity}: {count}")


# Main workflow functions
def start_model_evaluation():
    """Start the model evaluation workflow"""
    print("🔬 MODEL EVALUATION WORKFLOW")
    print("1. Run: setup_jupyter_interactivity()")
    print("2. Update file paths: NITF_PATH, JSON_REPORT_PATH, CLASS_MAP")
    print('3. (Optional) Run: load_existing_observations("/path/to/observations.csv")')
    print("4. Run: fig = create_review_plot()")
    print("5. Click on detections to inspect and log observations")
    print("6. Use show_current_statistics() to see progress")


def create_review_plot():
    """Create the main review plot"""
    if not NITF_PATH or not JSON_REPORT_PATH:
        print("❌ Please update NITF_PATH and JSON_REPORT_PATH variables first!")
        print("Example:")
        print('NITF_PATH = "/path/to/your/file.nitf"')
        print('JSON_REPORT_PATH = "/path/to/your/detection_matching.json"')
        return None

    if not os.path.exists(NITF_PATH):
        print(f"❌ NITF file does not exist: {NITF_PATH}")
        return None

    if not os.path.exists(JSON_REPORT_PATH):
        print(f"❌ JSON report file does not exist: {JSON_REPORT_PATH}")
        return None

    print("🎯 Creating review plot...")
    print(f"   Image:  {Path(NITF_PATH).name}")
    print(f"   Report: {Path(JSON_REPORT_PATH).name}")

    fig = create_interactive_visualization(
        nitf_path=NITF_PATH,
        json_report_path=JSON_REPORT_PATH,
        keys_to_visualize=KEYS_TO_VISUALIZE,
        class_map=CLASS_MAP,
        figsize=(16, 12)
    )

    return fig


# Initialize on import
print("🎯 DETECTION MODEL EVALUATION TOOLKIT")
print("Quick start: run start_model_evaluation() for instructions")

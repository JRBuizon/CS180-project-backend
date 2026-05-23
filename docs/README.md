# CS180 Project Backend — Personal Posture Journal

## Overview

A desktop application that uses computer vision + machine learning to detect and classify a user's posture in real time via webcam. Built with Python, MediaPipe, OpenCV, Tkinter, and scikit-learn.

---

## File Structure

```
CS180-project-backend/
├── app.py                  # Main GUI application (459 lines)
├── test.py                 # CLI data-collection test script (152 lines)
├── requirements.txt        # Python dependencies
├── pose_landmarker.task       # MediaPipe pose estimation model (auto-downloaded)
├── posture_data_csv/          # Directory for labeled training data CSVs
│   ├── posture_data.csv
│   ├── posture_data(1).csv
│   └── posture_data(2).csv
├── posture_models/            # Saved model weight files (.pkl + .pkl.bkp backups)
├── plans/                     # Implementation plans for agents
│   ├── model_persistence_plan.md
│   └── immediate_training_plan.md
├── opencode.json              # OpenCode AI config (MCP servers, permissions)
├── .gitignore                 # Ignores cs180/ directory
└── docs/
    └── README.md              # This file — codebase documentation for agents
```

---

## Core Architecture (`app.py`)

### Class: `PostureApp`
A Tkinter-based GUI application with a left camera feed panel and a right control panel.

### Pose Detection Pipeline

1. **MediaPipe Pose Landmarker** runs in `VIDEO` mode, detecting 33 body landmarks per frame.
2. **Relevant landmarks** tracked: `0` (nose), `11` (left shoulder), `12` (right shoulder), `23` (left hip), `24` (right hip).
3. **Stick figure rendering**: Lines drawn between `(11,12)`, `(0,11)`, `(0,12)`; dots drawn at joints.

### Feature Extraction (`extract_features` at `app.py:133`)

From landmarks, 3 numerical features are computed:

| Feature | Formula | Meaning |
|---|---|---|
| `head_forward` | `nose.y - shoulder_midpoint.y` | How far the head protrudes forward |
| `spine_angle` | Angle at hip midpoint, using vector `[0,1]` as reference | Trunk lean from vertical |
| `shoulder_tilt` | `abs(left_shoulder.y - right_shoulder.y)` | Asymmetry of shoulders |

### ML Training Methods

**`train_model_from_local_data`** (startup fallback if no `.pkl` exists)
- Reads all `posture_data*.csv` files via glob
- Concatenates all CSV data into a single DataFrame
- Trains a **`RandomForestClassifier`** (100 trees, `random_state=42`)
- Features: `head_forward`, `spine_angle`, `shoulder_tilt`
- Target: `label` (0 = good posture, 1 = slouch)
- Saves trained model to `posture_models/posture_model_<timestamp>.pkl`

**`_train_in_memory`** (called after each snapshot capture)
- Same training logic but **no disk write** — updates `self.model` in RAM only
- Called automatically by `save_snapshot` so the model is always up to date without spamming `.pkl` files

**`export_model_weights`** (called via "📦 EXPORT MODEL WEIGHTS" button)
- Writes current in-memory model to `posture_models/posture_model_<timestamp>.pkl`
- Backs up existing `.pkl` files as `.pkl.bkp`
- Does **not** retrain — only persists whatever model is currently in RAM

### Data Collection (`save_snapshot`)

- Pressing **G** or clicking "Capture GOOD" saves current features with `label=0`
- Pressing **S** or clicking "Capture SLOUCH" saves current features with `label=1`
- Appends a row to `posture_data.csv`
- **Immediately retrains the in-memory model** via `_train_in_memory()`

### Real-Time Inference Loop (`video_loop` at `app.py:205`)

- Runs in a separate **daemon thread**
- Targets 30 FPS with dynamic throttling (sleeps remaining frame time)
- On each frame: detect pose → extract features → predict with RandomForest → update UI
- If model predicts `0`: shows "GOOD POSTURE" (green)
- If model predicts `1`: shows "SLOUCHING!" (red)
- Gracefully handles missing frames by duplicating last valid frame

### UI Layout

```
┌─────────────────────────────────┬──────────────────┐
│                                 │  POSTURE STATUS   │
│                                 │  [GOOD/SLOUCH]   │
│      Camera Feed (640x480)      │──────────────────│
│       with stick figure         │  Live Metrics    │
│                                 │  Head Forward: X │
│                                 │  Spine Angle: Y° │
│                                 │──────────────────│
│                                 │ Data Collection  │
│                                 │ [Capture GOOD]   │
│                                 │ [Capture SLOUCH] │
│                                 │──────────────────│
│                                 │ [📦 EXPORT       │
│                                 │  MODEL WEIGHTS]  │
└─────────────────────────────────┴──────────────────┘
```

- Theme: Catppuccin Mocha-inspired dark color palette (`#1e1e2e`, `#cdd6f4`, `#89b4fa`, etc.)

---

## Data Collection Script (`test.py`)

A simpler OpenCV-only version for collecting labeled posture data:

- Displays webcam feed with stick figure overlay
- **G** key → saves current frame features as `label=0` (good)
- **S** key → saves current frame features as `label=1` (slouch)
- **Q** key → quits and saves all collected records to `posture_data.csv`
- HUD shows counts of good/slouch samples collected

---

## CSV Data Format

```csv
head_forward,spine_angle,shoulder_tilt,label
0.025,92.5,0.012,0
-0.018,78.3,0.045,1
```

---

## Dependencies (`requirements.txt`)

| Library | Purpose |
|---|---|
| `mediapipe==0.10.35` | Pose landmark detection |
| `opencv-python==4.13.0` | Camera capture + image processing |
| `scikit-learn==1.8.0` | RandomForest classifier |
| `pandas==3.0.3` | CSV data loading + concatenation |
| `pillow==12.2.0` | Image-to-Tkinter conversion |
| `numpy==2.4.6` | Numerical math |
| `sounddevice==0.5.5` | Installed but unused in current code |

---

## How It Works End-to-End

1. **Startup**: Downloads pose model if missing, loads existing CSV data, trains initial RandomForest model, opens webcam, starts video loop thread.
2. **Continuous loop**: For each camera frame → MediaPipe detects 33 pose landmarks → extracts 3 geometric features → renders stick figure on camera feed → features are fed into trained RandomForest for real-time prediction → UI updates with posture status.
3. **Data collection loop**: User presses G (GOOD) or S (SLOUCH) to capture labeled snapshots → data appended to CSV → in-memory model immediately retrained via `_train_in_memory()`. Click "📦 EXPORT MODEL WEIGHTS" to persist the current model to disk as a `.pkl` file.

---

## Permission & DevConfig (`opencode.json`)

- Configures MCP servers: DeepWiki, Context7, Svelte
- Sets default agent to "plan"
- Granular permission rules for bash, edit, and skill operations
- Used by the OpenCode AI tooling

---

## Key Design Decisions

- **Threaded video loop**: Video processing runs on a daemon thread to keep the Tkinter UI responsive
- **Dynamic throttling**: Adaptive sleep to maintain ~30 FPS regardless of processing time variance
- **Graceful degradation**: Last valid frame duplicated when camera fails to produce a new frame
- **Multi-CSV support**: Globs `posture_data_csv/posture_data*.csv` so the user can keep multiple labeled datasets in a dedicated directory
- **In-memory retrain on capture**: Each G/S snapshot automatically retrains the RandomForest in-memory via `_train_in_memory()` — no disk write, so no `.pkl` spam. The model is always current for real-time inference.
- **Explicit export only**: The "📦 EXPORT MODEL WEIGHTS" button is the sole way to write a `.pkl` to disk. On startup, the latest `.pkl` is loaded; if none exists, `train_model_from_local_data()` trains and exports one. Old `.pkl` files are renamed to `.pkl.bkp` on each export, preserving all backups.
- **MediaPipe VIDEO mode**: Uses timestamp-based tracking for smoother temporal detection

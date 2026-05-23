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
├── posture_logs/              # Directory for session history JSON logs
│   └── session_history.json
├── posture_models/            # Saved model weight files (.pkl + .pkl.bkp backups)
├── .gitignore                 # Ignores cs180/ directory
└── docs/
    └── README.md              # This file — codebase documentation for agents
```

---

## Core Architecture (`app.py`)

### Class: `PostureApp`

A Tkinter-based GUI application with two-page navigation (home page + monitoring page). The monitoring page has a left camera feed panel and a right control panel.

### Pose Detection Pipeline

1. **MediaPipe Pose Landmarker** runs in `VIDEO` mode, detecting 33 body landmarks per frame.
2. **Relevant landmarks** tracked: `0` (nose), `11` (left shoulder), `12` (right shoulder), `23` (left hip), `24` (right hip).
3. **Stick figure rendering**: Lines drawn between `(11,12)`, `(0,11)`, `(0,12)`; dots drawn at joints.

### Feature Extraction (`extract_features` at `app.py:222`)

From landmarks, 3 numerical features are computed:

| Feature         | Formula                                                  | Meaning                            |
| --------------- | -------------------------------------------------------- | ---------------------------------- |
| `head_forward`  | `nose.y - shoulder_midpoint.y`                           | How far the head protrudes forward |
| `spine_angle`   | Angle at hip midpoint, using vector `[0,1]` as reference | Trunk lean from vertical           |
| `shoulder_tilt` | `abs(left_shoulder.y - right_shoulder.y)`                | Asymmetry of shoulders             |

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

### Real-Time Inference Loop (`video_loop` at `app.py:359`)

- Runs in a separate **daemon thread**
- Targets ~60 FPS with fixed `time.sleep(0.016)` cap
- On each frame: detect pose → extract features → predict with RandomForest → update UI
- If model predicts `0`: shows "GOOD POSTURE" (green)
- If model predicts `1`: shows "SLOUCH ALERT" (red)
- **Rolling buffer**: A 15-frame prediction history uses majority voting to smooth out flickering classifications
- **Thread-safe UI updates**: `update_cam_label` helper uses `window.after(0, ...)` to safely push frames to the Tkinter main thread

### Session Tracking & Alerts

The application tracks session-specific metrics and provides real-time alerts for prolonged slouching.

- **Session Metrics**: `total_alerts`, `good_posture_duration`, `bad_posture_duration` are recorded.
- **Slouch Detection**: If slouching is detected continuously for `max_slouch_seconds` (default 10s), an alert is triggered.
- **Alert Mechanism**: A `messagebox.showwarning` popup is displayed to remind the user to correct their posture.
- **Session Summary**: When the monitoring session ends (by returning to the home page or closing the app), a summary of the session (`session_date`, `total_duration`, `good_posture_duration`, `bad_posture_duration`, `total_slouch_alerts`, `trained_this_session`) is saved to `posture_logs/session_history.json`.

### UI Layout

#### Home Page

```
┌──────────────────────────────────────────────┐
│                                              │
│         Personal Posture Journal             │
│       Real-time posture monitoring app       │
│                                              │
│              ┌──────────────┐                │
│              │ START SESSION│                │
│              └──────────────┘                │
│                                              │
│         [ Settings ]   [ Logs ]              │
│                                              │
└──────────────────────────────────────────────┘
```

#### Monitoring Page

```
┌─────────────────────────────────┬──────────────────┐
│                                 │ ← Back to Home   │
│                                 ├──────────────────┤
│      Camera Feed (640x480)      │  POSTURE STATUS  │
│       with stick figure         │  [GOOD/SLOUCH]   │
│                                 │──────────────────│
│                                 │  Live Metrics    │
│                                 │  Head Displace: X│
│                                 │  Vertebral Ang: Y│
│                                 │──────────────────│
│                                 │ Data Collection  │
│                                 │ [Capture GOOD]   │
│                                 │ [Capture SLOUCH] │
│                                 ├──────────────────┤
│                                 │ [📦 EXPORT      │
│                                 │  MODEL WEIGHTS]  │
└─────────────────────────────────┴──────────────────┘
```

- Theme: Catppuccin Mocha-inspired dark color palette (`#1e1e2e`, `#cdd6f4`, `#89b4fa`, etc.)
- Skeleton lines: tan `(226, 214, 180)`; joint dots: blue `(135, 180, 249)`

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

| Library                 | Purpose                              |
| ----------------------- | ------------------------------------ |
| `mediapipe==0.10.35`    | Pose landmark detection              |
| `opencv-python==4.13.0` | Camera capture + image processing    |
| `scikit-learn==1.8.0`   | RandomForest classifier              |
| `pandas==3.0.3`         | CSV data loading + concatenation     |
| `pillow==12.2.0`        | Image-to-Tkinter conversion          |
| `numpy==2.4.6`          | Numerical math                       |
| `sounddevice==0.5.5`    | Installed but unused in current code |

---

## How It Works End-to-End

1. **Startup**: Downloads pose model if missing, loads existing CSV data and/or `.pkl` model weights, opens **home page** — camera stays off.
2. **Home page**: User sees title, "START SESSION" button, and placeholder Settings/Logs buttons. Clicking "START SESSION" switches to monitoring.
3. **Monitoring page**: Camera initializes, video loop starts. For each frame → MediaPipe detects 33 pose landmarks → extracts 3 geometric features → renders stick figure on camera feed → features fed into trained RandomForest → rolling 15-frame majority vote stabilizes prediction → UI updates with posture status. If slouching persists for `max_slouch_seconds`, a warning popup is triggered.
4. **Data collection loop**: User presses G (GOOD) or S (SLOUCH) to capture labeled snapshots → data appended to CSV → in-memory model immediately retrained via `_train_in_memory()`. Click "📦 EXPORT MODEL WEIGHTS" to persist the current model to disk as a `.pkl` file.
5. **Session End**: Upon returning to the home page or closing the application, a summary of the session's posture metrics (good/bad posture duration, total alerts, trained in session status) is saved to `posture_logs/session_history.json`.

---

## Permission & DevConfig (`opencode.json`)

- Configures MCP servers: DeepWiki, Context7, Svelte
- Sets default agent to "plan"
- Granular permission rules for bash, edit, and skill operations
- Used by the OpenCode AI tooling

---

## Key Design Decisions

- **Threaded video loop**: Video processing runs on a daemon thread to keep the Tkinter UI responsive
- **Two-page navigation**: Home page vs monitoring page. Camera only initializes when the user clicks "START SESSION" and is released when they return home. Hotkeys G/S are bound only during monitoring.
- **Posture stabilization**: A 15-frame rolling buffer applies majority voting to prediction results, preventing classification flicker from frame-to-frame variance.
- **Thread-safe UI updates**: `update_cam_label` uses `window.after(0, ...)` to atomically push frames onto the Tkinter main thread, preventing race conditions and partial draw drops.
- **Fixed 60 FPS cap**: A consistent `time.sleep(0.016)` keeps frame rate steady without dynamic throttling complexity.
- **Multi-CSV support**: Globs `posture_data_csv/posture_data*.csv` so the user can keep multiple labeled datasets in a dedicated directory
- **In-memory retrain on capture**: Each G/S snapshot automatically retrains the RandomForest in-memory via `_train_in_memory()` — no disk write, so no `.pkl` spam. The model is always current for real-time inference.
- **Explicit export only**: The "📦 EXPORT MODEL WEIGHTS" button is the sole way to write a `.pkl` to disk. On startup, the latest `.pkl` is loaded; if none exists, `train_model_from_local_data()` trains and exports one. Old `.pkl` files are renamed to `.pkl.bkp` on each export, preserving all backups.
- **MediaPipe VIDEO mode**: Uses timestamp-based tracking for smoother temporal detection
- **Color palette**: Catppuccin Mocha-inspired dark theme. Skeleton lines rendered in tan `(226, 214, 180)`, joint dots in blue `(135, 180, 249)`.
- **Session Tracking & JSON Logging**: Comprehensive session metrics (good/bad posture duration, total alerts, etc.) are tracked and saved to a JSON log file on session conclusion.
- **Time-based Slouch Alerts**: A configurable `max_slouch_seconds` threshold triggers a desktop popup to alert users of prolonged poor posture, with a mechanism to prevent repetitive alerts.
- **`trained_this_session` flag**: A flag to indicate if the model was trained during the current session, useful for session summary and analytics.

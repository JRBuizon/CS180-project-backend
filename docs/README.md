# CS180 Project Backend — Personal Posture Journal

## Overview

A desktop application that uses computer vision + machine learning to detect and classify a user's posture in real time via webcam. Built with Python, MediaPipe, OpenCV, Tkinter, and scikit-learn.

---

## File Structure

```
CS180-project-backend/
├── app.py                  # Main GUI application (927 lines)
├── test.py                 # CLI data-collection test script (152 lines)
├── requirements.txt        # Python dependencies
├── settings.json           # Configurable slouch threshold (default 10s)
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

A Tkinter-based GUI application with multi-page navigation (home page, monitoring page, settings page, session logs page). The monitoring and settings pages share a left camera feed panel and a right control panel layout.

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

**`export_model_weights`** (called via "📦 Update Model" button)

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

### Settings Page (`show_settings` at `app.py:399`)

The Settings page mirrors the monitoring page's layout (camera feed, status, metrics) but replaces the session timer with a **Slouch Threshold** configuration.

- **Configurable Alert Timer**: A `ttk.Spinbox` (range 1–300s) lets the user adjust `max_slouch_seconds` before alerts trigger.
- **`load_settings()`** (`app.py:588`): Reads `settings.json` on startup; creates it with default 10s if missing.
- **`save_settings(new_seconds)`** (`app.py:603`): Writes the value back to `settings.json`.
- **`save_settings_from_spinbox()`** (`app.py:362`): Validates spinbox input and calls `save_settings`, then returns to home.
- **Data Collection**: Identical capture buttons (G/S) as the monitoring page — snapshots train the in-memory model the same way.
- **Active UI References**: The `active_cam_label`/`active_head_lbl`/`active_spine_lbl`/`active_status_lbl` pattern lets both monitoring and settings pages share the same `video_loop` code, switching which labels get updated based on `self.session_context` ("monitoring" vs "settings").

### Session Logs Page (`show_session_logs` at `app.py:415`)

A read-only page that displays all past session summaries from `posture_logs/session_history.json`.

- **`load_session_history()`** (`app.py:425`): Reads the JSON file, returns a list of session dicts (or empty list on failure).
- **`render_session_logs()`** (`app.py:440`): Destroys old widgets, iterates sessions in reverse-chronological order, creating a card per session showing:
  - `session_date`, `total_duration`, `good_posture_duration` (green), `bad_posture_duration` (red), `total_slouch_alerts`, `trained_this_session`.
- **Scrollable canvas**: A `tk.Canvas` with vertical scrollbar supports mousewheel scrolling (bound on enter/leave).
- **Empty state**: Shows "No saved sessions yet." label when no history exists.

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
│              [   Exit   ]                    │
└──────────────────────────────────────────────┘
```

#### Monitoring Page

```
┌─────────────────────────────────┬──────────────────┐
│                                 │ ⏹ End Session   │
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
│                                 │ [📦 Update      │
│                                 │  Model]          │
│                                 ├──────────────────┤
│                                 │ SESSION DURATION │
│                                 │    00:00:00      │
└─────────────────────────────────┴──────────────────┘
```

- Theme: Catppuccin Mocha-inspired dark color palette (`#1e1e2e`, `#cdd6f4`, `#89b4fa`, etc.)
- Skeleton lines: tan `(226, 214, 180)`; joint dots: blue `(135, 180, 249)`

#### Settings Page

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
│                                 ├──────────────────┤
│                                 │ Slouch Threshold │
│                                 │ Delay: [Spinbox] │
│                                 │    (1-300 secs)  │
│                                 ├──────────────────┤
│                                 │ Data Collection  │
│                                 │ [Capture GOOD]   │
│                                 │ [Capture SLOUCH] │
│                                 ├──────────────────┤
│                                 │ [📦 Update      │
│                                 │  Model]          │
│                                 │ [💾 Save        │
│                                 │  Settings]       │
└─────────────────────────────────┴──────────────────┘
```

#### Session Logs Page

```
┌───────────────────────────────────────────────┐
│ ← Back to Home                                │
│                                               │
│               Session Logs                    │
│                                               │
│ ┌───────────────────────────────────────────┐ │
│ │ 2026-05-23 23:03:44           (card)      │ │
│ │ Total: 46.47s    Good: 23.12s             │ │
│ │ Bad: 23.35s     Alerts: 1                │ │
│ │ Trained: False   Duration Log: 46.47s     │ │
│ └───────────────────────────────────────────┘ │
│                                               │
│ (scrollable list of session history cards)     │
└───────────────────────────────────────────────┘
```

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
2. **Home page**: User sees title, "START SESSION" button, and Settings/Logs/Exit buttons. Clicking "START SESSION" switches to monitoring. Clicking "Settings" opens the settings page with a live camera feed and configurable slouch threshold.
3. **Monitoring page**: Camera initializes, video loop starts. For each frame → MediaPipe detects 33 pose landmarks → extracts 3 geometric features → renders stick figure on camera feed → features fed into trained RandomForest → rolling 15-frame majority vote stabilizes prediction → UI updates with posture status. If slouching persists for `max_slouch_seconds`, a warning popup is triggered. Session duration is shown live.
4. **Data collection loop**: User presses G (GOOD) or S (SLOUCH) to capture labeled snapshots → data appended to CSV → in-memory model immediately retrained via `_train_in_memory()`. Click "📦 Update Model" to persist the current model to disk as a `.pkl` file.
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
- **Multi-page navigation**: Home page, monitoring page, settings page, and session logs page. Camera only initializes when entering monitoring or settings (both need the feed) and is released when returning home. Hotkeys G/S are bound only during monitoring/settings.
- **Posture stabilization**: A 15-frame rolling buffer applies majority voting to prediction results, preventing classification flicker from frame-to-frame variance.
- **Thread-safe UI updates**: `update_cam_label` uses `window.after(0, ...)` to atomically push frames onto the Tkinter main thread, preventing race conditions and partial draw drops.
- **Fixed 60 FPS cap**: A consistent `time.sleep(0.016)` keeps frame rate steady without dynamic throttling complexity.
- **Multi-CSV support**: Globs `posture_data_csv/posture_data*.csv` so the user can keep multiple labeled datasets in a dedicated directory
- **In-memory retrain on capture**: Each G/S snapshot automatically retrains the RandomForest in-memory via `_train_in_memory()` — no disk write, so no `.pkl` spam. The model is always current for real-time inference.
- **Explicit export only**: The "📦 Update Model" button is the sole way to write a `.pkl` to disk. On startup, the latest `.pkl` is loaded; if none exists, `train_model_from_local_data()` trains and exports one. Old `.pkl` files are renamed to `.pkl.bkp` on each export, preserving all backups.
- **MediaPipe VIDEO mode**: Uses timestamp-based tracking for smoother temporal detection
- **Color palette**: Catppuccin Mocha-inspired dark theme. Skeleton lines rendered in tan `(226, 214, 180)`, joint dots in blue `(135, 180, 249)`.
- **Settings page with configurable slouch threshold**: The settings page provides a spinbox (1–300s) to adjust `max_slouch_seconds` before alerts fire. Saved persistently to `settings.json` via `load_settings()`/`save_settings()`.
- **Session Logs page**: A dedicated scrollable page displays session history cards, showing per-session metrics (duration, good/bad breakdown, alerts, training status).
- **Active UI reference pattern**: `active_cam_label`, `active_head_lbl`, `active_spine_lbl`, `active_status_lbl` are assigned when entering monitoring or settings mode, allowing the shared `video_loop` to write to the correct widgets based on `session_context`.
- **Session context tracking**: `self.session_context` distinguishes "monitoring" (triggers session start/reset + alert popups) from "settings" (no session metrics or alerts).
- **Session Tracking & JSON Logging**: Comprehensive session metrics (good/bad posture duration, total alerts, etc.) are tracked and saved to a JSON log file on session conclusion.
- **Time-based Slouch Alerts**: A configurable `max_slouch_seconds` threshold triggers a desktop popup to alert users of prolonged poor posture, with a mechanism to prevent repetitive alerts.
- **`trained_this_session` flag**: A flag to indicate if the model was trained during the current session, useful for session summary and analytics.

import cv2
import csv
import glob
import os
import time
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import threading
import numpy as np
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

from datetime import datetime
import joblib

from sklearn.ensemble import RandomForestClassifier

class PostureApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Personal Posture Journal")
        self.window.geometry("1000x650")
        self.window.configure(bg="#1e1e2e")

        # --- Data & Model State ---
        self.model = None
        self.is_trained = False
        self.current_features = None
        self.csv_filename = "posture_data_csv/posture_data.csv"
        
        # --- Session Summary Metrics ---
        self.session_start_wall_time = None
        self.total_alerts = 0
        self.good_posture_duration = 0.0
        self.bad_posture_duration = 0.0
        self.last_state_timestamp = None
        self.current_session_state = None  # Tracks 0 for good, 1 for bad
        
        # --- Rolling Buffer for Posture Stabilization ---
        self.prediction_history = []
        self.buffer_size = 15  # Tracks last 15 frames for majority voting
        
        # --- Time-based Alert Tracking ---
        self.slouch_start_time = None
        self.alertpopup_active = False       # Flag to prevent multiple popups from staking up
        self.max_slouch_seconds = 10    # Time threshold before triggering an alert
        
        # Download MediaPipe task file if missing
        self.model_path = "pose_landmarker.task"
        if not os.path.exists(self.model_path):
            print("Downloading pose landmarker model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
                self.model_path
            )

        # Load existing model weights or train from CSVs
        pkl_files = sorted(
            glob.glob("posture_models/posture_model_*.pkl"),
            key=os.path.getctime
        )
        if pkl_files:
            self.model = joblib.load(pkl_files[-1])
            self.is_trained = True
            print(f"Loaded model from {pkl_files[-1]}")
        else:
            self.train_model_from_local_data()

        # --- MediaPipe Setup ---
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(self.options)

        # --- UI Layout ---
        self.setup_styles()
        self.create_widgets()
        # Start on home page
        self.show_home()

        # Handle window close cleanly
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Arial", 11))
        style.configure("TButton", font=("Arial", 10, "bold"), background="#89b4fa", foreground="#11111b")
        style.map("TButton", background=[("active", "#b4befe")])

    def create_widgets(self):
        # Page container — holds both home and monitoring pages
        self.page_container = tk.Frame(self.window, bg="#1e1e2e")
        self.page_container.pack(fill=tk.BOTH, expand=True)

        # ===================== Home Page =====================
        self.home_frame = tk.Frame(self.page_container, bg="#1e1e2e")

        # Title Section
        tk.Label(self.home_frame, text="Personal Posture Journal",
                 font=("Arial", 28, "bold"), bg="#1e1e2e", fg="#cdd6f4"
                 ).pack(pady=(120, 10))
        tk.Label(self.home_frame, text="Real-time posture monitoring app",
                 font=("Arial", 14), bg="#1e1e2e", fg="#a6adc8"
                 ).pack(pady=(0, 60))

        # Start Session Button
        tk.Button(self.home_frame, text="START SESSION",
                  font=("Arial", 16, "bold"), bg="#89b4fa", fg="#11111b",
                  width=20, height=2, command=self.show_monitoring
                  ).pack(pady=20)

        # Placeholder Buttons Row
        btn_frame = tk.Frame(self.home_frame, bg="#1e1e2e")
        btn_frame.pack(pady=40)

        tk.Button(btn_frame, text="Settings", font=("Arial", 11),
                  bg="#313244", fg="#cdd6f4", width=12, command=lambda: None
                  ).pack(side=tk.LEFT, padx=10)

        tk.Button(btn_frame, text="Logs", font=("Arial", 11),
                  bg="#313244", fg="#cdd6f4", width=12, command=self.show_session_logs
                  ).pack(side=tk.LEFT, padx=10)

        # ===================== Session Logs Page =====================
        self.session_logs_frame = tk.Frame(self.page_container, bg="#1e1e2e")
        session_top = tk.Frame(self.session_logs_frame, bg="#1e1e2e")
        session_top.pack(fill=tk.X, padx=20, pady=20)
        tk.Button(session_top, text="← Back to Home",
                  font=("Arial", 10, "bold"), bg="#45475a", fg="#cdd6f4",
                  command=self.show_home
                  ).pack(side=tk.LEFT)
        tk.Label(self.session_logs_frame, text="Session Logs",
                 font=("Arial", 24, "bold"), bg="#1e1e2e", fg="#cdd6f4"
                 ).pack(pady=(10, 20))

        self.session_logs_list_container = tk.Frame(self.session_logs_frame, bg="#1e1e2e")
        self.session_logs_list_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        self.session_logs_canvas = tk.Canvas(self.session_logs_list_container,
                                            bg="#1e1e2e", highlightthickness=0, bd=0)
        self.session_logs_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        session_scrollbar = tk.Scrollbar(self.session_logs_list_container,
                                         orient=tk.VERTICAL, command=self.session_logs_canvas.yview)
        session_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.session_logs_canvas.configure(yscrollcommand=session_scrollbar.set)

        self.session_logs_content = tk.Frame(self.session_logs_canvas, bg="#1e1e2e")
        self.session_logs_window = self.session_logs_canvas.create_window((0, 0), window=self.session_logs_content, anchor="nw")

        self.session_logs_content.bind(
            "<Configure>",
            lambda event: self.session_logs_canvas.configure(scrollregion=self.session_logs_canvas.bbox("all"))
        )
        self.session_logs_canvas.bind(
            "<Configure>",
            lambda event: self.session_logs_canvas.itemconfig(self.session_logs_window, width=event.width)
        )
        self.session_logs_canvas.bind("<Enter>", self._bind_session_logs_mousewheel)
        self.session_logs_canvas.bind("<Leave>", self._unbind_session_logs_mousewheel)

        # Show a placeholder on first startup before logs are loaded
        self.session_logs_empty_label = tk.Label(self.session_logs_content,
                                                  text="No saved sessions yet.",
                                                  font=("Arial", 12), bg="#1e1e2e", fg="#a6adc8")
        self.session_logs_empty_label.pack(pady=10)

        # ===================== Monitoring Page =====================
        self.monitoring_frame = tk.Frame(self.page_container, bg="#1e1e2e")

        # Left Panel - Camera Feed
        self.left_frame = tk.Frame(self.monitoring_frame, bg="#1e1e2e")
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        self.cam_label = tk.Label(self.left_frame, bg="#313244", width=640, height=480)
        self.cam_label.pack(fill=tk.BOTH, expand=True)

        # Right Panel - Control Center
        self.right_frame = tk.Frame(self.monitoring_frame, bg="#181825", width=300)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 20), pady=20)
        self.right_frame.pack_propagate(False)

        # End Session Button
        tk.Button(self.right_frame, text="← End Session",
                  font=("Arial", 10, "bold"), bg="#45475a", fg="#cdd6f4",
                  command=self.show_home
                  ).pack(fill=tk.X, padx=15, pady=(15, 5))

        # Status Display
        tk.Label(self.right_frame, text="POSTURE STATUS",
                 font=("Arial", 14, "bold"), bg="#181825", fg="#a6adc8"
                 ).pack(pady=(20, 5))
        self.status_lbl = tk.Label(self.right_frame, text="NO MODEL TRAINED",
                                   font=("Arial", 18, "bold"),
                                   bg="#313244", fg="#f38ba8", width=18, pady=10)
        self.status_lbl.pack(pady=10)

        # Live Metrics
        self.metrics_frame = tk.LabelFrame(self.right_frame, text=" Live Metrics ",
                                           bg="#181825", fg="#cdd6f4",
                                           font=("Arial", 10, "bold"), padx=10, pady=10)
        self.metrics_frame.pack(fill=tk.X, padx=15, pady=15)

        self.head_lbl = ttk.Label(self.metrics_frame, text="Head Forward: --")
        self.head_lbl.pack(anchor="w", pady=2)
        self.spine_lbl = ttk.Label(self.metrics_frame, text="Spine Angle: --")
        self.spine_lbl.pack(anchor="w", pady=2)

        # Data Collection Section
        self.train_frame = tk.LabelFrame(self.right_frame, text=" Data Collection ",
                                         bg="#181825", fg="#cdd6f4",
                                         font=("Arial", 10, "bold"), padx=10, pady=10)
        self.train_frame.pack(fill=tk.X, padx=15, pady=15)

        ttk.Label(self.train_frame, text="Collect new snapshots:").pack(pady=5)

        tk.Button(self.train_frame, text="Capture GOOD (G)",
                  bg="#a6e3a1", fg="#11111b", font=("Arial", 10, "bold"),
                  command=lambda: self.save_snapshot(0)
                  ).pack(fill=tk.X, pady=4)

        tk.Button(self.train_frame, text="Capture SLOUCH (S)",
                  bg="#f38ba8", fg="#11111b", font=("Arial", 10, "bold"),
                  command=lambda: self.save_snapshot(1)
                  ).pack(fill=tk.X, pady=4)

        # Export Action Button
        tk.Button(self.right_frame, text="📦 EXPORT MODEL WEIGHTS",
                  bg="#fab387", fg="#11111b", font=("Arial", 12, "bold"),
                  command=self.export_model_weights
                  ).pack(fill=tk.X, padx=15, pady=(15, 8))

        # Session Duration Display
        duration_frame = tk.Frame(self.right_frame, bg="#181825")
        duration_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        tk.Label(duration_frame, text="SESSION DURATION",
                 font=("Arial", 10, "bold"), bg="#181825", fg="#cdd6f4"
                 ).pack(anchor="w")
        self.session_duration_lbl = tk.Label(duration_frame, text="00:00:00",
                                             font=("Arial", 14, "bold"),
                                             bg="#313244", fg="#cdd6f4",
                                             width=18, pady=10)
        self.session_duration_lbl.pack(fill=tk.X)

    def show_home(self):
        # Save session tracking statistics before winding down the interface
        if hasattr(self, 'running') and self.running:
            self.save_session_summary_json()
        
        self.stop_camera()
        self.monitoring_frame.pack_forget()
        self.session_logs_frame.pack_forget()
        self.home_frame.pack(fill=tk.BOTH, expand=True)
        self.window.unbind('<g>')
        self.window.unbind('<s>')

    def show_monitoring(self):
        self.session_logs_frame.pack_forget()
        self.home_frame.pack_forget()
        self.monitoring_frame.pack(fill=tk.BOTH, expand=True)
        self.window.bind('<g>', lambda e: self.save_snapshot(0))
        self.window.bind('<s>', lambda e: self.save_snapshot(1))
        self.start_camera()

    def show_session_logs(self):
        self.stop_camera()
        self.monitoring_frame.pack_forget()
        self.home_frame.pack_forget()
        self.session_logs_frame.pack(fill=tk.BOTH, expand=True)
        self.window.unbind('<g>')
        self.window.unbind('<s>')
        self.render_session_logs()
        self.session_logs_canvas.yview_moveto(0)

    def load_session_history(self):
        log_filepath = os.path.join("posture_logs", "session_history.json")
        if not os.path.exists(log_filepath):
            return []

        try:
            import json
            with open(log_filepath, "r") as json_file:
                history_records = json.load(json_file)
            if not isinstance(history_records, list):
                return []
            return history_records
        except Exception:
            return []

    def render_session_logs(self):
        for child in self.session_logs_content.winfo_children():
            child.destroy()

        sessions = self.load_session_history()
        if not sessions:
            tk.Label(self.session_logs_content, text="No saved sessions yet.",
                     font=("Arial", 12), bg="#1e1e2e", fg="#a6adc8"
                     ).pack(pady=10)
            return

        # Most recent sessions first
        for session in reversed(sessions):
            entry_frame = tk.Frame(self.session_logs_content, bg="#313244", bd=1, relief=tk.SOLID)
            entry_frame.pack(fill=tk.X, pady=8)

            header_text = session.get("session_date", "Unknown date")
            tk.Label(entry_frame, text=header_text,
                     font=("Arial", 12, "bold"), bg="#313244", fg="#cdd6f4"
                     ).pack(anchor="w", padx=10, pady=(8, 2))

            metrics_frame = tk.Frame(entry_frame, bg="#313244")
            metrics_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

            left_frame = tk.Frame(metrics_frame, bg="#313244")
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            right_frame = tk.Frame(metrics_frame, bg="#313244")
            right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            tk.Label(left_frame, text=f"Total: {session.get('total_duration', 0)}s",
                     font=("Arial", 10), bg="#313244", fg="#cdd6f4"
                     ).pack(anchor="w")
            tk.Label(left_frame, text=f"Good: {session.get('good_posture_duration', 0)}s",
                     font=("Arial", 10), bg="#313244", fg="#a6e3a1"
                     ).pack(anchor="w", pady=2)
            tk.Label(left_frame, text=f"Bad: {session.get('bad_posture_duration', 0)}s",
                     font=("Arial", 10), bg="#313244", fg="#f38ba8"
                     ).pack(anchor="w", pady=2)

            tk.Label(right_frame, text=f"Alerts: {session.get('total_slouch_alerts', 0)}",
                     font=("Arial", 10), bg="#313244", fg="#cdd6f4"
                     ).pack(anchor="w")
            tk.Label(right_frame, text=f"Trained: {session.get('trained_this_session', False)}",
                     font=("Arial", 10), bg="#313244", fg="#cdd6f4"
                     ).pack(anchor="w", pady=2)
            tk.Label(right_frame, text=f"Duration Log: {session.get('total_duration', 0)}s",
                     font=("Arial", 10), bg="#313244", fg="#cdd6f4"
                     ).pack(anchor="w", pady=2)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()
        
        # --- Initialize Session Statistics ---
        self.session_start_wall_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.total_alerts = 0
        self.good_posture_duration = 0.0
        self.bad_posture_duration = 0.0
        self.last_state_timestamp = time.time()
        self.current_session_state = None
        self.trained_this_session = False
        self.session_duration_timer_id = None
        
        if self.landmarker is None:
            self.landmarker = vision.PoseLandmarker.create_from_options(self.options)
        self.running = True
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        self.update_session_duration()

    def stop_camera(self):
        self.running = False
        if hasattr(self, 'session_duration_timer_id') and self.session_duration_timer_id is not None:
            self.window.after_cancel(self.session_duration_timer_id)
            self.session_duration_timer_id = None
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'landmarker') and self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def format_duration(self, total_seconds):
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _bind_session_logs_mousewheel(self, event):
        self.session_logs_canvas.bind_all("<MouseWheel>", self.on_session_logs_mousewheel)
        self.session_logs_canvas.bind_all("<Button-4>", self.on_session_logs_mousewheel)
        self.session_logs_canvas.bind_all("<Button-5>", self.on_session_logs_mousewheel)

    def _unbind_session_logs_mousewheel(self, event):
        self.session_logs_canvas.unbind_all("<MouseWheel>")
        self.session_logs_canvas.unbind_all("<Button-4>")
        self.session_logs_canvas.unbind_all("<Button-5>")

    def on_session_logs_mousewheel(self, event):
        delta = 0
        if hasattr(event, 'delta') and event.delta:
            delta = int(-1 * (event.delta / 120))
        elif getattr(event, 'num', None) == 4:
            delta = -1
        elif getattr(event, 'num', None) == 5:
            delta = 1

        if delta:
            self.session_logs_canvas.yview_scroll(delta, "units")

    def update_session_duration(self):
        total_duration = 0.0
        if hasattr(self, 'good_posture_duration') and hasattr(self, 'bad_posture_duration'):
            total_duration = self.good_posture_duration + self.bad_posture_duration
        if self.current_session_state in (0, 1) and self.last_state_timestamp is not None:
            total_duration += time.time() - self.last_state_timestamp
        if hasattr(self, 'session_duration_lbl'):
            self.session_duration_lbl.config(text=self.format_duration(total_duration))
        self.session_duration_timer_id = self.window.after(500, self.update_session_duration)

    # --- Geometric Math Utilities ---
    def get_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def extract_features(self, landmarks):
        nose  = [landmarks[0].x,  landmarks[0].y]
        l_sh  = [landmarks[11].x, landmarks[11].y]
        r_sh  = [landmarks[12].x, landmarks[12].y]
        l_hip = [landmarks[23].x, landmarks[23].y]
        r_hip = [landmarks[24].x, landmarks[24].y]

        sh_mid  = [(l_sh[0] + r_sh[0]) / 2,  (l_sh[1] + r_sh[1]) / 2]
        hip_mid = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]

        head_forward  = nose[1] - sh_mid[1]
        spine_angle   = self.get_angle(sh_mid, hip_mid, [hip_mid[0], hip_mid[1] + 1])
        shoulder_tilt = abs(l_sh[1] - r_sh[1])

        return {
            "head_forward":   head_forward,
            "spine_angle":    spine_angle,
            "shoulder_tilt":  shoulder_tilt,
        }

    # --- Training Operations ---
    def train_model_from_local_data(self):
        # Look for all variations of files you specified
        csv_files = glob.glob("posture_data_csv/posture_data*.csv")
        if not csv_files:
            self.is_trained = False
            return

        try:
            dataframes = [pd.read_csv(f) for f in csv_files]
            if not dataframes:
                return
                
            df = pd.concat(dataframes, ignore_index=True)
            if df.empty or 'label' not in df.columns:
                return

            X = df.drop('label', axis=1)
            y = df['label']


            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            self.is_trained = True

            # Ensure models directory exists
            os.makedirs("posture_models", exist_ok=True)

            # Backup all existing .pkl files before saving new one
            for old_pkl in glob.glob("posture_models/posture_model_*.pkl"):
                os.rename(old_pkl, old_pkl + ".bkp")

            # Save new model with creation timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            joblib.dump(self.model, f"posture_models/posture_model_{timestamp}.pkl")
            print(f"Saved model to posture_models/posture_model_{timestamp}.pkl")

            print(f"Successfully trained local model on {len(df)} samples from {csv_files}!")
            
            # Flash UI visual confirmation if running
            if hasattr(self, 'status_lbl'):
                self.status_lbl.config(text="MODEL READY", fg="#a6e3a1")
        except Exception as e:
            print(f"Error parsing local CSV datasets: {e}")

    def _train_in_memory(self):
        csv_files = glob.glob("posture_data_csv/posture_data*.csv")
        if not csv_files:
            self.is_trained = False
            return

        try:
            dataframes = [pd.read_csv(f) for f in csv_files]
            if not dataframes:
                return

            df = pd.concat(dataframes, ignore_index=True)
            if df.empty or 'label' not in df.columns:
                return

            X = df.drop('label', axis=1)
            y = df['label']

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            self.is_trained = True

            print(f"Trained in-memory model on {len(df)} samples from {csv_files}!")

            if hasattr(self, 'status_lbl'):
                self.status_lbl.config(text="MODEL READY", fg="#a6e3a1")
        except Exception as e:
            print(f"Error parsing local CSV datasets: {e}")

    def export_model_weights(self):
        if not self.is_trained or self.model is None:
            print("No model to export — collect data first.")
            return

        os.makedirs("posture_models", exist_ok=True)

        for old_pkl in glob.glob("posture_models/posture_model_*.pkl"):
            os.rename(old_pkl, old_pkl + ".bkp")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(self.model, f"posture_models/posture_model_{timestamp}.pkl")
        print(f"Exported model to posture_models/posture_model_{timestamp}.pkl")

        if hasattr(self, 'status_lbl'):
            self.status_lbl.config(text="WEIGHTS EXPORTED", fg="#a6e3a1")
            self.window.after(1500, lambda: self.status_lbl.config(
                text="MODEL READY" if self.is_trained else "NO MODEL TRAINED"
            ))

    def save_snapshot(self, label_val):
        if self.current_features is None:
            return
        
        self.trained_this_session = True
            
        file_exists = os.path.exists(self.csv_filename)
        row = self.current_features.copy()
        row["label"] = label_val

        with open(self.csv_filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows([row])

        # Soft UI temporary flash confirmation
        current_bg = self.status_lbl.cget("bg")
        self.status_lbl.config(bg="#fab387", text="SNAPSHOT + TRAINED")
        self.window.after(400, lambda: self.status_lbl.config(bg=current_bg))

        # Retrain model immediately with updated data
        self._train_in_memory()

    # --- Core Video Processing Core ---
    def video_loop(self):
        relevant_indices = {0, 11, 12}
        connections = [(11, 12), (0, 11), (0, 12)]

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            # 1. Create a safe working copy to prevent GUI thread collisions
            display_frame = cv2.flip(frame, 1)
            h, w, _ = display_frame.shape
            
            # Convert to RGB once for MediaPipe processing
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.time() - self.start_time) * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            # 2. Draw all overlays onto our display_frame buffer
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                
                # Overlay skeleton lines
                for a, b in connections:
                    lm_a, lm_b = landmarks[a], landmarks[b]
                    if lm_a.visibility > 0.5 and lm_b.visibility > 0.5:
                        cv2.line(display_frame, (int(lm_a.x*w), int(lm_a.y*h)), (int(lm_b.x*w), int(lm_b.y*h)), (226, 214, 180), 2)
                
                # Overlay joint dots
                for idx, lm in enumerate(landmarks):
                    if idx in relevant_indices and lm.visibility > 0.5:
                        cv2.circle(display_frame, (int(lm.x*w), int(lm.y*h)), 5, (135, 180, 249), -1)

                all_visible = all(landmarks[i].visibility > 0.5 for i in relevant_indices)
                
                if all_visible:
                    self.current_features = self.extract_features(landmarks)
                    
                    # Update numerical labels dynamically
                    self.head_lbl.config(text=f"Head Displacement: {self.current_features['head_forward']:.3f}")
                    self.spine_lbl.config(text=f"Vertebral Angle:  {self.current_features['spine_angle']:.1f}°")

                    if self.is_trained and self.model:
                        feat_df = pd.DataFrame([self.current_features])
                        raw_prediction = self.model.predict(feat_df)[0]
                        
                        # 1. Append the raw model guess to our rolling history array
                        self.prediction_history.append(raw_prediction)
                        
                        # Keep our historical sliding window locked to the max size
                        if len(self.prediction_history) > self.buffer_size:
                            self.prediction_history.pop(0)
                        
                        # 2. Run a majority vote over our recent time buffer
                        # (Takes whichever label appears most frequently)
                        stabilized_prediction = max(set(self.prediction_history), key=self.prediction_history.count)
                        

                        # --- Data Recording ---
                        current_time = time.time()
                        if self.last_state_timestamp is not None:
                            elapsed_chunk = current_time - self.last_state_timestamp
                            if self.current_session_state == 0:
                                self.good_posture_duration += elapsed_chunk
                            elif self.current_session_state == 1:
                                self.bad_posture_duration += elapsed_chunk
                        
                        # Set current state context for the next cycle calculation
                        self.current_session_state = stabilized_prediction
                        self.last_state_timestamp = current_time
                        
                        # 3. Update UI states using the clean, smoothed result
                        if stabilized_prediction == 0:
                            self.status_lbl.config(text="GOOD POSTURE", fg="#a6e3a1", bg="#313244")
                            if hasattr(self, 'status_card'): 
                                self.status_card.config(highlightbackground="#a6e3a1")
                            
                            # Reset slouch time tracking since posture is good
                            self.slouch_start_time = None
                        else:
                            self.status_lbl.config(text="SLOUCH ALERT", fg="#f38ba8", bg="#512530")
                            if hasattr(self, 'status_card'): 
                                self.status_card.config(highlightbackground="#f38ba8")
                            
                            # Start clock if this is the beginning of a slouch stretch
                            if self.slouch_start_time is None:
                                self.slouch_start_time = time.time()
                            elif not self.alertpopup_active:
                                # Calculate continuous elapsed slouch time
                                elapsed_slouch = time.time() - self.slouch_start_time
                                
                                # If they breach the limit and a popup isn't already active, trigger alert
                                if elapsed_slouch >= self.max_slouch_seconds:
                                    self.alertpopup_active = True
                                    self.total_alerts += 1
                                    # Reset tracking clock so it doesn't loop fire while the box is open
                                    self.slouch_start_time = None 
                                    # Safely pass popup command back to Tkinter's main thread
                                    self.window.after(0, self.trigger_alert_popup)
                else:
                    self.current_features = None
                    self.last_state_timestamp = time.time()
            else:
                self.current_features = None
                self.last_state_timestamp = time.time()

            # 3. Double-buffered UI rendering: Convert the completed frame
            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # 4. Atomically swap the image asset into the UI label to prevent partial draw drops
            if self.running:
                self.window.after(0, self.update_cam_label, img_tk)

            # Cap frame loop calculation steps to match ~60 FPS update ceilings
            time.sleep(0.016)

    def update_cam_label(self, img_tk):
        """Helper to ensure image assignment happens safely on Tkinter's main thread"""
        if self.running:
            self.cam_label.img_tk = img_tk
            self.cam_label.config(image=img_tk)

    def trigger_alert_popup(self):
        """Displays a thread-safe warning popup on the primary desktop layer"""
        self.window.bell() 
        
        messagebox.showwarning(
            "Posture Reminder", 
            f"You have been slouching for over {self.max_slouch_seconds} seconds!\n"
        )
        
        # When the user clicks "OK", lower the flag so the system can watch for future slouch windows
        self.alertpopup_active = False

    def save_session_summary_json(self):
        """Finalizes time tracking variables and commits the session metrics to a JSON log"""
        if self.session_start_wall_time is None:
            return

        # Factor in the final remaining time block right up to the close action
        current_time = time.time()
        if self.last_state_timestamp is not None:
            elapsed_chunk = current_time - self.last_state_timestamp
            if self.current_session_state == 0:
                self.good_posture_duration += elapsed_chunk
            elif self.current_session_state == 1:
                self.bad_posture_duration += elapsed_chunk

        total_session_duration = self.good_posture_duration + self.bad_posture_duration

        # Guard against saving empty metrics files if the session was immediately skipped
        if total_session_duration < 1.0:
            return

        session_payload = {
            "session_date": self.session_start_wall_time,
            "total_duration": round(total_session_duration, 2),
            "good_posture_duration": round(self.good_posture_duration, 2),
            "bad_posture_duration": round(self.bad_posture_duration, 2),
            "total_slouch_alerts": self.total_alerts,
            "trained_this_session": self.trained_this_session
        }

        log_directory = "posture_logs"
        log_filepath = os.path.join(log_directory, "session_history.json")
        os.makedirs(log_directory, exist_ok=True)

        import json
        history_records = []
        
        # If history ledger already exists, read old logs to append new entry cleanly
        if os.path.exists(log_filepath):
            try:
                with open(log_filepath, "r") as json_file:
                    history_records = json.load(json_file)
                    if not isinstance(history_records, list):
                        history_records = []
            except Exception:
                history_records = []

        history_records.append(session_payload)

        with open(log_filepath, "w") as json_file:
            json.dump(history_records, json_file, indent=4)
        print(f"Session data successfully compiled and saved to {log_filepath}")

    def on_close(self):
        if hasattr(self, 'running') and self.running:
            self.save_session_summary_json()
        self.stop_camera()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureApp(root)
    root.mainloop()

import cv2
import csv
import glob
import os
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import numpy as np
import pandas as pd
import json

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
        self.window.configure(bg="#1f1f1e")

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
        
        # --- Active UI References & Context ---
        self.active_cam_label = None
        self.active_head_lbl = None
        self.active_spine_lbl = None
        self.active_status_lbl = None
        self.session_context = None  # "monitoring" or "settings"
        
        # --- Rolling Buffer for Posture Stabilization ---
        self.prediction_history = []
        self.buffer_size = 15  # Tracks last 15 frames for majority voting
        
        # --- Time-based Alert Tracking ---
        self.slouch_start_time = None
        self.alertpopup_active = False       # Flag to prevent multiple popups from staking up
        self.max_slouch_seconds = self.load_settings()    # Time threshold before triggering an alert Sourced from settings file
        self.alert_dismiss_timer_id = None                # Pending auto-dismiss timer ID
        
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
        style.configure("TLabel", background="#30302e", foreground="#f1f0ec", font=("Arial", 11))
        style.configure("TButton", font=("Arial", 10), background="#30302e", foreground="#f1f0ec")
        style.map("TButton", background=[("active", "#1f1f1e"), ("hover", "#1f1f1e")])

    def create_widgets(self):
        # Page container — holds both home and monitoring pages
        self.page_container = tk.Frame(self.window, bg="#1f1f1e")
        self.page_container.pack(fill=tk.BOTH, expand=True)

        # ===================== Home Page =====================
        self.home_frame = tk.Frame(self.page_container, bg="#1f1f1e")

        # Title Section
        tk.Label(self.home_frame, text="Personal Posture Journal",
                 font=("Arial", 28), bg="#1f1f1e", fg="#f1f0ec"
                 ).pack(pady=(120, 10))
        tk.Label(self.home_frame, text="Real-time posture monitoring app",
                 font=("Arial", 14), bg="#1f1f1e", fg="#f1f0ec"
                 ).pack(pady=(0, 60))

        # Start Session Button
        tk.Button(self.home_frame, text="Start Session",
                  font=("Arial", 16), bg="#30302e", fg="#f1f0ec", activebackground="#1f1f1e", activeforeground="#f1f0ec",highlightbackground="#f1f0ec",
                  width=20, height=2, command=self.show_monitoring
                  ).pack(pady=20)

        # Placeholder Buttons Row
        btn_frame = tk.Frame(self.home_frame, bg="#1f1f1e")
        btn_frame.pack(pady=40)

        tk.Button(btn_frame, text="Settings", font=("Arial", 11),
                   bg="#30302e", fg="#f1f0ec", activebackground="#1f1f1e", activeforeground="#f1f0ec",highlightbackground="#f1f0ec", width=12, command=self.show_settings
                  ).pack(side=tk.LEFT, padx=10)

        tk.Button(btn_frame, text="Logs", font=("Arial", 11),
                   bg="#30302e", fg="#f1f0ec", activebackground="#1f1f1e", activeforeground="#f1f0ec",highlightbackground="#f1f0ec", width=12, command=self.show_session_logs
                  ).pack(side=tk.LEFT, padx=10)

        exit_frame = tk.Frame(self.home_frame, bg="#1f1f1e")
        exit_frame.pack(pady=(0, 40))

        tk.Button(exit_frame, text="Exit", font=("Arial", 11),
                   bg="#30302e", fg="#f1f0ec", activebackground="#1f1f1e", activeforeground="#f1f0ec",highlightbackground="#f1f0ec", width=12, command=self.on_close
                  ).pack()

        # ===================== Session Logs Page =====================
        self.session_logs_frame = tk.Frame(self.page_container, bg="#1f1f1e")
        session_top = tk.Frame(self.session_logs_frame, bg="#1f1f1e")
        session_top.pack(fill=tk.X, padx=20, pady=20)
        tk.Button(session_top, text="← Back to Home",
                  font=("Arial", 10),  bg="#30302e", fg="#f1f0ec", activebackground="#1f1f1e", activeforeground="#f1f0ec",highlightbackground="#f1f0ec",
                  command=self.show_home
                  ).pack(side=tk.LEFT)
        tk.Label(self.session_logs_frame, text="Session Logs",
                 font=("Arial", 24), bg="#1f1f1e", fg="#f1f0ec"
                 ).pack(pady=(10, 20))

        self.session_logs_list_container = tk.Frame(self.session_logs_frame, bg="#1f1f1e")
        self.session_logs_list_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        self.session_logs_canvas = tk.Canvas(self.session_logs_list_container,
                                            bg="#1f1f1e", highlightthickness=0, bd=0)
        self.session_logs_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        session_scrollbar = tk.Scrollbar(self.session_logs_list_container,
                                         orient=tk.VERTICAL, command=self.session_logs_canvas.yview)
        session_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.session_logs_canvas.configure(yscrollcommand=session_scrollbar.set)

        self.session_logs_content = tk.Frame(self.session_logs_canvas, bg="#1f1f1e")
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
                                                  font=("Arial", 12), bg="#1f1f1e", fg="#f1f0ec")
        self.session_logs_empty_label.pack(pady=10)

        # ===================== Monitoring Page =====================
        self.monitoring_frame = tk.Frame(self.page_container, bg="#1f1f1e")

        # Left Panel - Camera Feed
        self.left_frame = tk.Frame(self.monitoring_frame, bg="#1f1f1e")
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        self.cam_label = tk.Label(self.left_frame, bg="#30302e", width=640, height=480)
        self.cam_label.pack(fill=tk.BOTH, expand=True)

        # Right Panel - Control Center
        self.right_frame = tk.Frame(self.monitoring_frame, bg="#30302e", width=300)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 20), pady=20)
        self.right_frame.pack_propagate(False)

        # End Session Button
        tk.Button(self.right_frame, text="⏹ End Session",
                  font=("Arial", 10), bg="#30302e", fg="#f1f0ec", activebackground="#1f1f1e", activeforeground="#f1f0ec",highlightbackground="#f1f0ec",
                  command=self.show_home
                  ).pack(fill=tk.X, padx=15, pady=(15, 5))

        # Status Display
        tk.Label(self.right_frame, text="STATUS",
                 font=("Arial", 14), bg="#30302e", fg="#f1f0ec"
                 ).pack(pady=(20, 5))
        self.status_lbl = tk.Label(self.right_frame, text=("UNAVAILABLE" if self.is_trained else "NO MODEL TRAINED"),
                                   font=("Arial", 18),
                                   bg="#30302e", fg="#f38ba8", width=18, pady=10)
        self.status_lbl.pack(pady=10)

        # Live Metrics
        self.metrics_frame = tk.LabelFrame(self.right_frame, text=" Live Metrics ",
                                           bg="#30302e", fg="#f1f0ec",
                                           font=("Arial", 10), padx=10, pady=10)
        self.metrics_frame.pack(fill=tk.X, padx=15, pady=15)

        self.head_lbl = ttk.Label(self.metrics_frame, text="Head Forward: --")
        self.head_lbl.pack(anchor="w", pady=2)
        self.spine_lbl = ttk.Label(self.metrics_frame, text="Spine Angle: --")
        self.spine_lbl.pack(anchor="w", pady=2)

        # Data Collection Section
        self.train_frame = tk.LabelFrame(self.right_frame, text=" Data Collection ",
                                         bg="#30302e", fg="#f1f0ec",
                                         font=("Arial", 10), padx=10, pady=10)
        self.train_frame.pack(fill=tk.X, padx=15, pady=15)

        ttk.Label(self.train_frame, text="Collect new snapshots:").pack(pady=5)

        tk.Button(self.train_frame, text="Capture GOOD (G)",
                  bg="#576B57", fg="#a6e3a1", activebackground="#1f1f1e", activeforeground="#f1f0ec",highlightbackground="#f1f0ec", font=("Arial", 10),
                  command=lambda: self.save_snapshot(0)
                  ).pack(fill=tk.X, pady=4)

        tk.Button(self.train_frame, text="Capture SLOUCH (S)",
                  bg="#512530", fg="#f38ba8", activebackground="#1f1f1e", activeforeground="#f1f0ec",highlightbackground="#f1f0ec",font=("Arial", 10),
                  command=lambda: self.save_snapshot(1)
                  ).pack(fill=tk.X, pady=4)


        # Session Duration Display
        duration_frame = tk.Frame(self.right_frame, bg="#30302e")
        duration_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        tk.Label(duration_frame, text="SESSION DURATION",
                 font=("Arial", 10), bg="#30302e", fg="#f1f0ec"
                 ).pack(anchor="w")
        self.session_duration_lbl = tk.Label(duration_frame, text="00:00:00",
                                             font=("Arial", 14),
                                             bg="#30302e", fg="#f1f0ec",
                                             width=18, pady=10)
        self.session_duration_lbl.pack(fill=tk.X)

        # ===================== Settings Page =====================
        self.settings_frame = tk.Frame(self.page_container, bg="#1f1f1e")

        # Left Panel - Camera Feed Calibration
        self.settings_left_frame = tk.Frame(self.settings_frame, bg="#1f1f1e")
        self.settings_left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        self.settings_cam_label = tk.Label(self.settings_left_frame, bg="#30302e", width=640, height=480)
        self.settings_cam_label.pack(fill=tk.BOTH, expand=True)

        # Right Panel - Configuration Center
        self.settings_right_frame = tk.Frame(self.settings_frame, bg="#30302e", width=300)
        self.settings_right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 20), pady=20)
        self.settings_right_frame.pack_propagate(False)

        # Back to Home Button
        tk.Button(self.settings_right_frame, text="← Back to Home",
                  font=("Arial", 10), bg="#30302e", fg="#f1f0ec", activebackground="#1f1f1e", activeforeground="#f1f0ec",highlightbackground="#f1f0ec",
                  command=self.show_home
                  ).pack(fill=tk.X, padx=15, pady=(15, 5))

        # Status Display
        tk.Label(self.settings_right_frame, text="POSTURE STATUS",
                 font=("Arial", 14), bg="#30302e", fg="#f1f0ec"
                 ).pack(pady=(20, 5))
        self.settings_status_lbl = tk.Label(self.settings_right_frame, text="NO MODEL TRAINED",
                                            font=("Arial", 18),
                                            bg="#512530", fg="#f38ba8", width=18, pady=10)
        self.settings_status_lbl.pack(pady=10)

        # Live Metrics
        self.settings_metrics_frame = tk.LabelFrame(self.settings_right_frame, text=" Live Metrics ",
                                                   bg="#30302e", fg="#f1f0ec",
                                                   font=("Arial", 10), padx=10, pady=10)
        self.settings_metrics_frame.pack(fill=tk.X, padx=15, pady=15)

        self.settings_head_lbl = ttk.Label(self.settings_metrics_frame, text="Head Forward: --")
        self.settings_head_lbl.pack(anchor="w", pady=2)
        self.settings_spine_lbl = ttk.Label(self.settings_metrics_frame, text="Spine Angle: --")
        self.settings_spine_lbl.pack(anchor="w", pady=2)

        # Configurable Alert Timer Section
        self.alert_timer_frame = tk.LabelFrame(self.settings_right_frame, text=" Slouch Threshold ",
                                               bg="#30302e", fg="#f1f0ec",
                                               font=("Arial", 10), padx=10, pady=10)
        self.alert_timer_frame.pack(fill=tk.X, padx=15, pady=10)

        ttk.Label(self.alert_timer_frame, text="Delay (seconds):").pack(side=tk.LEFT, padx=5)
        self.settings_delay_var = tk.StringVar()
        self.settings_spinbox = ttk.Spinbox(
            self.alert_timer_frame,
            from_=1, to=300, increment=1,
            textvariable=self.settings_delay_var,
            width=8,
            command=self.on_spinbox_change
        )
        self.settings_spinbox.pack(side=tk.LEFT, padx=5)
        self.settings_spinbox.bind("<FocusOut>", lambda e: self.on_spinbox_change())
        self.settings_spinbox.bind("<Return>", lambda e: self.on_spinbox_change())

        # Data Collection Section
        self.settings_train_frame = tk.LabelFrame(self.settings_right_frame, text=" Data Collection ",
                                                 bg="#30302e", fg="#f1f0ec",
                                                 font=("Arial", 10), padx=10, pady=10)
        self.settings_train_frame.pack(fill=tk.X, padx=15, pady=10)

        ttk.Label(self.settings_train_frame, text="Collect calibration snapshots:").pack(pady=5)
        tk.Button(self.settings_train_frame, text="Capture GOOD (G)",
                  bg="#576B57", fg="#a6e3a1", activebackground="#1f1f1e", activeforeground="#f1f0ec",highlightbackground="#f1f0ec", font=("Arial", 10),
                  command=lambda: self.save_snapshot(0)
                  ).pack(fill=tk.X, pady=4)

        tk.Button(self.settings_train_frame, text="Capture SLOUCH (S)",
                  bg="#512530", fg="#f38ba8", activebackground="#1f1f1e", activeforeground="#f1f0ec",highlightbackground="#f1f0ec", font=("Arial", 10),
                  command=lambda: self.save_snapshot(1)
                  ).pack(fill=tk.X, pady=4)

        tk.Button(self.settings_right_frame, text="💾 Save Settings",
                  bg="#30302e", fg="#f1f0ec", activebackground="#1f1f1e", activeforeground="#f1f0ec",highlightbackground="#f1f0ec", font=("Arial", 12),
                  command=self.save_settings_from_spinbox
                  ).pack(fill=tk.X, padx=15, pady=(5, 20))

    def on_spinbox_change(self):
        pass

    def save_settings_from_spinbox(self):
        try:
            val = int(self.settings_delay_var.get())
            if val < 1:
                val = 1
            self.save_settings(val)
            self.show_home()
        except ValueError:
            pass

    def show_home(self):
        # Save session tracking statistics before winding down the interface
        if hasattr(self, 'running') and self.running and self.session_context == "monitoring":
            self.save_session_summary_json()
        
        self.stop_camera()
        self.monitoring_frame.pack_forget()
        self.settings_frame.pack_forget()
        self.session_logs_frame.pack_forget()
        self.home_frame.pack(fill=tk.BOTH, expand=True)
        self.window.unbind('<g>')
        self.window.unbind('<s>')

    def show_monitoring(self):
        self.session_logs_frame.pack_forget()
        self.home_frame.pack_forget()
        self.monitoring_frame.pack(fill=tk.BOTH, expand=True)
        
        self.active_cam_label = self.cam_label
        self.active_head_lbl = self.head_lbl
        self.active_spine_lbl = self.spine_lbl
        self.active_status_lbl = self.status_lbl
        
        self.window.bind('<g>', lambda e: self.save_snapshot(0))
        self.window.bind('<s>', lambda e: self.save_snapshot(1))
        self.start_camera(context="monitoring")

    def show_settings(self):
        self.home_frame.pack_forget()
        self.settings_frame.pack(fill=tk.BOTH, expand=True)
        
        self.active_cam_label = getattr(self, 'settings_cam_label', None)
        self.active_head_lbl = getattr(self, 'settings_head_lbl', None)
        self.active_spine_lbl = getattr(self, 'settings_spine_lbl', None)
        self.active_status_lbl = getattr(self, 'settings_status_lbl', None)
        
        saved_val = self.load_settings()
        self.settings_delay_var.set(str(saved_val))
        
        self.window.bind('<g>', lambda e: self.save_snapshot(0))
        self.window.bind('<s>', lambda e: self.save_snapshot(1))
        self.start_camera(context="settings")

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
                     font=("Arial", 12), bg="#1f1f1e", fg="#f1f0ec"
                     ).pack(pady=10)
            return

        # Most recent sessions first
        for session in reversed(sessions):
            entry_frame = tk.Frame(self.session_logs_content, bg="#30302e", bd=1, relief=tk.SOLID)
            entry_frame.pack(fill=tk.X, pady=8)

            header_text = session.get("session_date", "Unknown date")
            tk.Label(entry_frame, text=header_text,
                     font=("Arial", 12), bg="#30302e", fg="#f1f0ec"
                     ).pack(anchor="w", padx=10, pady=(8, 2))

            metrics_frame = tk.Frame(entry_frame, bg="#30302e")
            metrics_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

            left_frame = tk.Frame(metrics_frame, bg="#30302e")
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            right_frame = tk.Frame(metrics_frame, bg="#30302e")
            right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            tk.Label(left_frame, text=f"Total: {session.get('total_duration', 0)}s",
                     font=("Arial", 10), bg="#30302e", fg="#f1f0ec"
                     ).pack(anchor="w")
            tk.Label(left_frame, text=f"Good: {session.get('good_posture_duration', 0)}s",
                     font=("Arial", 10), bg="#30302e", fg="#a6e3a1"
                     ).pack(anchor="w", pady=2)
            tk.Label(left_frame, text=f"Bad: {session.get('bad_posture_duration', 0)}s",
                     font=("Arial", 10), bg="#30302e", fg="#f38ba8"
                     ).pack(anchor="w", pady=2)

            tk.Label(right_frame, text=f"Alerts: {session.get('total_slouch_alerts', 0)}",
                     font=("Arial", 10), bg="#30302e", fg="#f1f0ec"
                     ).pack(anchor="w")
            tk.Label(right_frame, text=f"Trained: {session.get('trained_this_session', False)}",
                     font=("Arial", 10), bg="#30302e", fg="#f1f0ec"
                     ).pack(anchor="w", pady=2)
            tk.Label(right_frame, text=f"Duration Log: {session.get('total_duration', 0)}s",
                     font=("Arial", 10), bg="#30302e", fg="#f1f0ec"
                     ).pack(anchor="w", pady=2)

    def start_camera(self, context="monitoring"):
        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()
        self.session_context = context
        if context == "monitoring":
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
        if context == "monitoring":
            self.update_session_duration()

    def stop_camera(self):
        self.running = False
        if hasattr(self, 'session_duration_timer_id') and self.session_duration_timer_id is not None:
            self.window.after_cancel(self.session_duration_timer_id)
            self.session_duration_timer_id = None
        if self.alertpopup_active:
            self.dismiss_alert_popup()
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
        l_ear = [landmarks[7].x,  landmarks[7].y]
        r_ear = [landmarks[8].x,  landmarks[8].y]
        l_sh  = [landmarks[11].x, landmarks[11].y]
        r_sh  = [landmarks[12].x, landmarks[12].y]

        sh_mid   = [(l_sh[0] + r_sh[0]) / 2,   (l_sh[1] + r_sh[1]) / 2]
        head_mid = [(l_ear[0] + r_ear[0]) / 2, (l_ear[1] + r_ear[1]) / 2]

        head_forward = nose[0] - sh_mid[0]

        spine_angle = self.get_angle(head_mid, sh_mid, [sh_mid[0], sh_mid[1] + 1])

        shoulder_tilt = abs(l_sh[1] - r_sh[1])

        return {
            "head_forward":   head_forward,
            "spine_angle":    spine_angle,
            "shoulder_tilt":  shoulder_tilt,
        }

    def load_settings(self):
        """Loads settings from settings.json or initializes a default if missing."""
        self.settings_filepath = "settings.json"
        default_delay = 10
        if os.path.exists(self.settings_filepath):
            try:
                with open(self.settings_filepath, "r") as f:
                    data = json.load(f)
                    return data.get("max_slouch_seconds", default_delay)
            except Exception:
                pass
        # Save default if file didn't exist or failed to load
        self.save_settings(default_delay)
        return default_delay

    def save_settings(self, new_seconds):
        """Saves current settings back to settings.json."""
        self.max_slouch_seconds = new_seconds
        with open("settings.json", "w") as f:
            json.dump({"max_slouch_seconds": new_seconds}, f, indent=4)

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
            if self.active_status_lbl and self.active_status_lbl.winfo_exists():
                self.active_status_lbl.config(text="MODEL READY", fg="#a6e3a1")
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

            if self.active_status_lbl and self.active_status_lbl.winfo_exists():
                self.active_status_lbl.config(text="MODEL READY", fg="#a6e3a1")
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

        if self.active_status_lbl and self.active_status_lbl.winfo_exists():
            self.active_status_lbl.config(text="WEIGHTS EXPORTED", fg="#a6e3a1")
            self.window.after(1500, lambda: self.active_status_lbl.config(
                text="MODEL READY" if self.is_trained else "NO MODEL TRAINED"
            ) if self.active_status_lbl and self.active_status_lbl.winfo_exists() else None)

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
        current_bg = self.active_status_lbl.cget("bg")
        self.active_status_lbl.config(bg="#fab387", text="SNAPSHOT + TRAINED")
        self.window.after(400, lambda: self.active_status_lbl.config(bg=current_bg) if self.active_status_lbl and self.active_status_lbl.winfo_exists() else None)

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
                    self.active_head_lbl.config(text=f"Head Displacement: {self.current_features['head_forward']:.3f}")
                    self.active_spine_lbl.config(text=f"Vertebral Angle:  {self.current_features['spine_angle']:.1f}°")

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
                            self.active_status_lbl.config(text="GOOD POSTURE", fg="#a6e3a1", bg="#576B57")
                            if hasattr(self, 'status_card'): 
                                self.status_card.config(highlightbackground="#a6e3a1")

                            # Reset slouch time tracking since posture is good
                            self.slouch_start_time = None

                            # Schedule alert auto-dismiss with a short delay
                            if self.alertpopup_active and self.alert_dismiss_timer_id is None:
                                self.alert_dismiss_timer_id = self.window.after(
                                    0, self.dismiss_alert_popup
                                )
                        else:
                            self.active_status_lbl.config(text="SLOUCH ALERT", fg="#f38ba8", bg="#512530")
                            if hasattr(self, 'status_card'): 
                                self.status_card.config(highlightbackground="#f38ba8")

                            # Cancel pending popup dismissal if user slouches again during cooldown
                            if self.alert_dismiss_timer_id is not None:
                                self.window.after_cancel(self.alert_dismiss_timer_id)
                                self.alert_dismiss_timer_id = None

                            # Start clock if this is the beginning of a slouch stretch
                            if self.slouch_start_time is None:
                                self.slouch_start_time = time.time()
                            elif not self.alertpopup_active:
                                # Calculate continuous elapsed slouch time
                                elapsed_slouch = time.time() - self.slouch_start_time
                                
                                # If they breach the limit and a popup isn't already active, trigger alert
                                if elapsed_slouch >= self.max_slouch_seconds:
                                    self.alertpopup_active = True
                                    self.slouch_start_time = None
                                    if self.session_context == "monitoring":
                                        self.total_alerts += 1
                                        self.window.after(0, self.trigger_alert_popup)
                else:
                    self.active_status_lbl.config(text="NO USER FOUND", bg="#30302e", fg="#f1f0ec")
                    self.current_features = None
                    self.last_state_timestamp = time.time()
            else:
                self.active_status_lbl.config(text="NO USER FOUND", bg="#30302e", fg="#f1f0ec")
                self.current_features = None
                self.last_state_timestamp = time.time()

            # Full-frame slouch alert overlay
            if self.alertpopup_active:
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (48, 37, 81), -1)
                cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)

                cv2.putText(display_frame, "SLOUCHING!",
                            (w // 2 - 250, h // 2 - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 5)

                cv2.putText(display_frame, "Fix your posture!",
                            (w // 2 - 200, h // 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (168, 139, 243), 3)

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
            self.active_cam_label.img_tk = img_tk
            self.active_cam_label.config(image=img_tk)

    def trigger_alert_popup(self):
        """Plays an audio alert when slouch threshold is exceeded."""
        self.window.bell()

    def dismiss_alert_popup(self):
        """Clears the alert state when posture returns to good."""
        if self.alert_dismiss_timer_id is not None:
            self.window.after_cancel(self.alert_dismiss_timer_id)
            self.alert_dismiss_timer_id = None
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
        self.export_model_weights()
        self.stop_camera()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureApp(root)
    root.mainloop()

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

        # --- Camera Thread ---
        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()
        self.running = True
        
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

        # Handle window close cleanly
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Arial", 11))
        style.configure("TButton", font=("Arial", 10, "bold"), background="#89b4fa", foreground="#11111b")
        style.map("TButton", background=[("active", "#b4befe")])

    def create_widgets(self):
        # Left Panel - Camera Feed
        self.left_frame = tk.Frame(self.window, bg="#1e1e2e")
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        self.cam_label = tk.Label(self.left_frame, bg="#313244", width=640, height=480)
        self.cam_label.pack(fill=tk.BOTH, expand=True)

        # Right Panel - Control Center
        self.right_frame = tk.Frame(self.window, bg="#181825", width=300)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 20), pady=20)
        self.right_frame.pack_propagate(False)

        # Status Display
        tk.Label(self.right_frame, text="POSTURE STATUS", font=("Arial", 14, "bold"), bg="#181825", fg="#a6adc8").pack(pady=(20, 5))
        self.status_lbl = tk.Label(self.right_frame, text="NO MODEL TRAINED", font=("Arial", 18, "bold"), bg="#313244", fg="#f38ba8", width=18, pady=10)
        self.status_lbl.pack(pady=10)

        # Live Metrics
        self.metrics_frame = tk.LabelFrame(self.right_frame, text=" Live Metrics ", bg="#181825", fg="#cdd6f4", font=("Arial", 10, "bold"), padx=10, pady=10)
        self.metrics_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.head_lbl = ttk.Label(self.metrics_frame, text="Head Forward: --")
        self.head_lbl.pack(anchor="w", pady=2)
        self.spine_lbl = ttk.Label(self.metrics_frame, text="Spine Angle: --")
        self.spine_lbl.pack(anchor="w", pady=2)

        # Interactive Training Section
        self.train_frame = tk.LabelFrame(self.right_frame, text=" Data Collection ", bg="#181825", fg="#cdd6f4", font=("Arial", 10, "bold"), padx=10, pady=10)
        self.train_frame.pack(fill=tk.X, padx=15, pady=15)

        ttk.Label(self.train_frame, text="Collect new snapshots:").pack(pady=5)
        
        btn_good = tk.Button(self.train_frame, text="Capture GOOD (G)", bg="#a6e3a1", fg="#11111b", font=("Arial", 10, "bold"), command=lambda: self.save_snapshot(0))
        btn_good.pack(fill=tk.X, pady=4)
        
        btn_slouch = tk.Button(self.train_frame, text="Capture SLOUCH (S)", bg="#f38ba8", fg="#11111b", font=("Arial", 10, "bold"), command=lambda: self.save_snapshot(1))
        btn_slouch.pack(fill=tk.X, pady=4)

        # Export Action Button
        self.export_btn = tk.Button(self.right_frame, text="📦 EXPORT MODEL WEIGHTS", bg="#fab387", fg="#11111b", font=("Arial", 12, "bold"), command=self.export_model_weights)
        self.export_btn.pack(fill=tk.X, padx=15, pady=20)

        # Bind hotkeys matching your old script layout
        self.window.bind('<g>', lambda e: self.save_snapshot(0))
        self.window.bind('<s>', lambda e: self.save_snapshot(1))

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

        target_fps = 30
        frame_duration = 1.0 / target_fps

        last_valid_frame = None

        while self.running:
            start_frame_time = time.time()

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if last_valid_frame is not None:
                    # Duplicate the last good frame so the UI never displays a black void
                    frame = last_valid_frame.copy()
                else:
                    # If we don't even have a first frame yet, wait patiently
                    time.sleep(0.01)
                    continue
            else:
                last_valid_frame = frame.copy()

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.time() - self.start_time) * 1000)

            try:
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                print(f"MediaPipe inference skip: {e}")
                result = None

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                
                # Render Stick Figure Lines
                for a, b in connections:
                    lm_a, lm_b = landmarks[a], landmarks[b]
                    if lm_a.visibility > 0.5 and lm_b.visibility > 0.5:
                        cv2.line(frame, (int(lm_a.x*w), int(lm_a.y*h)), (int(lm_b.x*w), int(lm_b.y*h)), (166, 227, 161), 2)
                
                # Render Joint Dots
                for idx, lm in enumerate(landmarks):
                    if idx in relevant_indices and lm.visibility > 0.5:
                        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (243, 139, 168), -1)

                all_visible = all(landmarks[i].visibility > 0.5 for i in relevant_indices)
                
                if all_visible:
                    # Update local state calculations
                    self.current_features = self.extract_features(landmarks)
                    
                    # Safe thread-bound GUI push text metrics updates
                    self.head_lbl.config(text=f"Head Forward: {self.current_features['head_forward']:.3f}")
                    self.spine_lbl.config(text=f"Spine Angle: {self.current_features['spine_angle']:.1f}°")

                    # Handle continuous local live inference evaluations
                    if self.is_trained and self.model:
                        feat_df = pd.DataFrame([self.current_features])
                        prediction = self.model.predict(feat_df)[0]
                        
                        if prediction == 0:
                            self.status_lbl.config(text="GOOD POSTURE", fg="#a6e3a1", bg="#313244")
                        else:
                            self.status_lbl.config(text="SLOUCHING!", fg="#f38ba8", bg="#512530")
                else:
                    self.current_features = None
            else:
                self.current_features = None

            # Map the finalized OpenCV output array cleanly into Tkinter Frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)

            if self.running:
                self.cam_label.img_tk = img_tk
                self.cam_label.config(image=img_tk)
                self.cam_label.image = img_tk  # <-- This stops garbage collection on the widget
                self.current_frame_ref = img_tk # <-- This double-locks it in the class instance

            # Dynamic Throttling: Calculate how long processing took 
            # and sleep only for the remaining time left in the frame window.
            elapsed = time.time() - start_frame_time
            sleep_time = max(0.001, frame_duration - elapsed)
            time.sleep(sleep_time)

    def on_close(self):
        self.running = False
        self.cap.release()
        self.landmarker.close()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureApp(root)
    root.mainloop()

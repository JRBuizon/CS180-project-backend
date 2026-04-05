import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import csv
import time
import urllib.request

# Download the model file once
urllib.request.urlretrieve(
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker.task"
)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (0, 11), (0, 12),
]

RELEVANT = {0, 11, 12, 23, 24}

def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    for a, b in POSE_CONNECTIONS:
        if a in RELEVANT and b in RELEVANT:
            lm_a, lm_b = landmarks[a], landmarks[b]
            if lm_a.visibility > 0.5 and lm_b.visibility > 0.5:
                x1, y1 = int(lm_a.x * w), int(lm_a.y * h)
                x2, y2 = int(lm_b.x * w), int(lm_b.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for i, lm in enumerate(landmarks):
        if i in RELEVANT and lm.visibility > 0.5:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame

def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def extract_features(landmarks):
    nose  = [landmarks[0].x,  landmarks[0].y]
    l_sh  = [landmarks[11].x, landmarks[11].y]
    r_sh  = [landmarks[12].x, landmarks[12].y]
    l_hip = [landmarks[23].x, landmarks[23].y]
    r_hip = [landmarks[24].x, landmarks[24].y]

    sh_mid  = [(l_sh[0] + r_sh[0]) / 2,  (l_sh[1] + r_sh[1]) / 2]
    hip_mid = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]

    head_forward  = nose[1] - sh_mid[1]
    spine_angle   = get_angle(sh_mid, hip_mid, [hip_mid[0], hip_mid[1] + 1])
    shoulder_tilt = abs(l_sh[1] - r_sh[1])

    return {
        "head_forward":   head_forward,
        "spine_angle":    spine_angle,
        "shoulder_tilt":  shoulder_tilt,
    }

base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
start_time = time.time()
records = []
last_label = None

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            frame = draw_landmarks(frame, landmarks)

            # Check all relevant landmarks are visible
            all_visible = all(
                landmarks[i].visibility > 0.5 for i in RELEVANT
            )

            key = cv2.waitKey(1) & 0xFF

            if all_visible:
                if key == ord('g'):
                    features = extract_features(landmarks)
                    features["label"] = 0
                    records.append(features)
                    last_label = "good"
                elif key == ord('s'):
                    features = extract_features(landmarks)
                    features["label"] = 1
                    records.append(features)
                    last_label = "slouch"

        else:
            key = cv2.waitKey(1) & 0xFF

        # HUD
        good_count   = sum(1 for r in records if r["label"] == 0)
        slouch_count = sum(1 for r in records if r["label"] == 1)

        cv2.putText(frame, f"Good: {good_count}  Slouch: {slouch_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "G = good  S = slouch  Q = quit & save",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if last_label:
            color = (0, 255, 0) if last_label == "good" else (0, 0, 255)
            cv2.putText(frame, f"Saved: {last_label}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            last_label = None

        cv2.imshow("Posture", frame)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
if records:
    with open("posture_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} records to posture_data.csv")
    print(f"  Good: {good_count}  Slouch: {slouch_count}")
else:
    print("No data recorded.")
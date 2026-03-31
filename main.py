import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ======================
# CONFIG & THRESHOLDS
# ======================
MODEL_PATH = "face_landmarker.task"
EAR_THRESHOLD = 0.22      # Threshold for a blink
WINK_EAR_THRESHOLD = 0.16 # Threshold for a deep wink
BLINK_COOLDOWN = 0.3      # Debounce blinks
WINDOW_TIME = 2.0         # 3 blinks must happen within 2s

# Landmark indices for EAR
LEFT_EYE = [33, 160, 158, 133, 153, 144] 
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ======================
# STATE MACHINE
# ======================
state = "IDLE"
blink_count = 0
start_window_time = 0
last_blink_time = 0

def calculate_ear(landmarks, eye_indices):
    v1 = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) - 
                        np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
    v2 = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) - 
                        np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
    h = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) - 
                       np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))
    return (v1 + v2) / (2.0 * h)

# ======================
# INITIALIZE
# ======================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    running_mode=vision.RunningMode.VIDEO)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect_for_video(rgb_frame, int(time.time() * 1000))
    now = time.time()
    
    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        left_ear = calculate_ear(landmarks, LEFT_EYE)
        right_ear = calculate_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2

        # ----------------------
        # LOCKING LOGIC (3 Blinks)
        # ----------------------
        if state != "LOCKED":
            if avg_ear < EAR_THRESHOLD and (now - last_blink_time > BLINK_COOLDOWN):
                last_blink_time = now
                if state == "IDLE":
                    state = "COUNTING"
                    blink_count = 1
                    start_window_time = now
                elif state == "COUNTING":
                    blink_count += 1

            if state == "COUNTING":
                if blink_count >= 3:
                    state = "LOCKED"
                elif (now - start_window_time) > WINDOW_TIME:
                    state = "IDLE"
                    blink_count = 0

        # ----------------------
        # UNLOCKING LOGIC (Left Eye Wink)
        # ----------------------
        elif state == "LOCKED":
            # A wink is one eye closed while the other is open
            if left_ear < WINK_EAR_THRESHOLD and right_ear > EAR_THRESHOLD:
                state = "IDLE"
                blink_count = 0

        # UI
        status_color = (0, 0, 255) if state == "LOCKED" else (0, 255, 0)
        cv2.putText(frame, f"L-EAR: {left_ear:.2f} R-EAR: {right_ear:.2f}", (30, 40), 1, 1.5, status_color, 2)
        cv2.putText(frame, f"State: {state} | Blinks: {blink_count}", (30, 80), 1, 1.5, status_color, 2)

        if state == "LOCKED":
            # Visual Lock Overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[2]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.putText(frame, "LOCKED", (frame.shape[1]//2 - 100, frame.shape[0]//2), 1, 3, (0,0,255), 4)
            cv2.putText(frame, "Wink Left Eye to Unlock", (frame.shape[1]//2 - 120, frame.shape[0]//2 + 50), 1, 1, (255,255,255), 1)

    cv2.imshow('BlinkLock v2.0', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
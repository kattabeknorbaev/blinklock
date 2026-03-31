import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import numpy as np
import urllib.request
import os
import time

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
CAMERA_INDEX = 0

EAR_THRESHOLD    = 0.22   # below this = eye closed
BLINK_MIN_FRAMES = 2      # eye must be closed this many frames to count as blink
BLINK_MAX_FRAMES = 15     # eye closed longer than this = wink (not blink)
BLINK_WINDOW     = 2.0    # seconds to collect 3 blinks
BLINKS_TO_LOCK   = 3      # how many blinks to trigger lock
PIN              = "1234" # fallback unlock PIN
# ───────────────────────────────────────────────────────────────────────────────

# EAR landmark indices for left and right eye
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading face landmarker model (~30MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded.")

def ear(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    # vertical distances
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # horizontal distance
    h1 = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * h1)

def main():
    download_model()

    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    # ── State machine ──────────────────────────────────────────────────────
    # States: IDLE → COUNTING → LOCKED
    #         LOCKED → UNLOCK_CHECK → IDLE
    state = "IDLE"
    blink_count = 0
    blink_window_start = 0

    eye_closed_frames = 0   # how many consecutive frames eye has been closed
    blink_registered = False

    pin_input = ""
    flash_msg = ""
    flash_timer = 0

    print("[INFO] Running BlinkLock.")
    print(f"  Blink {BLINKS_TO_LOCK}x fast → LOCK")
    print(f"  Wink (hold {BLINK_MAX_FRAMES}+ frames) → UNLOCK")
    print(f"  PIN fallback: type '{PIN}' + Enter when locked")
    print("  Q → quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        avg_ear = None
        eye_closed = False
        wink_detected = False

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            left_ear  = ear(lm, LEFT_EYE,  w, h)
            right_ear = ear(lm, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            eye_closed = avg_ear < EAR_THRESHOLD

            if eye_closed:
                eye_closed_frames += 1
            else:
                # eye just opened — classify what happened
                if BLINK_MIN_FRAMES <= eye_closed_frames <= BLINK_MAX_FRAMES:
                    # it was a blink
                    if not blink_registered:
                        blink_registered = True

                        if state == "IDLE":
                            state = "COUNTING"
                            blink_count = 1
                            blink_window_start = time.time()
                        elif state == "COUNTING":
                            blink_count += 1
                            if blink_count >= BLINKS_TO_LOCK:
                                state = "LOCKED"
                                flash_msg = "LOCKED"
                                flash_timer = 60
                                print("[LOCK] Screen locked!")

                elif eye_closed_frames > BLINK_MAX_FRAMES:
                    # it was a wink — unlock
                    if state == "LOCKED":
                        state = "IDLE"
                        blink_count = 0
                        flash_msg = "UNLOCKED"
                        flash_timer = 60
                        print("[UNLOCK] Wink detected — unlocked!")

                eye_closed_frames = 0
                blink_registered = False

        # check if counting window expired
        if state == "COUNTING":
            if time.time() - blink_window_start > BLINK_WINDOW:
                state = "IDLE"
                blink_count = 0

        # ── Draw ──────────────────────────────────────────────────────────
        # locked overlay
        if state == "LOCKED":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, "LOCKED", (w//2 - 80, h//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            cv2.putText(frame, "Wink to unlock  |  type PIN + Enter",
                        (w//2 - 200, h//2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            if pin_input:
                cv2.putText(frame, f"PIN: {'*' * len(pin_input)}", (w//2 - 50, h//2 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)

        # HUD bar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, 0), (w, 60), (15, 15, 15), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "BlinkLock", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 50), 2)

        state_colors = {"IDLE": (150,150,150), "COUNTING": (0,200,255), "LOCKED": (0,0,255)}
        cv2.putText(frame, f"State: {state}  Blinks: {blink_count}/{BLINKS_TO_LOCK}",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_colors.get(state, (150,150,150)), 1)

        if avg_ear is not None:
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (w - 140, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # flash message
        if flash_timer > 0:
            cv2.putText(frame, flash_msg, (w//2 - 100, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 100) if flash_msg == "UNLOCKED" else (0, 100, 255), 3)
            flash_timer -= 1

        cv2.imshow("BlinkLock", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif state == "LOCKED":
            if key == 13:  # Enter
                if pin_input == PIN:
                    state = "IDLE"
                    blink_count = 0
                    flash_msg = "UNLOCKED"
                    flash_timer = 60
                    print("[UNLOCK] PIN correct — unlocked!")
                else:
                    flash_msg = "WRONG PIN"
                    flash_timer = 40
                pin_input = ""
            elif key == 8:  # Backspace
                pin_input = pin_input[:-1]
            elif 48 <= key <= 57:  # 0-9
                pin_input += chr(key)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[INFO] Exited cleanly.")

if __name__ == "__main__":
    main()
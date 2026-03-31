# 👁️ BlinkLock (Day 4: Moderate)

A computer vision security layer that locks your screen based on eye movement patterns. This project demonstrates **software debouncing** and **state machine** logic used in firmware development.

## 🚀 Features
- **EAR Tracking**: Calculates Eye Aspect Ratio (EAR) using MediaPipe FaceMesh.
- **Triple-Blink Lock**: Three rapid blinks within 2 seconds triggers the `LOCKED` state.
- **Wink-to-Unlock**: A deliberate left-eye wink (while the right eye stays open) resets the state to `IDLE`.
- **Software Debounce**: Implements a cooldown timer to prevent "switch bounce" where one blink is registered multiple times.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** OpenCV, MediaPipe, NumPy
- **Model:** MediaPipe FaceLandmarker (`face_landmarker.task`)

## 🎮 How to Use
1. Install dependencies: `pip install mediapipe opencv-python numpy`
2. Run the script: `python main.py`
3. **To Lock**: Blink 3 times rapidly.
4. **To Unlock**: Wink with your left eye (or press 'u' on your keyboard).
5. **To Quit**: Press 'q'.

## 🧠 State Machine Logic
The system transitions through four logical phases:
1. **IDLE**: Scanning for the first blink.
2. **COUNTING**: First blink detected; waiting for subsequent blinks within the 2s window.
3. **LOCKED**: 3 blinks reached. Visual overlay active.
4. **RECOVERY**: Checking for a wink or PIN fallback to return to IDLE.

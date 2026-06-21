import cv2
import mediapipe as mp

# Path to a MediaPipe Tasks hand_landmarker .task file. Leave empty to require
# using the legacy `mp.solutions.hands` (older mediapipe releases) instead.
MODEL_PATH = "hand_landmarker.task"  # Set to your .task file path or leave empty for legacy API

# Try to use legacy solutions API if available
_USE_SOLUTIONS = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
hand_landmarker = None
vision_core = None

if _USE_SOLUTIONS:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils
else:
    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import core as vision_core

        if MODEL_PATH:
            running_mode = getattr(vision, "RunningMode", None) or getattr(
                vision, "VisionTaskRunningMode", None
            )
            if running_mode is None:
                raise ImportError(
                    "MediaPipe Tasks does not expose RunningMode or VisionTaskRunningMode"
                )
            hand_options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=running_mode.IMAGE,
                num_hands=1,
            )
            hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    except Exception as e:
        print("Gesture control init error:", e)
        hand_landmarker = None
        vision_core = None


def _landmarks_to_pixel_list(landmarks, frame_shape):
    h, w = frame_shape[:2]
    lm_list = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        lm_list.append((x, y))
    return lm_list


def detect_gesture(frame):
    """Return a string like '3_fingers' or None if no hand detected.

    Uses `mp.solutions.hands` when available; otherwise uses MediaPipe Tasks
    `HandLandmarker` (requires `MODEL_PATH` to be set to a .task model).
    """
    # Legacy solutions API
    if _USE_SOLUTIONS:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        hand_label = None
        if result.multi_handedness:
            hand_label = result.multi_handedness[0].classification[0].label

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    lm_list.append((int(lm.x * w), int(lm.y * h)))

                fingers_up = 0
                tip_ids = [4, 8, 12, 16, 20]

                # Thumb
                if hand_label == "Left":
                    if lm_list[tip_ids[0]][0] < lm_list[tip_ids[0] - 1][0]:
                        fingers_up += 1
                else:
                    if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
                        fingers_up += 1

                # Other fingers
                for i in range(1, 5):
                    if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1]:
                        fingers_up += 1

                return f"{fingers_up}_fingers"

        return None

    # Tasks API
    if hand_landmarker is None or vision_core is None:
        # Don't crash the whole app if the Tasks model isn't provided.
        # Return None so `main.py` keeps running; detection will be disabled.
        try:
            # Print once for visibility when running interactively.
            if not getattr(detect_gesture, "_warned", False):
                print(
                    "Warning: HandLandmarker not initialized. Set MODEL_PATH to a hand_landmarker.task file to enable detection."
                )
                detect_gesture._warned = True
        except Exception:
            pass
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = vision_core.image.Image(vision_core.image.ImageFormat.SRGB, rgb_frame)
    result = hand_landmarker.detect(image)

    if not result or not getattr(result, "hand_landmarks", None):
        return None

    first = result.hand_landmarks[0]
    landmarks = getattr(first, "landmark", None) or getattr(first, "landmarks", None) or first
    lm_list = _landmarks_to_pixel_list(landmarks, frame.shape)

    fingers_up = 0
    tip_ids = [4, 8, 12, 16, 20]

    # Thumb
    if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
        fingers_up += 1

    # Other fingers
    for i in range(1, 5):
        if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1]:
            fingers_up += 1

    return f"{fingers_up}_fingers"

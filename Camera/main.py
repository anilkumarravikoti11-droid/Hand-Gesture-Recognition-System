import cv2
from filters import apply_filter
from gesture_control import detect_gesture

cap = cv2.VideoCapture(0)
current_filter = "original"
gesture_to_filter = {
    "1_fingers": "original",
    "2_fingers": "gray",
    "3_fingers": "blur",
    "4_fingers": "sepia",
    "5_fingers": "edges",
}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gesture = detect_gesture(frame)
    current_filter = gesture_to_filter.get(gesture, current_filter)

    filtered = apply_filter(frame, current_filter)

    if len(filtered.shape) == 2:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        filtered,
        f"Current Filter: {current_filter}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        filtered,
        f"Gesture: {gesture or 'no hand detected'}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Gesture Controlled Filters", filtered)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
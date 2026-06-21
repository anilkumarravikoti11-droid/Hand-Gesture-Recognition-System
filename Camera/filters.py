import cv2
import numpy as np

def apply_filter(frame, filter_name):

    if filter_name == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    elif filter_name == "sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        return cv2.transform(frame, sepia_filter)

    elif filter_name == "blur":
        return cv2.GaussianBlur(frame, (15, 15), 0)

    elif filter_name == "edges":
        return cv2.Canny(frame, 100, 200)

    return frame
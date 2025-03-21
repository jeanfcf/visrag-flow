import cv2
import numpy as np

# Exemplo simples de detecção de movimento usando background adaptativo
background_frame = None

def is_motion_detected(frame, threshold=30):
    global background_frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if background_frame is None:
        background_frame = gray
        return False

    diff = cv2.absdiff(background_frame, gray)
    thresh_img = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

    diff_pixels = np.sum(thresh_img) / 255
    if diff_pixels > threshold:
        alpha = 0.1
        background_frame = cv2.addWeighted(gray, alpha, background_frame, 1 - alpha, 0)
        return True
    else:
        alpha = 0.02
        background_frame = cv2.addWeighted(gray, alpha, background_frame, 1 - alpha, 0)
        return False

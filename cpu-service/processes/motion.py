def load():
    import cv2, numpy as np
    background_frame = None

    def _process_frame(frame, threshold=30, **_):
        nonlocal background_frame
        results = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if background_frame is None:
            background_frame = gray
            return False
        diff = cv2.absdiff(background_frame, gray)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion = bool(np.sum(mask) / 255 > threshold) 
        alpha  = 0.1 if motion else 0.02
        background_frame = cv2.addWeighted(gray, alpha, background_frame, 1-alpha, 0)
        results["motion_detected"] = motion
        return results

    return _process_frame





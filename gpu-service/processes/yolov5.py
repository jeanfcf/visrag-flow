def load():
    import warnings
    warnings.simplefilter("ignore", FutureWarning)

    import torch
    import cv2

    _model = None
    if _model is None:
        _model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device)
        _model.eval()

    def run_inference(model, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, size=640)
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        people_detections = [d for d in detections if d.get("name", "").lower() == "person"]
        return people_detections

    def _process_frame(frame, **kwargs):
        detections = run_inference(_model, frame)
        return detections

    return _process_frame





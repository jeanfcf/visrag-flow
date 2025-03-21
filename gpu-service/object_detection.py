import torch
import cv2

def load_model():
    """
    Carrega modelo YOLOv5 pré-treinado (por ex. 'yolov5s').
    Necessita PyTorch e 'git', 'ultralytics' etc. instalados.
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    if torch.cuda.is_available():
        model.to('cuda')
    model.eval()
    return model

def run_inference(model, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, size=640)
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return detections

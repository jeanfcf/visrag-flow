import cv2
from PIL import Image
import torch
import clip

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(["a photo of a person", "a photo of an object"]).to(device)
    with torch.no_grad():
        logits, _ = model(image_input, text_inputs)
    probs = logits.softmax(dim=-1).cpu().numpy()
    return f"CLIP probabilities: {probs.tolist()}"

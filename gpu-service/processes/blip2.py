import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def process_frame(frame):
    # Converte de BGR para RGB e cria imagem PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    # Utiliza a variante mais leve: "Salesforce/blip2-flan-t5-base" em vez de t5-xxl
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-base")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-base", torch_dtype=torch.float16)
    # Move os tensores para GPU; se sua GPU não suportar FP16 ou for muito limitada, considere rodar em CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(pil_image, return_tensors="pt").to(device)
    model.to(device)
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

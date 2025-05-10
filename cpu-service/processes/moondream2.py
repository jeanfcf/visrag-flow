import cv2
import numpy as np
from PIL import Image
import moondream as md

# Se você quiser ver a imagem numa janela, pode importar matplotlib:
# import matplotlib.pyplot as plt

_model = None

def load_model():
    global _model
    if _model is None:
        # Ajuste para o caminho correto do seu modelo .mf
        _model = md.MoonDreamModel(model="moondream-0_5b-int8.mf")
    return _model

def process_frame(frame):
    """
    Converte o frame recebido (BGR) para imagem PIL, utiliza o modelo Moondream2 para gerar uma legenda,
    e retorna a legenda como string.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    model = load_model()
    caption_result = model.caption(pil_image, length="normal")
    caption = caption_result.get("caption", "No caption generated")
    return caption
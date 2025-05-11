def load():
    import cv2
    import numpy as np
    from PIL import Image
    import moondream as md
    
    _model = None
    if _model is None:
        # Ajuste para o caminho correto do seu modelo .mf
        _model = md.MoonDreamModel(model="moondream-0_5b-int8.mf")

    def _process_frame(frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        caption_result = _model.caption(pil_image, length="normal")
        caption = caption_result.get("caption", "No caption generated")
        return caption

    return _process_frame

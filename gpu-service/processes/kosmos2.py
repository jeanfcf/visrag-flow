def _load():
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    def _process_frame(frame):
        # Observação: Kosmos-2 é um modelo pesado. Para testes locais com sua GPU (1650 Super 4GB),
        # a performance pode ser comprometida. Considere desabilitar este módulo para testes rápidos.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-base")
        model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-base").to("cuda")
        inputs = processor(pil_image, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    return _process_frame




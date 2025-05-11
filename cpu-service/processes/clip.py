def load():
    import clip
    import torch
    from PIL import Image
    import cv2

    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    def _process_frame(frame):
        """
        Converte o frame recebido (BGR) para imagem PIL, utiliza o modelo CLIP para gerar uma legenda,
        e retorna a legenda como string.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        image_input = preprocess(pil_image).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(["a photo of a person", "a photo of an object"]).to(device)
        with torch.no_grad():
            logits, _ = model(image_input, text_inputs)
        probs = logits.softmax(dim=-1).cpu().numpy()
        return f"CLIP probabilities: {probs.tolist()}"

    return _process_frame

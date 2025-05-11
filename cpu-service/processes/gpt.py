def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    def process_text(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    def _process_frame(frame_caption):
        prompt = f"Generate a detailed report based on the following image description: {frame_caption}"
        return process_text(prompt)

    return _process_frame



from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class MLDetector:
    def __init__(self, model_path="./local_fake_news_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def detect(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        pred = torch.argmax(outputs.logits).item()
        is_fake = pred == 0
        reasoning = f"ML model classified as {'Fake News' if is_fake else 'Real News'}"
        return is_fake, reasoning

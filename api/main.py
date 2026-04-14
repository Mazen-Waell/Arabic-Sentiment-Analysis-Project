from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "arabert-sentiment")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Arabic Sentiment Analysis API"}

@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_id = probs.argmax().item()

    return {
        "text": input.text,
        "sentiment": id2label[pred_id],
        "confidence": round(probs[pred_id].item(), 3),
        "scores": {
            id2label[i]: round(probs[i].item(), 3)
            for i in range(3)
        }
    }
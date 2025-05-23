print("🚀 Starting CLIP API...")

from fastapi import FastAPI, Request
from transformers import CLIPTokenizer, CLIPModel
import torch

print("📦 Loading tokenizer and model...")

try:
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading CLIP model:", e)
    raise

app = FastAPI()

@app.post("/embed")
async def embed_text(request: Request):
    body = await request.json()
    text = body.get("text", "")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    return {"embedding": embeddings[0].tolist()}

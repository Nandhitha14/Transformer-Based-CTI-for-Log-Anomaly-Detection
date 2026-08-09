# Precompute embeddings using DistilBERT (will download model from HuggingFace automatically)
import os
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm

CSV = "data/datasets/train.csv"  # place your CSV here with columns: timestamp,risk_level,message,label
OUT_DIR = "embeddings_cache"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CSV):
    raise FileNotFoundError(f"Dataset CSV not found at {CSV}. Put your CSV there.")

print('Loading DistilBERT tokenizer and model (will download if needed)...')
tok = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

df = pd.read_csv(CSV)
msgs = df['message'].astype(str).tolist()
labels = df['label'].astype(int).values

batch_size = 16
embs = []
with torch.no_grad():
    for i in tqdm(range(0, len(msgs), batch_size)):
        batch = msgs[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        vecs = out.last_hidden_state.mean(dim=1).cpu().numpy()
        embs.append(vecs)

embs = np.vstack(embs).astype('float32')
print('Embeddings shape:', embs.shape)
np.save(os.path.join(OUT_DIR, 'X.npy'), embs)
np.save(os.path.join(OUT_DIR, 'y.npy'), labels)
print('Saved embeddings to', OUT_DIR)

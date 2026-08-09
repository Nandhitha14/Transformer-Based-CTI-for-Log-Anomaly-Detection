# Lightweight on-the-fly preprocessing for the app: compute DistilBERT embeddings for messages
import numpy as np, torch
from transformers import DistilBertTokenizerFast, DistilBertModel

_tok = None
_model = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _ensure_model():
    global _tok, _model
    if _tok is None:
        _tok = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    if _model is None:
        _model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(_device)
        _model.eval()

def preprocess_for_app(df):
    # df must have column 'message'
    if 'message' not in df.columns:
        raise ValueError("CSV must contain 'message' column")
    _ensure_model()
    msgs = df['message'].astype(str).tolist()
    batch = 8
    embs = []
    with torch.no_grad():
        for i in range(0, len(msgs), batch):
            batch_msgs = msgs[i:i+batch]
            enc = _tok(batch_msgs, padding=True, truncation=True, return_tensors='pt')
            input_ids = enc['input_ids'].to(_device)
            am = enc['attention_mask'].to(_device)
            out = _model(input_ids=input_ids, attention_mask=am)
            vecs = out.last_hidden_state.mean(dim=1).cpu().numpy()
            embs.append(vecs)
    return np.vstack(embs).astype('float32')

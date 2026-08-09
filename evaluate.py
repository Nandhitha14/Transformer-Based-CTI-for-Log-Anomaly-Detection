import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

from model import TransformerTCN
from preprocess_stub import preprocess_for_app  # DistilBERT embeddings

def load_model(model_path, device):
    """Load checkpoint and infer num_classes automatically."""
    state = torch.load(model_path, map_location=device)
    fc_keys = [k for k in state.keys() if k.endswith(".weight") and "fc" in k]
    fc_key = sorted(fc_keys)[-1]
    num_classes = state[fc_key].shape[0]
    model = TransformerTCN(input_dim=768, hidden_dim=256, num_classes=num_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, num_classes

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, num_classes = load_model(args.model, device)
    print(f"Loaded model {args.model} with {num_classes} output classes")

    # Load test data
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")
    df = pd.read_csv(args.test_csv)
    if "message" not in df.columns or "label" not in df.columns:
        raise ValueError("Test CSV must contain at least 'message' and 'label' columns")

    # Preprocess embeddings
    t0 = time.time()
    X = preprocess_for_app(df)  # numpy (N,768)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
    y_true = df["label"].astype(int).to_numpy()
    preprocess_time = time.time() - t0

    # Model inference
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    # Metrics
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, preds, average="macro")

    cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))

    # FPR/FNR per class
    fpr, fnr = {}, {}
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        fpr[c] = float(fp / (fp + tn + 1e-12))
        fnr[c] = float(fn / (fn + tp + 1e-12))

    # ROC-AUC and PR-AUC
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        roc_auc = {}
        pr_auc = {}
        for c in range(num_classes):
            roc_auc[c] = float(roc_auc_score(y_true_bin[:, c], probs[:, c]))
            pr_auc[c] = float(average_precision_score(y_true_bin[:, c], probs[:, c]))
    except Exception:
        roc_auc = None
        pr_auc = None

    avg_latency = (preprocess_time / len(df)) if len(df) else 0

    results = {
        "num_samples": len(df),
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "fpr_per_class": fpr,
        "fnr_per_class": fnr,
        "roc_auc_per_class": roc_auc,
        "pr_auc_per_class": pr_auc,
        "avg_latency_s": avg_latency
    }

    # Save metrics
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Metrics saved to results/metrics.json")

    # Print summary
    print("\n=== Evaluation Summary ===")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default=r"C:\\Users\\mohanraj\\Desktop\\brocode\\log_anomaly_app_distil\\models\\model_fast_distil.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--test_csv", type=str, default="data/test.csv",
                        help="Path to test CSV with 'message' and 'label'")
    args = parser.parse_args()
    main(args)

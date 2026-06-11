"""classify.py — обучение BERT-классификаторов, ансамбль, калибровка."""

import gc
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score


class _TextDS(Dataset):
    def __init__(self, enc, labels):
        self.enc = enc; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids":      self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": torch.tensor(self.labels[i], dtype=torch.long)}


def _train(cfg_m, train_df, test_df, num_labels, device, log):
    from utils.utils import set_seed
    set_seed(42)
    name = cfg_m["name"]
    log.info(f"Обучаю: {name}")
    tok = AutoTokenizer.from_pretrained(cfg_m["model_name"])
    def enc(texts):
        return tok(texts, max_length=cfg_m["max_length"],
                   truncation=True, padding="max_length", return_tensors="pt")
    g = torch.Generator(); g.manual_seed(42)
    tr = DataLoader(_TextDS(enc(train_df["text"].tolist()), train_df["label_id"].values),
                    batch_size=cfg_m["batch_size"], shuffle=True, generator=g)
    te = DataLoader(_TextDS(enc(test_df["text"].tolist()),  test_df["label_id"].values),
                    batch_size=cfg_m["batch_size"], shuffle=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg_m["model_name"], num_labels=num_labels).to(device)
    opt   = AdamW(model.parameters(), lr=cfg_m["learning_rate"])
    sched = get_linear_schedule_with_warmup(opt, 100, len(tr) * cfg_m["epochs"])
    for epoch in range(cfg_m["epochs"]):
        model.train(); tot = 0
        for b in tr:
            opt.zero_grad()
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += out.loss.item()
        log.info(f"  {name} эп {epoch+1}/{cfg_m['epochs']}: loss={tot/len(tr):.4f}")
    model.eval(); logits_list = []
    with torch.no_grad():
        for b in te:
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device))
            logits_list.append(out.logits.cpu())
    logits = torch.cat(logits_list).numpy()
    preds  = logits.argmax(1)
    y      = test_df["label_id"].values
    m = {"balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 4),
         "macro_f1":          round(float(f1_score(y, preds, average="macro", zero_division=0)), 4),
         "accuracy":          round(float(accuracy_score(y, preds)), 4)}
    log.info(f"  {name}: bal_acc={m['balanced_accuracy']} | macro_f1={m['macro_f1']}")
    del model, opt, sched; gc.collect(); torch.cuda.empty_cache()
    return logits, m


def _ece(probs, labels, n_bins=10):
    conf = probs.max(1); pred = probs.argmax(1)
    acc  = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i+1])
        if m.sum() > 0:
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def run(cfg, log):
    from utils.utils import timer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Устройство: {device}")

    with timer(log, "Stage 5: классификация + ансамбль + калибровка"):
        train_df = pd.read_csv(cfg.TRAIN_FINAL_CSV)
        test_df  = pd.read_csv(cfg.TEST_CSV)
        log.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

        le = LabelEncoder(); le.fit(test_df["label"])
        num_labels = len(le.classes_)
        train_df["label_id"] = le.transform(train_df["label"])
        test_df["label_id"]  = le.transform(test_df["label"])
        y_test = test_df["label_id"].values

        cfg.ENSEMBLE_PROBS_DIR.mkdir(parents=True, exist_ok=True)
        all_logits, all_metrics = {}, {}

        for cfg_m in cfg.CLASSIFIER_MODELS:
            npy = cfg.ENSEMBLE_PROBS_DIR / f"{cfg_m['name']}_logits.npy"
            if cfg.SKIP_IF_EXISTS and npy.exists():
                log.info(f"[SKIP] {cfg_m['name']} логиты уже есть")
                logits = np.load(npy)
                preds  = logits.argmax(1)
                all_metrics[cfg_m['name']] = {
                    "balanced_accuracy": round(float(balanced_accuracy_score(y_test, preds)), 4),
                    "macro_f1": round(float(f1_score(y_test, preds, average="macro", zero_division=0)), 4),
                    "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                }
            else:
                logits, metrics = _train(cfg_m, train_df, test_df, num_labels, device, log)
                np.save(npy, logits)
                all_metrics[cfg_m['name']] = metrics
            all_logits[cfg_m['name']] = np.load(npy)

        log.info("Ансамбль...")
        probs_list = [F.softmax(torch.tensor(v), dim=1).numpy() for v in all_logits.values()]
        ens_probs  = np.mean(probs_list, axis=0)
        ens_preds  = ens_probs.argmax(1)
        ens_m = {"balanced_accuracy": round(float(balanced_accuracy_score(y_test, ens_preds)), 4),
                 "macro_f1":          round(float(f1_score(y_test, ens_preds, average="macro", zero_division=0)), 4),
                 "accuracy":          round(float(accuracy_score(y_test, ens_preds)), 4)}
        log.info(f"Ансамбль: bal_acc={ens_m['balanced_accuracy']} | macro_f1={ens_m['macro_f1']}")

        log.info("Калибровка...")
        lg = torch.tensor(np.mean(list(all_logits.values()), axis=0))
        yt = torch.tensor(y_test)
        best_T, best_nll = 1.0, float("inf")
        for T in np.arange(0.5, 5.01, 0.05):
            nll = F.cross_entropy(lg / T, yt).item()
            if nll < best_nll: best_nll, best_T = nll, T
        ece_before = _ece(ens_probs, y_test)
        ece_after  = _ece(F.softmax(lg / best_T, dim=1).numpy(), y_test)
        log.info(f"T={best_T:.2f} | ECE: {ece_before:.4f} → {ece_after:.4f}")

        results = {"individual": all_metrics, "ensemble": ens_m,
                   "calibration": {"temperature": round(float(best_T), 2),
                                   "ece_before": round(ece_before, 4),
                                   "ece_after":  round(ece_after,  4)}}
        with open(cfg.METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info(f"Метрики: {cfg.METRICS_JSON}")
        for name, m in all_metrics.items():
            log.info(f"  {name}: bal_acc={m['balanced_accuracy']} | macro_f1={m['macro_f1']}")
        log.info(f"  АНСАМБЛЬ: bal_acc={ens_m['balanced_accuracy']} | macro_f1={ens_m['macro_f1']}")

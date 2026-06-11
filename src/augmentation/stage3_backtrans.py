"""stage3_backtrans.py — обратный перевод RU→EN→RU через NLLB-200."""

import gc
import random
import numpy as np
import pandas as pd
import torch

from aug_utils import mask_placeholders, unmask_placeholders, PLACEHOLDER_RE, UNREST_RE, BROKEN_RE, CJK_RE

LANG_RU    = "rus_Cyrl"
LANG_EN    = "eng_Latn"
BATCH_SIZE = 32
MAX_LENGTH = 512


def _is_valid_bt(text, orig, min_len):
    if not text or len(text.strip()) < min_len:
        return False
    if CJK_RE.search(text):
        return False
    if UNREST_RE.search(text) or BROKEN_RE.search(text):
        return False
    orig_ph = set(PLACEHOLDER_RE.findall(orig))
    if orig_ph and not set(PLACEHOLDER_RE.findall(text)):
        return False
    return True


def run(cfg, log):
    from utils.utils import set_seed, skip_if_exists, timer
    if skip_if_exists(cfg.STAGE3_CSV, "stage3_backtrans", cfg.SKIP_IF_EXISTS, log):
        return

    with timer(log, "Stage 3: back-translation"):
        set_seed(42)
        df = pd.read_csv(cfg.STAGE2_CSV)
        log.info(f"Загружено: {len(df)} строк")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Устройство: {device}")

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        log.info(f"Загружаю NLLB: {cfg.BT_MODEL_NAME}")
        nllb_tok = AutoTokenizer.from_pretrained(cfg.BT_MODEL_NAME)
        nllb_mod = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.BT_MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        nllb_mod.eval()

        def translate_batch(texts, src_lang, tgt_lang):
            try:
                nllb_tok.src_lang = src_lang
                inputs = nllb_tok(texts, return_tensors="pt", padding=True,
                                  truncation=True, max_length=MAX_LENGTH).to(device)
                tgt_id = nllb_tok.convert_tokens_to_ids(tgt_lang)
                with torch.no_grad():
                    outputs = nllb_mod.generate(
                        **inputs, forced_bos_token_id=tgt_id, max_length=MAX_LENGTH,
                        do_sample=True, temperature=1.2, top_p=0.9)
                return nllb_tok.batch_decode(outputs, skip_special_tokens=True)
            except Exception as e:
                log.warning(f"NLLB batch error: {e}")
                return [""] * len(texts)

        def back_translate(texts):
            from tqdm import tqdm
            masked, all_phs = [], []
            for t in texts:
                m, phs = mask_placeholders(t)
                masked.append(m); all_phs.append(phs)
            en = []
            for i in tqdm(range(0, len(masked), BATCH_SIZE), desc="RU→EN"):
                en.extend(translate_batch(masked[i:i+BATCH_SIZE], LANG_RU, LANG_EN))
            ru = []
            for i in tqdm(range(0, len(en), BATCH_SIZE), desc="EN→RU"):
                ru.extend(translate_batch(en[i:i+BATCH_SIZE], LANG_EN, LANG_RU))
            return [unmask_placeholders(t.strip(), phs) for t, phs in zip(ru, all_phs)]

        counts = df['label'].value_counts()
        to_aug = counts[counts < cfg.STAGE3_TARGET].sort_values()
        log.info(f"Классов для BT: {len(to_aug)}")

        new_rows = []
        for i, (cls, cur) in enumerate(to_aug.items(), 1):
            need = cfg.STAGE3_TARGET - cur
            existing = df[df['label'] == cls]['text'].tolist()
            log.info(f"[{i}/{len(to_aug)}] {cls[:40]}: есть {cur}, нужно {need}")
            sources = [existing[j % len(existing)] for j in range(need * 3)]
            bt_results = back_translate(sources)
            accepted = 0
            for src, bt in zip(sources, bt_results):
                if accepted >= need:
                    break
                if _is_valid_bt(bt, src, cfg.MIN_LENGTH):
                    new_rows.append({"label": cls, "text": bt, "source": "stage3_backtranslation"})
                    existing.append(bt); accepted += 1
            log.info(f"  принято {accepted}/{need}")
            pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True).to_csv(cfg.STAGE3_CSV, index=False)

        result = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        result.to_csv(cfg.STAGE3_CSV, index=False)
        log.info(f"Добавлено {len(new_rows)} BT. Итого: {len(result)}")
        nllb_mod.cpu(); del nllb_mod, nllb_tok; gc.collect(); torch.cuda.empty_cache()

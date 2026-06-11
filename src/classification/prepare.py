"""prepare.py — загрузка JSON, очистка текстов, train/test split."""

import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def remove_repeated_words(text):
    return re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

def remove_repeated_chunks(text, min_w=5, max_w=30):
    words = text.split()
    out, i = [], 0
    while i < len(words):
        matched = False
        for size in range(max_w, min_w - 1, -1):
            if i + 2 * size > len(words):
                continue
            if words[i:i+size] == words[i+size:i+2*size]:
                out.extend(words[i:i+size])
                j = i + size
                while j + size <= len(words) and words[j:j+size] == words[i:i+size]:
                    j += size
                i, matched = j, True; break
        if not matched:
            out.append(words[i]); i += 1
    return ' '.join(out)

def remove_duplicate_lines(text):
    seen, out = set(), []
    for line in text.split('\n'):
        norm = line.strip().lower()
        if not norm: out.append(line)
        elif norm not in seen: out.append(line); seen.add(norm)
    return '\n'.join(out)

def remove_duplicate_sentences(text):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    seen, out = set(), []
    for s in sents:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key); out.append(s.strip())
    return ' '.join(out)

def trim_attachments(text, max_len=4000):
    if len(text) <= max_len:
        return text
    triggers = [
        r'[Тт]аблиц[а-я]*', r'Публичная оферта',
        r'заключили настоящий [Дд]оговор', r'1\.\s*Предмет договора',
        r'Акт сверки', r'Акт\s+№?\s*\d+', r'Приложение\s+\d+\s+к',
        r'ПОЛОЖЕНИЕ\s+О\s+', r'1\.\s*Общие положения',
    ]
    cut = len(text)
    for trig in triggers:
        m = re.search(trig, text)
        if m: cut = min(cut, m.start())
    return text[:cut].strip() if cut < len(text) else text

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace('\u00ab','"').replace('\u00bb','"')
    text = text.replace('\u2013','-').replace('\u2014','-')
    text = remove_repeated_words(text)
    text = remove_repeated_chunks(text)
    text = remove_duplicate_lines(text)
    text = remove_duplicate_sentences(text)
    text = trim_attachments(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

def is_anomalous(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return len(words) > 200 and len(set(words)) < 50


def run(cfg, log):
    from utils.utils import skip_if_exists, timer
    if skip_if_exists(cfg.TRAIN_CSV, "stage0_prepare", cfg.SKIP_IF_EXISTS, log):
        return

    with timer(log, "Stage 0: подготовка данных"):
        with open(cfg.INPUT_JSON_PATH, encoding='utf-8') as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)[['label','text']]
        log.info(f"Загружено: {len(df)} документов, {df['label'].nunique()} классов")

        n0 = len(df)
        df = df[~df['text'].apply(is_anomalous)].reset_index(drop=True)
        log.info(f"Удалено аномальных: {n0 - len(df)}")
        df['text'] = df['text'].apply(clean_text)
        df = df[df['text'].str.strip().astype(bool)].reset_index(drop=True)
        log.info(f"После очистки: {len(df)}")

        test_min = df.groupby('label', group_keys=False).sample(
            n=1, random_state=cfg.RANDOM_STATE)
        remaining = df.drop(index=test_min.index)
        target_test_n = int(round(len(df) * cfg.TEST_SIZE)) - len(test_min)
        vc = remaining['label'].value_counts()
        rem_ok  = remaining[remaining['label'].isin(vc[vc >= 2].index)]
        rem_bad = remaining[~remaining['label'].isin(vc[vc >= 2].index)]
        train_e, test_e = train_test_split(
            rem_ok, test_size=target_test_n,
            random_state=cfg.RANDOM_STATE, stratify=rem_ok['label'])
        train_df = pd.concat([train_e, rem_bad], ignore_index=True)
        test_df  = pd.concat([test_e,  test_min], ignore_index=True)
        train_df.to_csv(cfg.TRAIN_CSV, index=False)
        test_df.to_csv(cfg.TEST_CSV,   index=False)
        log.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

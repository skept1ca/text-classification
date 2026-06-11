"""aug_utils.py — общие валидаторы и утилиты для этапов аугментации."""

import re
import random
from langdetect import detect, LangDetectException

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")
PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z_]*(?:\s[A-Z_]*)?\]")
UNREST_RE = re.compile(r"<\s*\d+\s*>")
BROKEN_RE = re.compile(r'[«»"]\s*\d+\s*>|<\s*\d+\s*[«»"]')

LEAK_MARKERS = [
    "конечно,", "конечно!", "вот письмо", "вот пример", "вот несколько",
    "пример письма", "напиши одно", "переформулирован", "как языковая модель",
    "я создам", "для класса «", "предоставь пример", "вот ещё одно",
    "вот переформулирован", "вот другой вариант", "я переформулировал", "вот текст",
]


def is_truncated(text):
    t = text.rstrip()
    return not t or t[-1] not in '.!?)"»]'


def is_degenerate(text):
    words = text.lower().split()
    if len(words) < 3:
        return True
    if len(set(words)) / len(words) < 0.2:
        return True
    if re.search(r'(\b\w+(?:\s+\w+){2,})\s+\1\s+\1', text.lower()):
        return True
    return False


def is_prompt_leak(text):
    low = text.strip().lower()
    if any(m in low[:150] for m in LEAK_MARKERS):
        return True
    return bool(re.search(r'\*\*[^*\n]{3,}?\*\*|(?:^|\n)###?\s', text, re.MULTILINE))


def is_russian(text):
    try:
        return detect(text) == "ru"
    except LangDetectException:
        return False


def placeholders_preserved(orig, para, min_ratio=0.5):
    orig_ph = PLACEHOLDER_RE.findall(orig)
    if not orig_ph:
        return True
    return len(PLACEHOLDER_RE.findall(para)) >= len(orig_ph) * min_ratio


def mask_placeholders(text):
    """Заменяет [PERSON] и др. на <0>, <1>, ... для NLLB-перевода."""
    phs = PLACEHOLDER_RE.findall(text)
    masked = text
    for i, ph in enumerate(phs):
        masked = masked.replace(ph, f"<{i}>", 1)
    return masked, phs


def unmask_placeholders(text, phs):
    """Восстанавливает оригинальные плейсхолдеры из <0>, <1>, ..."""
    for i, ph in enumerate(phs):
        text = text.replace(f"<{i}>", ph, 1)
    return text


def make_examples(texts, k=5, max_chars=1500):
    sample = random.sample(texts, min(k, len(texts)))
    return "\n---\n".join(t[:max_chars] for t in sample)


def validate_generated(candidates, existing_texts, sbert, sim_low, sim_high, min_len):
    """Валидация для stage1 (генерация) и общий фильтр."""
    seen = {t.strip().lower() for t in existing_texts}
    passed = []
    for t in candidates:
        if not t or len(t.strip()) < min_len:
            continue
        norm = t.strip().lower()
        if norm in seen:
            continue
        if is_truncated(t) or is_degenerate(t) or is_prompt_leak(t):
            continue
        if CJK_RE.search(t) or not is_russian(t):
            continue
        seen.add(norm)
        passed.append(t)
    if not passed or not existing_texts:
        return passed
    from sklearn.metrics.pairwise import cosine_similarity
    emb_new = sbert.encode(passed, show_progress_bar=False)
    emb_old = sbert.encode(list(existing_texts)[:500], show_progress_bar=False)
    sims = cosine_similarity(emb_new, emb_old).max(axis=1)
    return [t for t, s in zip(passed, sims) if sim_low <= s < sim_high]


def validate_paraphrases(pairs, existing_texts, sbert, sim_low, sim_high, min_len):
    """Валидация для stage2 (парафраз) — дополнительно проверяет плейсхолдеры."""
    seen = {t.strip().lower() for t in existing_texts}
    passed = []
    for orig, para in pairs:
        if not para or len(para.strip()) < min_len:
            continue
        norm = para.strip().lower()
        if norm in seen or norm == orig.strip().lower():
            continue
        if is_truncated(para) or is_degenerate(para) or is_prompt_leak(para):
            continue
        if CJK_RE.search(para) or not is_russian(para):
            continue
        if not placeholders_preserved(orig, para):
            continue
        seen.add(norm)
        passed.append(para)
    if not passed or not existing_texts:
        return passed
    from sklearn.metrics.pairwise import cosine_similarity
    emb_new = sbert.encode(passed, show_progress_bar=False)
    emb_old = sbert.encode(list(existing_texts)[:500], show_progress_bar=False)
    sims = cosine_similarity(emb_new, emb_old).max(axis=1)
    return [t for t, s in zip(passed, sims) if sim_low <= s < sim_high]

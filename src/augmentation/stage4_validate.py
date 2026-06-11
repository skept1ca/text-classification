"""stage4_validate.py — финальная валидация аугментированного датасета."""

import pandas as pd
from aug_utils import UNREST_RE, BROKEN_RE, PLACEHOLDER_RE


def run(cfg, log):
    from utils.utils import skip_if_exists, timer
    if skip_if_exists(cfg.TRAIN_FINAL_CSV, "stage4_validate", cfg.SKIP_IF_EXISTS, log):
        return

    with timer(log, "Stage 4: финальная валидация"):
        df = pd.read_csv(cfg.STAGE3_CSV)
        n0 = len(df)
        log.info(f"Загружено: {n0} строк")

        df = df.drop_duplicates(subset=['label','text'], keep='first').reset_index(drop=True)
        n_dup = n0 - len(df)

        aug = df['source'] != 'original'
        ph_count = df['text'].apply(lambda t: len(PLACEHOLDER_RE.findall(t)))

        bad = (aug & df['text'].str.contains(UNREST_RE)) | \
              (aug & df['text'].str.contains(BROKEN_RE)) | \
              (aug & (ph_count == 0))

        log.info(f"Удаляется: дубли={n_dup}, брак={bad.sum()}")
        df = df[~bad].reset_index(drop=True)
        log.info(f"Итого: {n0} → {len(df)}")

        df.to_csv(cfg.TRAIN_FINAL_CSV, index=False)
        vc = df['label'].value_counts()
        log.info(f"Классов: {len(vc)} | min={vc.min()} | max={vc.max()} | median={vc.median()}")
        log.info("Источники:\n" + df['source'].value_counts().to_string())

"""
run_pipeline.py — главный скрипт запуска пайплайна классификации.

Использование:
    python scripts/run_pipeline.py --data /path/to/data.json

Опции:
    --data PATH        Путь к входному JSON-файлу (обязательно)
    --from-stage N     Начать с этапа N (0-5), по умолчанию 0
    --only-stage N     Запустить только этап N
    --no-skip          Не пропускать уже выполненные этапы
    --work-dir PATH    Рабочая директория (по умолчанию ./pipeline_output)

Этапы:
    0 — подготовка данных (чистка + train/test split)
    1 — LLM-генерация (Qwen2.5-14B через vLLM)
    2 — парафраз (Qwen2.5-14B через vLLM)
    3 — обратный перевод (NLLB-200)
    4 — финальная валидация датасета
    5 — классификаторы + ансамбль + калибровка
"""

import argparse
import sys
import time
from pathlib import Path

# Добавляем корень проекта в sys.path чтобы находились модули src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "augmentation"))
sys.path.insert(0, str(ROOT / "src" / "classification"))
sys.path.insert(0, str(ROOT / "src" / "utils"))

import src.augmentation.stage1_generate   as stage1
import src.augmentation.stage2_paraphrase as stage2
import src.augmentation.stage3_backtrans  as stage3
import src.augmentation.stage4_validate   as stage4
import src.classification.prepare         as stage0
import src.classification.classify        as stage5
from src.utils.config import Config
from src.utils.utils  import get_logger

STAGES = {
    0: ("Подготовка данных",        stage0),
    1: ("LLM-генерация",            stage1),
    2: ("Парафраз",                  stage2),
    3: ("Обратный перевод",         stage3),
    4: ("Финальная валидация",      stage4),
    5: ("Классификация + ансамбль", stage5),
}


def parse_args():
    p = argparse.ArgumentParser(description="Пайплайн классификации деловых писем")
    p.add_argument("--data", required=True, type=Path,
                   help="Путь к входному JSON-файлу")
    p.add_argument("--from-stage", type=int, default=0, metavar="N",
                   help="Начать с этапа N (0-5)")
    p.add_argument("--only-stage", type=int, default=None, metavar="N",
                   help="Запустить только этап N")
    p.add_argument("--no-skip", action="store_true",
                   help="Не пропускать уже выполненные этапы")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Рабочая директория (по умолчанию ./pipeline_output)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.data.exists():
        print(f"Ошибка: файл не найден: {args.data}")
        sys.exit(1)

    cfg = Config(
        input_json  = args.data,
        work_dir    = args.work_dir,
        skip        = not args.no_skip,
    )
    log = get_logger("pipeline", cfg.LOG_DIR)

    log.info("=" * 60)
    log.info("ЗАПУСК ПАЙПЛАЙНА КЛАССИФИКАЦИИ")
    log.info(f"Входной файл : {cfg.INPUT_JSON_PATH}")
    log.info(f"Рабочая папка: {cfg.WORK_DIR}")
    log.info("=" * 60)

    if args.only_stage is not None:
        stages_to_run = [args.only_stage]
    else:
        stages_to_run = list(range(args.from_stage, len(STAGES)))

    t_total = time.time()
    for num in stages_to_run:
        if num not in STAGES:
            log.error(f"Неизвестный этап: {num}. Допустимые: 0-5")
            sys.exit(1)
        name, module = STAGES[num]
        log.info(f"\n{'─' * 50}")
        log.info(f"  Этап {num}: {name}")
        log.info(f"{'─' * 50}")
        try:
            module.run(cfg, log)
        except Exception as e:
            log.error(f"Ошибка на этапе {num} ({name}): {e}")
            log.error("Перезапуск с этого места:")
            log.error(f"  python scripts/run_pipeline.py --data {cfg.INPUT_JSON_PATH} --from-stage {num}")
            raise

    elapsed = int(time.time() - t_total)
    h, rem = divmod(elapsed, 3600)
    m, s   = divmod(rem, 60)
    log.info("\n" + "=" * 60)
    log.info(f"ГОТОВО за {h}ч {m}м {s}с")
    log.info(f"Метрики      : {cfg.METRICS_JSON}")
    log.info(f"Финальный датасет: {cfg.TRAIN_FINAL_CSV}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

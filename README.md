# Классификация текстовых документов на основе контекстуальных векторных представлений

**Выпускная квалификационная работа магистра**  
Ворошнина А.О., НИЯУ МИФИ, кафедра №42, 2026  
Направление: 01.04.02 «Прикладная математика и информатика»

---

## О проекте

Исследование методов классификации русскоязычных деловых документов в условиях ограниченных ресурсов. Корпус — 1774 деловых письма нефтедобывающей компании, 36 классов, сильный дисбаланс (коэффициент 200x), данные деперсонализированы через плейсхолдеры.

**Лучший результат:** ансамбль трёх моделей → macro F1 = 0.455, balanced accuracy = 0.451  
**Калибровка:** ECE снижена с 0.113 до 0.068 (T=1.25)

---

## Структура репозитория

```
vkr-text-classification/
├── data/                        # Датасеты
│   ├── original_data.json       # Исходный корпус (1774 документа)
│   ├── train.csv / test.csv     # После стратифицированного split 80/20
│   ├── train_after_stage1.csv   # После LLM-генерации
│   ├── train_after_stage2.csv   # После парафраза
│   ├── train_after_stage3.csv   # После обратного перевода
│   └── train_final.csv          # Финальный датасет (2278 документов)
│
├── src/                         # Исходный код
│   ├── augmentation/
│   │   ├── aug_utils.py         # Валидаторы, маскировка плейсхолдеров
│   │   ├── stage1_generate.py   # LLM-генерация (порог 20)
│   │   ├── stage2_paraphrase.py # Парафраз (порог 40)
│   │   ├── stage3_backtrans.py  # Обратный перевод NLLB-200 (порог 55)
│   │   └── stage4_validate.py   # Финальная валидация
│   ├── classification/
│   │   ├── prepare.py           # Очистка текстов + train/test split
│   │   └── classify.py          # BERT fine-tuning + ансамбль + калибровка
│   └── utils/
│       ├── config.py            # Конфигурация пайплайна
│       └── utils.py             # Логирование, set_seed, timer
│
├── scripts/
│   └── run_pipeline.py          # Точка входа — запуск всего пайплайна
│
├── notebooks/                   # Jupyter-ноутбуки (исследование)
│   ├── augmentation_stage1.ipynb
│   ├── augmentation_stage2.ipynb
│   ├── augmentation_stage3.ipynb
│   ├── final_validation.ipynb
│   ├── classifier_combined.ipynb
│   ├── ablation_rubert.ipynb
│   └── ensemble_calibration.ipynb
│
├── EDA/                         # Разведочный анализ данных
│   ├── eda.ipynb
│   └── *.png                    # Графики EDA
│
├── results/                     # Результаты экспериментов
│   ├── ablation_results.csv
│   ├── metrics.json
│   └── *.png                    # Графики результатов
│
├── requirements.txt
└── README.md
```

---

## Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск полного пайплайна

```bash
python scripts/run_pipeline.py --data /path/to/data.json
```

### Перезапуск с определённого этапа (если прервался)

```bash
python scripts/run_pipeline.py --data /path/to/data.json --from-stage 3
```

### Запуск только одного этапа

```bash
python scripts/run_pipeline.py --data /path/to/data.json --only-stage 5
```

### Указать рабочую директорию

```bash
python scripts/run_pipeline.py --data /path/to/data.json --work-dir /path/to/output
```

---

## Этапы пайплайна

| # | Название | Описание | Модель |
|---|----------|----------|--------|
| 0 | Подготовка | Очистка текстов + train/test split 80/20 | — |
| 1 | LLM-генерация | Генерация новых писем для классов < 20 | Qwen2.5-14B-Instruct-AWQ |
| 2 | Парафраз | Переформулирование для классов < 40 | Qwen2.5-14B-Instruct-AWQ |
| 3 | Обратный перевод | RU→EN→RU для классов < 55 | NLLB-200-distilled-600M |
| 4 | Валидация | Удаление брака, дубликатов | — |
| 5 | Классификация | Fine-tuning BERT + ансамбль + калибровка | rubert-base, ruRoberta-large, RuModernBERT |

---

## Конфигурация

Все настройки в одном месте — `src/utils/config.py`.  
Павлу нужно поменять только одну строку:

```python
self.INPUT_JSON_PATH = Path("/path/to/your/data.json")
```

Формат входного JSON:
```json
[
  {"idx": "10026", "text": "Уважаемый [PERSON]...", "label": "Блок технического директора"},
  ...
]
```

---

## Результаты

### Влияние аугментации (macro F1)

| Модель | Оригинал | Аугментация | Δ |
|--------|----------|-------------|---|
| TF-IDF + LogReg | 0.404 | 0.441 | +0.037 |
| rubert-tiny2 | 0.006 | 0.015 | +0.009 |
| rubert-base | 0.259 | 0.381 | +0.122 |
| ruRoberta-large | 0.273 | 0.488 | +0.215 |
| RuModernBERT-base | 0.347 | 0.401 | +0.054 |

### Ансамбль и калибровка

| Конфигурация | Balanced Accuracy | Macro F1 |
|--------------|-------------------|----------|
| rubert-base | 0.423 | 0.419 |
| ruRoberta-large | 0.452 | 0.451 |
| RuModernBERT-base | 0.319 | 0.326 |
| **Ансамбль** | **0.451** | **0.455** |

ECE до калибровки: 0.113 → после (T=1.25): **0.068**

---

## Требования

- Python 3.10+
- CUDA-совместимый GPU (рекомендуется ≥ 24 GB VRAM для этапов 1-2)
- Для этапа 3 (NLLB) достаточно 8 GB VRAM

---

## Автор

Ворошнина Анна Олеговна  
Научный консультант: Вавилов П.Д.  
НИЯУ МИФИ, 2026

"""config.py — конфигурация пайплайна. Павел меняет только INPUT_JSON_PATH."""

from pathlib import Path


class Config:
    def __init__(self, input_json: Path, work_dir: Path = None, skip: bool = True):

        # ── ВХОД ────────────────────────────────────────────────────────────
        self.INPUT_JSON_PATH = Path(input_json)

        # ── РАБОЧАЯ ДИРЕКТОРИЯ ──────────────────────────────────────────────
        self.WORK_DIR = Path(work_dir) if work_dir else Path("./pipeline_output")
        self.LOG_DIR  = self.WORK_DIR / "logs"
        self.WORK_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

        # ── ПРОМЕЖУТОЧНЫЕ ФАЙЛЫ ─────────────────────────────────────────────
        self.TRAIN_CSV          = self.WORK_DIR / "01_train.csv"
        self.TEST_CSV           = self.WORK_DIR / "01_test.csv"
        self.STAGE1_CSV         = self.WORK_DIR / "02_after_stage1.csv"
        self.STAGE2_CSV         = self.WORK_DIR / "03_after_stage2.csv"
        self.STAGE3_CSV         = self.WORK_DIR / "04_after_stage3.csv"
        self.TRAIN_FINAL_CSV    = self.WORK_DIR / "05_train_final.csv"
        self.METRICS_JSON       = self.WORK_DIR / "06_metrics.json"
        self.ENSEMBLE_PROBS_DIR = self.WORK_DIR / "07_logits"

        # ── ПРОПУСК УЖЕ ВЫПОЛНЕННЫХ ЭТАПОВ ─────────────────────────────────
        self.SKIP_IF_EXISTS = skip

        # ── ПОДГОТОВКА ДАННЫХ ────────────────────────────────────────────────
        self.RANDOM_STATE = 42
        self.TEST_SIZE    = 0.2

        # ── МОДЕЛИ ──────────────────────────────────────────────────────────
        self.LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-AWQ"
        self.BT_MODEL_NAME  = "facebook/nllb-200-distilled-600M"
        self.MAX_MODEL_LEN  = 8192

        # ── ПОРОГИ АУГМЕНТАЦИИ ───────────────────────────────────────────────
        self.STAGE1_TARGET = 20   # LLM-генерация
        self.STAGE2_TARGET = 40   # парафраз
        self.STAGE3_TARGET = 55   # back-translation

        # ── ПАРАМЕТРЫ ГЕНЕРАЦИИ ──────────────────────────────────────────────
        self.OVERSAMPLE    = 2
        self.MAX_RETRIES   = 5
        self.MIN_LENGTH    = 500
        self.SIM_HIGH      = 0.97
        self.SIM_LOW       = 0.50
        self.MAX_SRC_CHARS = 4000
        self.MAX_EXAMPLES  = 5

        # ── КЛАССИФИКАТОРЫ ───────────────────────────────────────────────────
        self.CLASSIFIER_MODELS = [
            {
                "name":          "rubert-base",
                "model_name":    "DeepPavlov/rubert-base-cased",
                "batch_size":    8,
                "epochs":        7,
                "learning_rate": 2e-5,
                "max_length":    512,
            },
            {
                "name":          "ruroberta-large",
                "model_name":    "ai-forever/ruRoberta-large",
                "batch_size":    4,
                "epochs":        5,
                "learning_rate": 1e-5,
                "max_length":    512,
            },
            {
                "name":          "rumodernbert-base",
                "model_name":    "deepvk/RuModernBERT-base",
                "batch_size":    8,
                "epochs":        5,
                "learning_rate": 2e-5,
                "max_length":    512,
            },
        ]

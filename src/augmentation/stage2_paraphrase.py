"""stage2_paraphrase.py — парафраз писем через vLLM + Qwen2.5-14B."""

import os
import gc
import random
import pandas as pd

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "0"
os.environ["TRANSFORMERS_NO_TORCHCODEC"] = "1"

from aug_utils import validate_paraphrases

SYSTEM_PROMPT = (
    "Ты переформулируешь деловое письмо на русском языке. "
    "Сохрани смысл, тему, деловой стиль и все плейсхолдеры вида "
    "[PERSON], [ORGANIZATION], [DATE_TIME] в неизменном виде. "
    "Измени формулировки, структуру предложений и порядок изложения. "
    "Пиши только текст письма — без пояснений и комментариев. "
    "Не используй Markdown-разметку."
)
PARA_PROMPT = (
    "Переформулируй следующее деловое письмо другими словами, "
    "сохранив смысл и все плейсхолдеры:\n\n{text}"
)


def run(cfg, log):
    from utils.utils import set_seed, skip_if_exists, timer
    if skip_if_exists(cfg.STAGE2_CSV, "stage2_paraphrase", cfg.SKIP_IF_EXISTS, log):
        return

    with timer(log, "Stage 2: парафраз"):
        set_seed(42)
        df = pd.read_csv(cfg.STAGE1_CSV)
        log.info(f"Загружено: {len(df)} строк")

        from vllm import LLM, SamplingParams
        llm = LLM(model=cfg.LLM_MODEL_NAME, max_model_len=cfg.MAX_MODEL_LEN,
                  dtype="float16", trust_remote_code=True)
        import sys; sys.modules["torchcodec"] = None
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("ai-forever/sbert_large_nlu_ru", device="cpu")
        para_params = SamplingParams(temperature=0.85, top_p=0.95, max_tokens=1024)

        def llm_para_batch(texts):
            convs = [[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":PARA_PROMPT.format(text=t[:cfg.MAX_SRC_CHARS])}]
                     for t in texts]
            outs = llm.chat(convs, para_params)
            return [o.outputs[0].text.strip() if o.outputs else "" for o in outs]

        def paraphrase_class(class_name, existing, n_needed):
            collected, pool = [], list(existing)
            for attempt in range(1, cfg.MAX_RETRIES + 1):
                need = n_needed - len(collected)
                if need <= 0:
                    break
                batch_n = need * cfg.OVERSAMPLE + 1
                sources = [existing[i % len(existing)] for i in range(batch_n)]
                random.shuffle(sources)
                paras = llm_para_batch(sources)
                valid = validate_paraphrases(list(zip(sources, paras)), pool, sbert,
                                             cfg.SIM_LOW, cfg.SIM_HIGH, cfg.MIN_LENGTH)
                take = valid[:need]
                collected.extend(take); pool.extend(take)
                log.info(f"  {class_name[:35]} | попытка {attempt}: "
                         f"ген {len(paras)}, прошло {len(valid)}, итого {len(collected)}/{n_needed}")
            return collected

        counts = df['label'].value_counts()
        to_aug = counts[counts < cfg.STAGE2_TARGET].sort_values()
        log.info(f"Классов для парафраза: {len(to_aug)}")

        new_rows = []
        for i, (cls, cur) in enumerate(to_aug.items(), 1):
            need = cfg.STAGE2_TARGET - cur
            existing = df[df['label'] == cls]['text'].tolist()
            log.info(f"[{i}/{len(to_aug)}] {cls[:40]}: есть {cur}, нужно {need}")
            for t in paraphrase_class(cls, existing, need):
                new_rows.append({"label": cls, "text": t, "source": "stage2_paraphrase"})
            pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True).to_csv(cfg.STAGE2_CSV, index=False)

        result = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        result.to_csv(cfg.STAGE2_CSV, index=False)
        log.info(f"Добавлено {len(new_rows)} парафразов. Итого: {len(result)}")
        del llm, sbert; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

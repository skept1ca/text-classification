"""stage1_generate.py — LLM-генерация новых писем (vLLM + Qwen2.5-14B)."""

import os
import gc
import random
import numpy as np
import pandas as pd

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "0"
os.environ["TRANSFORMERS_NO_TORCHCODEC"] = "1"

from aug_utils import validate_generated, make_examples

SYSTEM_PROMPT = (
    "Ты пишешь одно входящее электронное письмо на русском языке. "
    "Письмо должно быть развёрнутым и подробным — не менее 200 токенов. "
    "Пиши только текст письма — без пояснений и комментариев. "
    "Не используй Markdown-разметку."
)
CONTEXT_PROMPT = (
    "Вот примеры реальных писем одного класса:\n\n{examples}\n\n"
    "Опиши одним абзацем, о чём обычно эти письма: тематика, тип обращения, "
    "характерные особенности. Только описание, без вступлений."
)
GEN_PROMPT = (
    "Класс писем: «{class_name}»\n\nХарактеристика класса:\n{context}\n\n"
    "Примеры реальных писем этого класса:\n\n{examples}\n\n"
    "Напиши одно новое письмо для класса «{class_name}». "
    "Сохрани плейсхолдеры вида [PERSON], [ORGANIZATION], [DATE_TIME] "
    "в неизменном виде. Пиши только текст письма."
)


def run(cfg, log):
    from utils.utils import set_seed, skip_if_exists, timer
    if skip_if_exists(cfg.STAGE1_CSV, "stage1_generate", cfg.SKIP_IF_EXISTS, log):
        return

    with timer(log, "Stage 1: LLM-генерация"):
        set_seed(42)
        df = pd.read_csv(cfg.TRAIN_CSV)
        log.info(f"Загружено: {len(df)} строк, {df['label'].nunique()} классов")

        log.info(f"Загружаю vLLM: {cfg.LLM_MODEL_NAME}")
        from vllm import LLM, SamplingParams
        llm = LLM(model=cfg.LLM_MODEL_NAME, max_model_len=cfg.MAX_MODEL_LEN,
                  dtype="float16", trust_remote_code=True)

        import sys; sys.modules["torchcodec"] = None
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("ai-forever/sbert_large_nlu_ru", device="cpu")

        gen_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=1024)
        ctx_params = SamplingParams(temperature=0.3, top_p=0.9,  max_tokens=256)

        def llm_chat_batch(prompts, params, system=SYSTEM_PROMPT):
            convs = [[{"role":"system","content":system},
                      {"role":"user","content":p}] for p in prompts]
            outs = llm.chat(convs, params)
            return [o.outputs[0].text.strip() if o.outputs else "" for o in outs]

        def augment_class(class_name, existing, n_needed):
            ctx = llm_chat_batch(
                [CONTEXT_PROMPT.format(examples=make_examples(existing, 5))],
                ctx_params, system="Ты аналитик деловой переписки."
            )[0].split("\n\n")[0].strip()
            collected, pool = [], list(existing)
            for attempt in range(1, cfg.MAX_RETRIES + 1):
                need = n_needed - len(collected)
                if need <= 0:
                    break
                prompts = [
                    GEN_PROMPT.format(class_name=class_name, context=ctx,
                                      examples=make_examples(pool, cfg.MAX_EXAMPLES))
                    for _ in range(need * cfg.OVERSAMPLE + 1)
                ]
                cands = llm_chat_batch(prompts, gen_params)
                valid = validate_generated(cands, pool, sbert,
                                           cfg.SIM_LOW, cfg.SIM_HIGH, cfg.MIN_LENGTH)
                take = valid[:need]
                collected.extend(take); pool.extend(take)
                log.info(f"  {class_name[:35]} | попытка {attempt}: "
                         f"ген {len(cands)}, прошло {len(valid)}, итого {len(collected)}/{n_needed}")
            return collected

        counts = df['label'].value_counts()
        to_aug = counts[counts < cfg.STAGE1_TARGET].sort_values()
        log.info(f"Классов для генерации: {len(to_aug)}")

        new_rows = []
        df_orig = df.copy(); df_orig['source'] = 'original'
        for i, (cls, cur) in enumerate(to_aug.items(), 1):
            need = cfg.STAGE1_TARGET - cur
            existing = df[df['label'] == cls]['text'].tolist()
            log.info(f"[{i}/{len(to_aug)}] {cls[:40]}: есть {cur}, нужно {need}")
            for t in augment_class(cls, existing, need):
                new_rows.append({"label": cls, "text": t, "source": "stage1_llm_gen"})
            pd.concat([df_orig, pd.DataFrame(new_rows)], ignore_index=True).to_csv(cfg.STAGE1_CSV, index=False)

        result = pd.concat([df_orig, pd.DataFrame(new_rows)], ignore_index=True)
        result.to_csv(cfg.STAGE1_CSV, index=False)
        log.info(f"Добавлено {len(new_rows)} писем. Итого: {len(result)}")
        del llm, sbert; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

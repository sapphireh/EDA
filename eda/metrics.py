"""Generation metrics used by the Llama GSM-Plus experiment."""

from __future__ import annotations

import re
from typing import Dict, List

from transformers import PreTrainedTokenizerBase


ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"
EOT_TOKEN = "<|eot_id|>"

ANSWER_PATTERNS = [
    r"\\boxed\{([^}]*)\}",
    r"答案[:：]\s*([^\n]+)",
    r"Answer\s*[:：]\s*([^\n]+)",
    r"Final\s*Answer\s*[:：]\s*([^\n]+)",
    r"final\s*answer\s*is\s*([^\n]+)",
]


def normalize_answer(text: str) -> str:
    text = text.strip().replace(EOT_TOKEN, "")
    text = re.sub(r"\s+", " ", text)
    text = text.replace(",", "").replace("，", "")
    text = text.replace("。", "").replace("．", ".")
    text = text.replace(" ", "")
    text = text.strip("：:;；.!！?？（）()[]{}")
    text = re.sub(r"\\\(|\\\)", "", text)
    text = re.sub(r"\\\[|\\\]", "", text)
    return re.sub(r"\\text\{([^}]*)\}", r"\1", text)


def extract_answer(text: str) -> str:
    for pattern in ANSWER_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_answer(match.group(1))
    candidates = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
    if candidates:
        return normalize_answer(candidates[-1])
    return normalize_answer(text)


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase):
    def compute_metrics(eval_prediction) -> Dict[str, float]:
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids

        prediction_texts = tokenizer.batch_decode(
            predictions, skip_special_tokens=False
        )
        processed_predictions: List[str] = []
        for text in prediction_texts:
            header_index = text.rfind(ASSISTANT_HEADER)
            if header_index != -1:
                text = text[header_index + len(ASSISTANT_HEADER) :]
            processed_predictions.append(
                text.replace(EOT_TOKEN, "").strip()
            )

        label_texts = []
        for token_ids in labels:
            token_ids = [int(token) for token in token_ids if token != -100]
            label_texts.append(
                tokenizer.decode(
                    token_ids, skip_special_tokens=True
                ).strip()
            )

        total = min(len(processed_predictions), len(label_texts))
        correct = sum(
            extract_answer(prediction) == extract_answer(reference)
            for prediction, reference in zip(
                processed_predictions, label_texts
            )
        )
        exact_match = correct / total if total > 0 else 0.0
        return {"eval_em": exact_match}

    return compute_metrics

"""Dataset normalization, tokenization, collation, and balanced sampling."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerBase


IGNORE_INDEX = -100


def to_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize a record to a chat-style ``messages`` list."""

    if isinstance(example.get("messages"), list):
        return cast(List[Dict[str, str]], example["messages"])

    instruction = str(example.get("instruction") or "").strip()
    input_text = str(example.get("input") or "").strip()
    output = str(
        example.get("output")
        or example.get("response")
        or example.get("answer")
        or ""
    ).strip()

    if input_text:
        user_content = f"{instruction}\n\nInput:\n{input_text}"
    else:
        user_content = instruction

    messages: List[Dict[str, str]] = [
        {"role": "user", "content": user_content}
    ]
    if output:
        messages.append({"role": "assistant", "content": output})
    return messages


def build_chat_template_if_needed(tokenizer: PreTrainedTokenizerBase) -> None:
    """Install the Llama 3.1 chat template when generation markers are absent."""

    template = getattr(tokenizer, "chat_template", None)
    if not template or "{% generation %}" not in str(template):
        setattr(
            tokenizer,
            "chat_template",
            """{{ bos_token }}
{% for message in messages %}
{% set role = message['role'] %}
{% if role == 'system' %}
<|start_header_id|>system<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% elif role == 'user' %}
<|start_header_id|>user<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% elif role == 'assistant' %}
<|start_header_id|>assistant<|end_header_id|>
{% generation %}{{ message['content'] }}<|eot_id|>{% endgeneration %}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>
{% generation %}{% endgeneration %}
{% endif %}""",
        )


def apply_chat_template_safe(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
) -> List[int]:
    fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(fn):
        token_ids = fn(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors=None,
        )
        return cast(List[int], token_ids)

    parts: List[str] = []
    for message in messages:
        role = message.get("role", "user")
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n"
            f"{message.get('content', '')}<|eot_id|>"
        )
    if add_generation_prompt:
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    encoded = tokenizer("".join(parts), add_special_tokens=True)
    return cast(List[int], encoded["input_ids"])


def tokenize_training_example(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    cutoff_length: int,
    is_source: bool,
) -> Dict[str, Any]:
    """Tokenize source examples with full loss and targets without labels."""

    has_assistant = any(
        message.get("role") == "assistant"
        and str(message.get("content", "")).strip()
        for message in messages
    )

    if is_source and not has_assistant:
        raise ValueError(
            "source training records require a non-empty assistant answer"
        )

    if is_source and has_assistant:
        full_ids = apply_chat_template_safe(
            tokenizer, messages, add_generation_prompt=False
        )
        if cutoff_length and cutoff_length > 0:
            full_ids = full_ids[:cutoff_length]

        prompt_only = [
            message for message in messages if message.get("role") != "assistant"
        ]
        prompt_ids = apply_chat_template_safe(
            tokenizer, prompt_only, add_generation_prompt=False
        )
        if cutoff_length and cutoff_length > 0:
            prompt_ids = prompt_ids[:cutoff_length]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": full_ids.copy(),
            "domain": 0,
            "prompt_len": len(prompt_ids),
        }

    prompt_only = [
        message for message in messages if message.get("role") != "assistant"
    ]
    full_ids = apply_chat_template_safe(
        tokenizer, prompt_only, add_generation_prompt=False
    )
    if cutoff_length and cutoff_length > 0:
        full_ids = full_ids[:cutoff_length]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [IGNORE_INDEX] * len(full_ids),
        "domain": 1,
        "prompt_len": len(full_ids),
    }


def tokenize_evaluation_example(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    cutoff_length: int,
) -> Optional[Dict[str, Any]]:
    has_assistant = any(message.get("role") == "assistant" for message in messages)
    if not has_assistant:
        return None

    prompt_messages = [
        message for message in messages if message.get("role") != "assistant"
    ]
    prompt_ids = apply_chat_template_safe(
        tokenizer, prompt_messages, add_generation_prompt=True
    )
    # Kept to preserve the final experiment's tokenization path.
    full_ids = apply_chat_template_safe(
        tokenizer, messages, add_generation_prompt=False
    )
    if cutoff_length is not None and cutoff_length > 0:
        prompt_ids = prompt_ids[:cutoff_length]
        full_ids = full_ids[:cutoff_length]

    assistant_content = None
    for message in messages:
        if message.get("role") == "assistant":
            assistant_content = message.get("content", "")
            break
    if not assistant_content:
        return None

    answer_ids = tokenizer(
        assistant_content, add_special_tokens=False
    )["input_ids"]
    if len(answer_ids) == 0:
        return None

    return {
        "input_ids": prompt_ids,
        "attention_mask": [1] * len(prompt_ids),
        "labels": answer_ids,
        "domain": 1,
    }


def load_training_dataset(
    path: str,
    tokenizer: PreTrainedTokenizerBase,
    cutoff_length: int,
    is_source: bool,
    target_domain_key: Optional[str] = None,
    num_target_domains: int = 1,
) -> Dataset:
    loaded = load_dataset("json", data_files={"train": path})
    dataset_dict: DatasetDict = loaded
    raw = cast(Dataset, dataset_dict["train"])

    def map_record(example: Dict[str, Any]) -> Dict[str, Any]:
        record = tokenize_training_example(
            tokenizer,
            to_messages(example),
            cutoff_length,
            is_source,
        )
        if is_source:
            record["domain_id"] = 0
        else:
            domain_id = 1
            if target_domain_key:
                if target_domain_key not in example:
                    raise ValueError(
                        f"target record is missing domain key "
                        f"{target_domain_key!r}"
                    )
                try:
                    domain_id = int(example[target_domain_key])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"target domain {example[target_domain_key]!r} is "
                        "not an integer"
                    ) from error
                if not 1 <= domain_id <= num_target_domains:
                    raise ValueError(
                        f"target domain {domain_id} is outside [1, "
                        f"{num_target_domains}]"
                    )
            record["domain_id"] = domain_id
        return record

    dataset = cast(Dataset, raw.map(map_record))
    columns_to_keep = {
        "input_ids",
        "attention_mask",
        "labels",
        "domain",
        "domain_id",
        "prompt_len",
    }
    columns_to_remove = [
        column for column in dataset.column_names if column not in columns_to_keep
    ]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def load_evaluation_dataset(
    path: str,
    tokenizer: PreTrainedTokenizerBase,
    cutoff_length: int,
) -> Dataset:
    loaded = load_dataset("json", data_files={"eval": path})
    if not isinstance(loaded, DatasetDict):
        raise TypeError("Expected a DatasetDict from load_dataset for eval split")
    raw = cast(Dataset, loaded["eval"])

    def map_record(example: Dict[str, Any]):
        record = tokenize_evaluation_example(
            tokenizer, to_messages(example), cutoff_length
        )
        if record is not None:
            return record
        # Dataset.map cannot use None as a per-row deletion marker.  Emit an
        # empty record and remove it in the explicit filter below.
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "domain": 1,
        }

    mapped = raw.map(map_record)
    mapped = mapped.filter(
        lambda item: len(item["labels"]) > 0
        and len(item["input_ids"]) > 0
    )
    dataset = cast(Dataset, mapped)
    columns_to_keep = {"input_ids", "attention_mask", "labels", "domain"}
    columns_to_remove = [
        column for column in dataset.column_names if column not in columns_to_keep
    ]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def make_mixed_dataset(
    source_dataset: Dataset,
    target_dataset: Dataset,
    seed: int = 42,
) -> Dataset:
    mixed = concatenate_datasets([source_dataset, target_dataset])
    return mixed.shuffle(seed=seed)


def subsample_target_dataset(
    target_dataset: Dataset,
    sampling_ratio: float,
    seed: int = 42,
) -> Dataset:
    """Return the target split actually consumed by the balanced loader."""

    if not 0.0 < sampling_ratio <= 1.0:
        raise ValueError("target sampling ratio must be in (0, 1]")
    if sampling_ratio == 1.0:
        return target_dataset
    sample_count = max(1, int(len(target_dataset) * sampling_ratio))
    return cast(
        Dataset,
        target_dataset.shuffle(seed=seed).select(range(sample_count)),
    )


@dataclass
class CausalLMCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.max_length is not None:
            for feature in features:
                for key in ("input_ids", "attention_mask", "labels"):
                    value = feature.get(key)
                    if isinstance(value, list) and len(value) > self.max_length:
                        feature[key] = value[: self.max_length]

        encoded = self.tokenizer.pad(
            {
                "input_ids": [
                    cast(List[int], feature["input_ids"])
                    for feature in features
                ],
                "attention_mask": [
                    cast(List[int], feature["attention_mask"])
                    for feature in features
                ],
            },
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        input_ids = cast(torch.Tensor, encoded["input_ids"])
        attention_mask = cast(torch.Tensor, encoded["attention_mask"])

        # Training labels are prompt-aligned and therefore have the same
        # width as input_ids.  Evaluation labels are answer-only references
        # and may be longer than a short prompt; they must be padded on their
        # own axis because EDATrainer keeps them out of the causal-LM forward.
        max_length = max(
            input_ids.size(1),
            max(len(feature["labels"]) for feature in features),
        )
        labels = torch.full(
            (len(features), max_length), IGNORE_INDEX, dtype=torch.long
        )
        for index, feature in enumerate(features):
            feature_labels = cast(List[int], feature["labels"])
            labels[index, : len(feature_labels)] = torch.tensor(
                feature_labels, dtype=torch.long
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "domain_labels": torch.tensor(
                [int(feature.get("domain", 1)) for feature in features],
                dtype=torch.long,
            ),
            "domain_ids": torch.tensor(
                [int(feature.get("domain_id", 0)) for feature in features],
                dtype=torch.long,
            ),
            "prompt_lens": torch.tensor(
                [
                    int(feature.get("prompt_len", len(feature["input_ids"])))
                    for feature in features
                ],
                dtype=torch.long,
            ),
        }


class BalancedDomainBatchDataset(IterableDataset):
    """Yield pre-batched examples with equal source and target counts."""

    def __init__(
        self,
        source_dataset,
        target_dataset,
        batch_size: int,
        seed: int = 42,
        drop_last: bool = True,
    ):
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError("batch_size must be an even integer >= 2")
        if len(source_dataset) == 0 or len(target_dataset) == 0:
            raise ValueError("source and target datasets must both be non-empty")
        self.source = source_dataset
        self.target = target_dataset
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.seed = int(seed)
        self.drop_last = drop_last
        self._iteration = 0

    def __len__(self) -> int:
        source_batches = len(self.source) // self.half_batch
        target_batches = len(self.target) // self.half_batch
        batch_count = min(source_batches, target_batches)
        if not self.drop_last:
            batch_count = math.ceil(len(self.source) / self.half_batch)
            batch_count = min(
                batch_count,
                math.ceil(len(self.target) / self.half_batch),
            )
        return batch_count

    def __iter__(self):
        # Every worker builds the same deterministic shuffled index lists and
        # then consumes a disjoint subset of complete balanced batches.
        iteration = self._iteration
        self._iteration += 1
        rng = random.Random(self.seed + iteration)
        source_indices = list(range(len(self.source)))
        target_indices = list(range(len(self.target)))
        rng.shuffle(source_indices)
        rng.shuffle(target_indices)

        worker = get_worker_info()
        first_batch = worker.id if worker is not None else 0
        batch_stride = worker.num_workers if worker is not None else 1
        for batch_index in range(first_batch, len(self), batch_stride):
            source_start = batch_index * self.half_batch
            target_start = batch_index * self.half_batch
            source_batch_indices = [
                source_indices[(source_start + offset) % len(source_indices)]
                for offset in range(self.half_batch)
            ]
            target_batch_indices = [
                target_indices[(target_start + offset) % len(target_indices)]
                for offset in range(self.half_batch)
            ]
            examples = [
                self.source[index] for index in source_batch_indices
            ] + [self.target[index] for index in target_batch_indices]
            rng.shuffle(examples)
            yield examples

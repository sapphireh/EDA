#!/usr/bin/env python
"""Evaluate a Llama 3.1 LoRA adapter on GSM-Plus with vLLM."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve math problems step by step. "
    "Show calculations using <<a*b=c>> style and end with `#### <number>`."
    "If you can not answer the question, just response #### None."
)

# Preserve the legacy LoRA evaluator's nine-shot prompt verbatim, encoded as
# user/assistant turns through the saved Llama chat template.
FEWSHOT = [
    (
        "Three of the women at the cocktail party are wearing 4.5-inch heels and three are wearing 2.5-inch heels. What is the average height of heels at this party? ",
        "First find the total number of inches from the 4.5-inch heels: "
        "3 women * 4.5 inches/woman = <<3*4.5=13.5>>13.5 inches\n"
        "Then find the total number of inches from the 2.5-inch heels: "
        "3 women * 2.5 inches/woman = <<3*2.5=7.5>>7.5 inches\n"
        "Then add the total number of inches from each group of heels to find the overall total: 13.5 inches + 7.5 inches = <<13.5+7.5=21>>21 inches\n"
        "Then find the total number of women at the party: 3 women + 3 women = <<3+3=6>>6 \n"
        "Then divide the total number of heel inches by the number of women to find the average heel height per woman: 21 inches / 6 women = <<21/6=3.5>>3.5 inches / woman \n#### 3.5",
    ),
    (
        "Tommy caught 5 more fish than Darren and Darren caught 8 less than Naomi. If Naomi caught 20 fish, how many fish did Tommy catch?",
        "Darren caught 20 - 8 = <<20-8=12>>12 fish\nTommy caught 12 + 5 = <<12+5=17>>17 fish\n#### 17",
    ),
    (
        "Manny signed up for Karate classes for $600. His parents tell him that if his classes end up costing more than $100 per class, then they won't sign him up again. If there are 100 total classes, how many can he miss before they don't sign him up again?",
        "He must attend at least six classes, because 600 / 6 = <<600/6=100>>100\nHe can skip 4 class because 10 - 4 = <<10-4=6>>6\n#### 4",
    ),
    (
        "Julie operates the cash register exactly twice as fast as her less-experienced colleague Jewel. Daily, Jewel processes half of one hundred customers. What is the total weekly production for the two if they work all days of the week?",
        "While Jewel is processing half of one hundred orders in a day, Julie processes 0.5*100 orders/day * 2 = <<0.5*100*2=100>>100 orders/day.\nIn a day, they process 100 orders/day + 0.5*100 orders/day = <<100+0.5*100=150>>150 orders together.\nThe total number of orders the two processes in a week is 150 orders/day * 7 days/week = <<150*7=1050>>1050 orders\n#### 1050",
    ),
    (
        "Reggie, Lynn, and Paisley ran together. Paisley ran 4 miles. Reggie ran 5 times what Paisley ran and 3 miles farther than Lynn. After the run, they decided to run back to their starting point. How many miles did Lynn run?",
        "Paisley ran <<4=4>>4 miles.\nReggie ran 5 * 4 miles = <<5*4=20>>20 miles\nLynn ran 3 miles less than Reggie = 20 - 3 = 17 miles\nLynn ran <<17*2=34>>34 miles.\n#### 34",
    ),
    (
        "In a grocery store, four apples cost $5.20, and three oranges cost $3.30. Clyde pays $12 for a few apples and 5 oranges. How many apples does Clyde buy?",
        "Each apple costs $5.20/4 = $<<5.20/4=1.30>>1.30.\nEach orange cost $3.30/3 = $<<3.30/3=1.10>>1.10.\nAnd 5 oranges cost $1.10 x 5 = $<<5*1.1=5.50>>5.50.\nsome apples cost 12-5.5=6.5\nClyde buys 6.5/1.3=5 apples,\n#### 5",
    ),
    (
        "Bob is planning to excavate a pit with dimensions of 6 feet in length, 4 feet in width, and 3 feet in depth. Given that he can remove one cubic foot of soil every 3 seconds, how much time will he need to complete the excavation?",
        "The hole is 6 feet long by 4 feet wide by 3 feet deep or 6 * 4 * 3 = <<6*4*3=72>>72 cubic feet.\nBob shovels a cubic foot of earth every 3 seconds so it will take 72 * 3 = <<72*3=216>>216 seconds to dig the hole.\n#### 216",
    ),
    (
        "Bill is ordering a new truck for his construction business. He has decided to purchase a two-ton truck with several added features: a king cab upgrade, a towing package, leather seats, running boards, and the upgraded exterior light package. The base price of the truck is $30,000, and the other features are at extra cost. The king cab is an extra $7,500, leather seats are one-third the cost of the king cab upgrade, running boards are $500 less than the leather seats, and the upgraded exterior light package is $1500. Bill also considered getting a sunroof, which would have cost an additional $2000, and a premium sound system for $1500, but decided against these options. What is the total cost of Bill's new truck, in dollars?",
        "The leather seats are one-third the cost of the king cab upgrade, at a cost of $7,500/3=$<<7500/3=2500>>2500.\nThe running boards are $500 less than the leather seats at a cost of $2500-$500=$2000\nThe total cost of the truck is $30,000+$7500+$2500+$2000+$1500=$<<30000+7500+2500+2000+1500=43500>>43,500\n#### 43500",
    ),
    (
        "Tim gets a promotion that offers him a raise on his $20000 a month salary.  It also gives him a bonus worth half a month's salary. How much money will he make in a year?",
        "We don't know how much money Tim will make in a year. \n#### None",
    ),
]


NUMBER_PATTERN = (
    r"[+-]?(?:(?:\d[\d,]*)(?:\.\d+)?|\.\d+)"
    r"(?:/[+-]?(?:(?:\d[\d,]*)(?:\.\d+)?|\.\d+))?"
)


@dataclass
class EvaluationSample:
    index: int
    question: str
    reference_text: str
    reference_answer: str
    prompt_tokens: int = 0


def build_messages(question: str) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_question, example_answer in FEWSHOT:
        messages.append({"role": "user", "content": example_question})
        messages.append({"role": "assistant", "content": example_answer})
    messages.append({"role": "user", "content": question})
    return messages


def normalize_answer(answer: str) -> str:
    answer = answer.strip().replace("$", "").replace(",", "")
    if answer.lower() == "none":
        return "None"
    return answer.rstrip(".。")


def extract_final_answer(text: str) -> Optional[str]:
    """Extract GSM-Plus numeric, fractional, or ``None`` answers."""

    hash_none = re.findall(r"####\s*None\b", text, flags=re.IGNORECASE)
    hash_numbers = re.findall(
        rf"####\s*\$?\s*({NUMBER_PATTERN})", text
    )
    # Respect the final explicit #### marker if a response contains several.
    explicit_matches = [
        (match.start(), "None")
        for match in re.finditer(
            r"####\s*None\b", text, flags=re.IGNORECASE
        )
    ] + [
        (match.start(), normalize_answer(match.group(1)))
        for match in re.finditer(
            rf"####\s*\$?\s*({NUMBER_PATTERN})", text
        )
    ]
    if explicit_matches:
        return max(explicit_matches, key=lambda item: item[0])[1]

    if hash_none or hash_numbers:
        # Kept as a defensive fallback for unusual regex engines/inputs.
        return "None" if hash_none else normalize_answer(hash_numbers[-1])

    if re.search(
        r"\\boxed\{\s*(?:\\text\{\s*)?None\s*\}?\s*\}",
        text,
        flags=re.IGNORECASE,
    ):
        return "None"
    boxed = re.findall(
        rf"\\boxed\{{\s*\$?\s*({NUMBER_PATTERN})\s*\}}", text
    )
    if boxed:
        return normalize_answer(boxed[-1])

    for pattern in (
        rf"(?:the\s+)?answer\s+is\s*\$?\s*({NUMBER_PATTERN})",
        rf"final\s+answer\s*(?:is|:)?\s*\$?\s*({NUMBER_PATTERN})",
    ):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return normalize_answer(matches[-1])

    lower_text = text.lower()
    if any(
        phrase in lower_text
        for phrase in (
            "cannot compute",
            "can't compute",
            "cannot be determined",
            "not enough information",
            "don't know",
            "do not know",
        )
    ):
        return "None"

    # Do not treat an intermediate calculation's final number as the answer.
    # The prompt explicitly requires ####, while the two textual patterns
    # above cover common compliant alternatives.
    return None


def answers_equal(prediction: Optional[str], reference: str) -> bool:
    if prediction is None:
        return False
    if prediction == "None" or reference == "None":
        return prediction == reference
    try:
        difference = abs(Fraction(prediction) - Fraction(reference))
    except (ValueError, ZeroDivisionError):
        return prediction == reference
    return difference <= Fraction(1, 1_000_000)


def read_samples(path: Path, limit: Optional[int]) -> List[EvaluationSample]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and len(samples) >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            question = ""
            reference_text = ""
            messages = record.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if message.get("role") == "user":
                        question = str(message.get("content", ""))
                    elif message.get("role") == "assistant":
                        reference_text = str(message.get("content", ""))
            else:
                question = str(record.get("question", ""))
                reference_text = str(record.get("answer", ""))

            if not question or not reference_text:
                raise ValueError(
                    f"sample {index} is missing a question or reference answer"
                )
            reference_answer = extract_final_answer(reference_text)
            if reference_answer is None:
                raise ValueError(
                    f"could not extract the reference answer for sample {index}"
                )
            samples.append(
                EvaluationSample(
                    index=index,
                    question=question,
                    reference_text=reference_text,
                    reference_answer=reference_answer,
                )
            )
    if not samples:
        raise ValueError(f"no valid samples found in {path}")
    return samples


def infer_base_model(adapter_path: Path) -> str:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing adapter config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"base model path is absent from {config_path}")
    return str(base_model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Llama 3.1 LoRA adapter on GSM-Plus."
    )
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument(
        "--test-file",
        default="data/test/gsm-plus-testmini.jsonl",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Defaults to base_model_name_or_path in adapter_config.json.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Defaults to <adapter-path>/gsmplus_eval_results.jsonl.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--request-batch-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--dtype",
        choices=["auto", "half", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.request_batch_size <= 0:
        raise ValueError("--request-batch-size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_model_len <= args.max_new_tokens:
        raise ValueError("--max-model-len must exceed --max-new-tokens")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1)")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    adapter_path = Path(args.adapter_path).resolve()
    test_file = Path(args.test_file).resolve()
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(
            f"missing LoRA weights under {adapter_path}"
        )
    if not test_file.is_file():
        raise FileNotFoundError(f"test file does not exist: {test_file}")

    base_model = args.base_model or infer_base_model(adapter_path)
    output_file = Path(args.output_file).resolve() if args.output_file else (
        adapter_path / "gsmplus_eval_results.jsonl"
    )
    summary_file = output_file.with_suffix(".summary.json")
    if not args.overwrite:
        existing = [path for path in (output_file, summary_file) if path.exists()]
        if existing:
            raise FileExistsError(
                "evaluation output already exists; pass --overwrite: "
                + ", ".join(map(str, existing))
            )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {base_model}")
    print(f"LoRA adapter: {adapter_path}")
    print(f"Test file: {test_file}")
    samples = read_samples(test_file, args.limit)
    print(f"Samples: {len(samples)}")

    # Match the original Llama 3.1 evaluation protocol: prompts are rendered
    # with the base model's official chat template, while only the LoRA
    # weights are loaded from adapter_path.
    tokenizer_path = str(base_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.chat_template is None:
        raise ValueError("the base model tokenizer has no chat template")

    messages = []
    maximum_prompt_tokens = args.max_model_len - args.max_new_tokens
    for sample in tqdm(samples, desc="Preparing prompts"):
        sample_messages = build_messages(sample.question)
        prompt_ids = tokenizer.apply_chat_template(
            sample_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        sample.prompt_tokens = len(prompt_ids)
        if sample.prompt_tokens > maximum_prompt_tokens:
            raise ValueError(
                f"sample {sample.index} prompt has {sample.prompt_tokens} "
                f"tokens, exceeding the allowed {maximum_prompt_tokens}"
            )
        messages.append(sample_messages)

    with (adapter_path / "adapter_config.json").open(
        "r", encoding="utf-8"
    ) as handle:
        adapter_config = json.load(handle)
    lora_rank = int(adapter_config.get("r", 16))

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print("Initializing vLLM...")
    llm = LLM(
        model=base_model,
        tokenizer=tokenizer_path,
        enable_lora=True,
        max_lora_rank=lora_rank,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    lora_request = LoRARequest("eda_lora", 1, str(adapter_path))
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        seed=args.seed,
        stop=["<|eot_id|>"],
    )

    total = correct = extraction_failures = 0
    category_stats = {
        index: {"total": 0, "correct": 0, "extraction_failures": 0}
        for index in range(8)
    }

    with output_file.open("w", encoding="utf-8") as result_handle:
        for start in tqdm(
            range(0, len(samples), args.request_batch_size),
            desc="Evaluating batches",
        ):
            end = min(start + args.request_batch_size, len(samples))
            batch_samples = samples[start:end]
            batch_messages = messages[start:end]
            outputs = llm.chat(
                batch_messages,
                sampling_params=sampling_params,
                use_tqdm=True,
                lora_request=lora_request,
            )
            if len(outputs) != len(batch_samples):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for "
                    f"{len(batch_samples)} requests"
                )

            for sample, output in zip(batch_samples, outputs):
                completion = output.outputs[0] if output.outputs else None
                response = completion.text.strip() if completion else ""
                prediction = extract_final_answer(response)
                is_correct = answers_equal(
                    prediction, sample.reference_answer
                )
                extraction_failed = prediction is None
                total += 1
                correct += int(is_correct)
                extraction_failures += int(extraction_failed)

                category = sample.index % 8
                category_stats[category]["total"] += 1
                category_stats[category]["correct"] += int(is_correct)
                category_stats[category]["extraction_failures"] += int(
                    extraction_failed
                )

                record: Dict[str, Any] = {
                    "index": sample.index,
                    "variant_slot": category,
                    "question": sample.question,
                    "reference_answer": sample.reference_answer,
                    "predicted_answer": prediction,
                    "correct": is_correct,
                    "extraction_failed": extraction_failed,
                    "prompt_tokens": sample.prompt_tokens,
                    "generated_tokens": (
                        len(completion.token_ids) if completion else 0
                    ),
                    "finish_reason": (
                        completion.finish_reason if completion else None
                    ),
                    "model_response": response,
                }
                result_handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
            result_handle.flush()
            print(
                f"Processed {total}/{len(samples)} | "
                f"accuracy={correct / total:.2%} | "
                f"extraction_failures={extraction_failures}"
            )

    category_summary = {}
    for category, stats in category_stats.items():
        category_total = stats["total"]
        category_summary[str(category)] = {
            **stats,
            "accuracy": (
                stats["correct"] / category_total
                if category_total
                else None
            ),
        }

    summary = {
        "base_model": base_model,
        "tokenizer_path": tokenizer_path,
        "adapter_path": str(adapter_path),
        "test_file": str(test_file),
        "results_file": str(output_file),
        "num_fewshot_examples": len(FEWSHOT),
        "max_new_tokens": args.max_new_tokens,
        "request_batch_size": args.request_batch_size,
        "seed": args.seed,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "extraction_failures": extraction_failures,
        "extraction_success_rate": (
            (total - extraction_failures) / total if total else 0.0
        ),
        "variant_slot_stats": category_summary,
    }
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\nEvaluation complete")
    print(f"Accuracy: {correct}/{total} = {summary['accuracy']:.2%}")
    print(
        "Extraction success rate: "
        f"{summary['extraction_success_rate']:.2%}"
    )
    print(f"Detailed results: {output_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()

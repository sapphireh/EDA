#!/usr/bin/env python
"""Train Elastic Domain Adaptation on Llama 3.1."""

from __future__ import annotations

import math
import os

import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
)

from eda.config import parse_args
from eda.data import (
    CausalLMCollator,
    build_chat_template_if_needed,
    load_evaluation_dataset,
    load_training_dataset,
    make_mixed_dataset,
    subsample_target_dataset,
)
from eda.metrics import build_compute_metrics
from eda.model import EDAModel
from eda.trainer import (
    EDAProgressCallback,
    EDATrainer,
    TensorBoardMetricsCallback,
    pretrain_domain_discriminators,
)
from eda.utils import (
    build_lora_config,
    build_quantization_config,
    detect_attention_implementation,
    get_dtype,
    set_seed,
)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.log_dir is None:
        args.log_dir = os.path.join(args.output_dir, "tb")
    os.makedirs(args.log_dir, exist_ok=True)

    torch_dtype = get_dtype()
    quantization_config = build_quantization_config(args.use_qlora)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True
    )
    if getattr(tokenizer, "pad_token", None) is None:
        setattr(tokenizer, "pad_token", tokenizer.eos_token)
    tokenizer.padding_side = "right"
    build_chat_template_if_needed(tokenizer)
    effective_num_target_domains = (
        args.num_target_domains if args.enable_multi_domain else 1
    )

    print("Loading and tokenizing datasets...")
    source_dataset = load_training_dataset(
        args.source_train,
        tokenizer,
        args.cutoff_length,
        is_source=True,
        target_domain_key=None,
        num_target_domains=effective_num_target_domains,
    )
    target_dataset = load_training_dataset(
        args.target_train,
        tokenizer,
        args.cutoff_length,
        is_source=False,
        target_domain_key=(
            args.target_domain_key if args.enable_multi_domain else None
        ),
        num_target_domains=effective_num_target_domains,
    )
    target_dataset = subsample_target_dataset(
        target_dataset,
        sampling_ratio=args.target_sampling_ratio,
        seed=args.seed,
    )
    train_dataset = make_mixed_dataset(
        source_dataset,
        target_dataset,
        seed=args.seed,
    )
    evaluation_dataset = (
        load_evaluation_dataset(
            args.validation_file,
            tokenizer,
            args.cutoff_length,
        )
        if args.validation_file
        else None
    )
    collator = CausalLMCollator(
        tokenizer=tokenizer, max_length=args.cutoff_length
    )

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch_dtype,
        quantization_config=quantization_config,
        attn_implementation=detect_attention_implementation(),
        device_map="auto" if quantization_config is not None else None,
    )
    base_model.config.use_cache = False
    if quantization_config is not None:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=False,
        )
    elif torch.cuda.is_available():
        # Discriminator pretraining runs before Trainer performs placement.
        # Put a non-quantized model on its training device now instead of
        # accidentally executing that phase on CPU.
        base_model.to(torch.device("cuda"))
    base_model = get_peft_model(
        base_model,
        build_lora_config(
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
        ),
    )

    model = EDAModel(
        base_model=base_model,
        domain_loss_weight=args.domain_loss_weight,
        activation_pooling=args.activation_pooling,
        grl_gamma=args.grl_gamma,
        num_target_domains=effective_num_target_domains,
        enable_als=args.enable_als,
        als_temperature_init=args.als_temperature_init,
        als_temperature_min=args.als_temperature_min,
        als_use_ste=args.als_use_ste,
        als_ste_threshold=args.als_ste_threshold,
        als_distance_weight=args.als_distance_weight,
        als_distance_metric=args.als_distance_metric,
        esm_use_adapter=args.esm_use_adapter,
        esm_adapter_hidden_dim=args.esm_adapter_hidden_dim,
        esm_adapter_depth=args.esm_adapter_depth,
        esm_adapter_dropout=args.esm_adapter_dropout,
        esm_shift_scale=args.esm_shift_scale,
        esm_adapter_use_gate=args.esm_adapter_use_gate,
        esm_adapter_spectral_norm=args.esm_adapter_spectral_norm,
        esm_use_fusion=args.esm_use_fusion,
        esm_fusion_method=args.esm_fusion_method,
        esm_fusion_hidden_dim=args.esm_fusion_hidden_dim,
        esm_discriminator_hidden_dim=args.esm_discriminator_hidden_dim,
        esm_discriminator_depth=args.esm_discriminator_depth,
    )
    model.pad_token_id = int(
        tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    )

    print("Pre-initializing ALS and ESM...")
    hidden_size = base_model.config.hidden_size
    num_layers = base_model.config.num_hidden_layers
    first_parameter = next(base_model.parameters())
    model.initialize_eda_modules(
        hidden_size,
        num_layers,
        first_parameter.device,
        first_parameter.dtype,
    )
    print(
        f"Initialized {num_layers} layers (excluding embedding) "
        f"with {model.num_target_domains} target domains"
    )
    print(
        "  - Adaptive Layer Selection: "
        f"{model.adaptive_layer_selection is not None}"
    )
    print(
        "  - Elastic Shift Module: "
        f"{model.elastic_shift_module is not None}"
    )

    if args.discriminator_pretrain_steps > 0:
        print(
            "Pretraining domain discriminators before joint EDA training "
            f"for {args.discriminator_pretrain_steps} optimizer steps..."
        )
        pretrain_domain_discriminators(
            model=model,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            data_collator=collator,
            batch_size=args.train_batch_size,
            optimizer_steps=args.discriminator_pretrain_steps,
            layer_index=args.discriminator_pretrain_layer,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )

    effective_eval_strategy = (
        args.evaluation_strategy
        if evaluation_dataset is not None
        else "no"
    )
    evaluation_enabled = effective_eval_strategy != "no"
    effective_save_strategy = (
        effective_eval_strategy if evaluation_enabled else "steps"
    )
    effective_save_steps = args.save_steps
    if effective_eval_strategy == "steps":
        effective_save_steps = (
            math.ceil(args.save_steps / args.eval_steps) * args.eval_steps
        )

    if evaluation_enabled:
        generation_config = getattr(base_model, "generation_config", None)
        if generation_config is not None:
            generation_config.max_new_tokens = int(args.eval_max_new_tokens)
            generation_config.num_beams = int(args.eval_num_beams)

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=max(1, args.eval_batch_size),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=effective_save_steps,
        eval_steps=max(1, args.eval_steps),
        eval_strategy=effective_eval_strategy,
        save_strategy=effective_save_strategy,
        save_total_limit=2,
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16 and not args.use_qlora),
        optim="paged_adamw_32bit" if args.use_qlora else "adamw_torch",
        max_grad_norm=args.max_grad_norm,
        report_to=["tensorboard"],
        logging_dir=args.log_dir,
        predict_with_generate=evaluation_enabled,
        generation_num_beams=args.eval_num_beams,
        load_best_model_at_end=evaluation_enabled,
        metric_for_best_model="eval_em" if evaluation_enabled else None,
        greater_is_better=True if evaluation_enabled else None,
    )

    trainer = EDATrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=evaluation_dataset,
        data_collator=collator,
        callbacks=[
            EDAProgressCallback(),
            TensorBoardMetricsCallback(args.log_dir),
        ],
        compute_metrics=(
            build_compute_metrics(tokenizer)
            if evaluation_dataset is not None
            else None
        ),
        source_train_dataset=source_dataset,
        target_train_dataset=target_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    print("Saving LoRA adapter and tokenizer...")
    base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

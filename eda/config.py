"""Command-line configuration for EDA training."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Elastic Domain Adaptation for Llama 3.1 reasoning robustness"
    )

    # Paths
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--target-train", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-dir", default=None)

    # Validation
    parser.add_argument("--validation-file", default=None)
    parser.add_argument(
        "--evaluation-strategy",
        default="epoch",
        choices=["no", "steps", "epoch"],
    )
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-num-beams", type=int, default=1)

    # Optimization
    parser.add_argument("--cutoff-length", type=int, default=4096)
    parser.add_argument("--train-batch-size", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # LoRA / QLoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-qlora", action="store_true")

    # EDA objective
    parser.add_argument("--domain-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--activation-pooling",
        default="mean",
        choices=["last_token", "mean", "bos"],
    )
    parser.add_argument("--target-sampling-ratio", type=float, default=1.0)
    parser.add_argument("--grl-gamma", type=float, default=10.0)

    # Multi-domain adaptation
    parser.add_argument("--enable-multi-domain", action="store_true")
    parser.add_argument("--num-target-domains", type=int, default=8)
    parser.add_argument("--target-domain-key", default="variant_id")

    # Adaptive Layer Selection (ALS)
    parser.add_argument("--enable-als", action="store_true")
    parser.add_argument("--als-temperature-init", type=float, default=5.0)
    parser.add_argument("--als-temperature-min", type=float, default=0.5)
    parser.add_argument("--als-use-ste", action="store_true")
    parser.add_argument("--als-ste-threshold", type=float, default=0.5)
    parser.add_argument("--als-distance-weight", type=float, default=0.0)
    parser.add_argument(
        "--als-distance-metric",
        default="l2",
        choices=["l2", "l1", "cosine", "mmd", "coral"],
    )

    # Elastic Shift Module (ESM): perturbation-specific adapters
    parser.add_argument("--esm-use-adapter", action="store_true")
    parser.add_argument("--esm-adapter-hidden-dim", type=int, default=1024)
    parser.add_argument("--esm-adapter-depth", type=int, default=15)
    parser.add_argument("--esm-adapter-dropout", type=float, default=0.1)
    parser.add_argument("--esm-shift-scale", type=float, default=0.5)
    parser.add_argument("--esm-adapter-use-gate", action="store_true")
    parser.add_argument("--esm-adapter-spectral-norm", action="store_true")

    # Elastic Shift Module (ESM): activation fusion
    parser.add_argument("--esm-use-fusion", action="store_true")
    parser.add_argument(
        "--esm-fusion-method",
        default="film",
        choices=["concat", "film", "cross_attn"],
    )
    parser.add_argument("--esm-fusion-hidden-dim", type=int, default=1024)

    # Elastic Shift Module (ESM): domain discriminators
    parser.add_argument("--esm-discriminator-hidden-dim", type=int, default=1024)
    parser.add_argument("--esm-discriminator-depth", type=int, default=15)
    parser.add_argument(
        "--discriminator-pretrain-steps",
        type=int,
        default=0,
        help=(
            "Pretrain domain discriminators for N optimizer steps before "
            "joint EDA training (0 disables pretraining)."
        ),
    )
    parser.add_argument(
        "--discriminator-pretrain-layer",
        type=int,
        default=4,
        help=(
            "Zero-based transformer-block activation index used during "
            "discriminator pretraining (4 selects the fifth block)."
        ),
    )

    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 < args.target_sampling_ratio <= 1.0:
        parser.error("--target-sampling-ratio must be in (0, 1]")
    if args.train_batch_size < 2 or args.train_batch_size % 2 != 0:
        parser.error("--train-batch-size must be an even integer >= 2")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    if args.cutoff_length <= 0:
        parser.error("--cutoff-length must be positive")
    if args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be positive")
    if args.eval_steps <= 0 or args.save_steps <= 0:
        parser.error("--eval-steps and --save-steps must be positive")
    if args.eval_max_new_tokens <= 0 or args.eval_num_beams <= 0:
        parser.error(
            "--eval-max-new-tokens and --eval-num-beams must be positive"
        )
    if args.learning_rate <= 0 or args.num_train_epochs <= 0:
        parser.error(
            "--learning-rate and --num-train-epochs must be positive"
        )
    if not 0.0 <= args.warmup_ratio <= 1.0:
        parser.error("--warmup-ratio must be in [0, 1]")
    if args.domain_loss_weight < 0 or args.grl_gamma < 0:
        parser.error(
            "--domain-loss-weight and --grl-gamma must be non-negative"
        )
    if args.num_target_domains <= 0:
        parser.error("--num-target-domains must be positive")
    if args.als_temperature_init <= 0 or args.als_temperature_min <= 0:
        parser.error("ALS temperatures must be positive")
    if args.als_temperature_min > args.als_temperature_init:
        parser.error(
            "--als-temperature-min must not exceed --als-temperature-init"
        )
    if not 0.0 <= args.als_ste_threshold <= 1.0:
        parser.error("--als-ste-threshold must be in [0, 1]")
    if args.discriminator_pretrain_steps < 0:
        parser.error("--discriminator-pretrain-steps must be non-negative")
    if args.discriminator_pretrain_layer < 0:
        parser.error("--discriminator-pretrain-layer must be non-negative")
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        parser.error("--lora-r and --lora-alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must be in [0, 1)")
    if not 0.0 <= args.esm_adapter_dropout < 1.0:
        parser.error("--esm-adapter-dropout must be in [0, 1)")
    if (
        args.esm_adapter_hidden_dim <= 0
        or args.esm_adapter_depth < 0
        or args.esm_fusion_hidden_dim <= 0
        or args.esm_discriminator_hidden_dim <= 0
        or args.esm_discriminator_depth <= 0
    ):
        parser.error(
            "ESM dimensions and discriminator depth must be positive; "
            "adapter depth must be non-negative"
        )
    return args

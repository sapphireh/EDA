"""EDA-specific Trainer, callbacks, and discriminator pretraining."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from peft.utils.save_and_load import (
    load_peft_weights,
    set_peft_model_state_dict,
)
from transformers import (
    Seq2SeqTrainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer import TRAINING_ARGS_NAME

from .data import BalancedDomainBatchDataset
from .model import EDAModel


EDA_MODULE_WEIGHTS_NAME = "eda_modules.pt"


class TensorBoardMetricsCallback(TrainerCallback):
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer: Optional[SummaryWriter] = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if args.local_rank in (-1, 0):
            self.writer = SummaryWriter(self.log_dir)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.writer is None:
            return
        model = kwargs.get("model")
        step = int(state.global_step)

        total_loss = getattr(model, "_last_total_loss", None)
        lm_loss = getattr(model, "_last_lm_loss", None)
        domain_loss = getattr(model, "_last_domain_loss", None)
        if total_loss is not None:
            self.writer.add_scalar("train/loss", float(total_loss), step)
        if lm_loss is not None:
            self.writer.add_scalar("train/lm_loss", float(lm_loss), step)
        if domain_loss is not None:
            self.writer.add_scalar(
                "train/domain_loss", float(domain_loss), step
            )

        selected_layers = getattr(model, "_last_selected_layers", None)
        layer_weights = getattr(model, "_last_layer_weights", None)
        layer_scores = getattr(model, "_last_layer_scores", None)

        if selected_layers:
            self.writer.add_scalar(
                "als/num_selected_layers", len(selected_layers), step
            )
            self.writer.add_text(
                "als/selected_layers",
                ",".join(map(str, selected_layers)),
                step,
            )
            if layer_weights:
                for layer_index, weight in zip(
                    selected_layers, layer_weights
                ):
                    self.writer.add_scalar(
                        f"als/layer_{layer_index}_weight",
                        float(weight),
                        step,
                    )

        if layer_scores is not None:
            for layer_index, score in enumerate(layer_scores):
                self.writer.add_scalar(
                    f"als/layer_{layer_index}_score",
                    float(score),
                    step,
                )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ):
        if self.writer is None or metrics is None:
            return
        step = int(state.global_step)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(
                    f"eval/{key}", float(value), step
                )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


class EDAProgressCallback(TrainerCallback):
    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        model = kwargs.get("model")
        if isinstance(model, EDAModel) and state.max_steps > 0:
            if state.max_steps == 1:
                progress = 1.0
            else:
                # Step 0 uses the initial values and the final step reaches the
                # configured endpoints exactly.
                progress = state.global_step / (state.max_steps - 1)
            model.set_progress(progress)


def _build_prompt_only_subset(
    model: EDAModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    selected_indices: torch.Tensor,
    is_source: bool,
    prompt_lengths: Optional[torch.Tensor] = None,
):
    if selected_indices.numel() == 0:
        return None, None

    input_rows = []
    attention_rows = []
    for index in selected_indices.tolist():
        row_ids = input_ids[index]
        row_attention = attention_mask[index]
        if is_source:
            if prompt_lengths is None:
                prompt_length = int(row_attention.sum().item())
            else:
                prompt_length = int(prompt_lengths[index].item())
            length = max(1, prompt_length)
        else:
            length = max(1, int(row_attention.sum().item()))
        trimmed_ids = row_ids[:length]
        input_rows.append(trimmed_ids)
        attention_rows.append(torch.ones_like(trimmed_ids))

    padded_ids = pad_sequence(
        input_rows,
        batch_first=True,
        padding_value=model.pad_token_id,
    )
    padded_attention = pad_sequence(
        attention_rows,
        batch_first=True,
        padding_value=0,
    )
    device = input_ids.device
    return padded_ids.to(device), padded_attention.to(device)


def pretrain_domain_discriminators(
    model: EDAModel,
    source_dataset,
    target_dataset,
    data_collator,
    batch_size: int,
    optimizer_steps: int,
    layer_index: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    weight_decay: float,
    seed: int,
    num_workers: int = 0,
) -> None:
    """Pretrain discriminators before the joint EDA optimizer is created.

    Keeping this as a distinct phase prevents discriminator initialization
    from consuming language-model epochs, the joint learning-rate schedule, or
    the ALS/GRL annealing progress.
    """

    if optimizer_steps <= 0:
        return
    if model.elastic_shift_module is None:
        raise RuntimeError("ESM must be initialized before discriminator pretraining")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    balanced_dataset = BalancedDomainBatchDataset(
        source_dataset,
        target_dataset,
        batch_size=batch_size,
        seed=seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        balanced_dataset,
        batch_size=None,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if len(dataloader) == 0:
        raise ValueError("discriminator pretraining dataloader is empty")

    discriminator_parameters = list(
        model.elastic_shift_module.domain_discriminators.parameters()
    )
    optimizer = torch.optim.AdamW(
        discriminator_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    try:
        input_device = model.base_model.get_input_embeddings().weight.device
    except Exception:
        input_device = next(model.parameters()).device

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    completed_steps = 0

    while completed_steps < optimizer_steps:
        accumulated_micro_batches = 0
        accumulated_loss = 0.0
        made_progress = False

        for batch_index, batch in enumerate(dataloader):
            made_progress = True
            batch = {
                key: value.to(input_device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            domain_labels = batch["domain_labels"]
            domain_ids = batch["domain_ids"]
            prompt_lengths = batch["prompt_lens"]

            source_indices = torch.nonzero(
                domain_labels == 0, as_tuple=False
            ).flatten()
            target_indices = torch.nonzero(
                domain_labels == 1, as_tuple=False
            ).flatten()
            source_ids, source_mask = _build_prompt_only_subset(
                model,
                input_ids,
                attention_mask,
                source_indices,
                is_source=True,
                prompt_lengths=prompt_lengths,
            )
            target_ids, target_mask = _build_prompt_only_subset(
                model,
                input_ids,
                attention_mask,
                target_indices,
                is_source=False,
                prompt_lengths=prompt_lengths,
            )
            if source_ids is None or target_ids is None:
                continue

            with torch.no_grad():
                source_activations = model.extract_layer_activations(
                    source_ids, source_mask
                )
                target_activations = model.extract_layer_activations(
                    target_ids, target_mask
                )
            if not 0 <= layer_index < source_activations.size(1):
                raise ValueError(
                    f"discriminator pretrain layer {layer_index} is outside "
                    f"[0, {source_activations.size(1) - 1}]"
                )

            domain_loss = (
                model.elastic_shift_module.compute_discriminator_pretrain_loss(
                    source_activations[:, layer_index, :].detach(),
                    target_activations[:, layer_index, :].detach(),
                    domain_ids.index_select(0, target_indices),
                )
            )
            if domain_loss is None:
                continue

            domain_loss.backward()
            accumulated_micro_batches += 1
            accumulated_loss += float(domain_loss.detach())
            reached_accumulation = (
                accumulated_micro_batches >= gradient_accumulation_steps
            )
            reached_epoch_end = batch_index + 1 == len(dataloader)
            if not reached_accumulation and not reached_epoch_end:
                continue

            # Average the accumulated micro-batch gradients, including the
            # shorter final accumulation window at the epoch boundary.
            for parameter in discriminator_parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_micro_batches)
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    discriminator_parameters, max_grad_norm
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1
            mean_loss = accumulated_loss / accumulated_micro_batches
            print(
                "discriminator pretrain step "
                f"{completed_steps}/{optimizer_steps}: loss={mean_loss:.6f}"
            )
            accumulated_micro_batches = 0
            accumulated_loss = 0.0
            if completed_steps >= optimizer_steps:
                break

        if not made_progress:
            raise RuntimeError("discriminator pretraining produced no batches")

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()


class EDATrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        source_train_dataset=None,
        target_train_dataset=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.source_train_dataset = source_train_dataset
        self.target_train_dataset = target_train_dataset

        # Quantized or explicitly dispatched models must never be replicated
        # by nn.DataParallel.  The outer EDAModel exposes these attributes,
        # and this guard also covers single-device bitsandbytes maps where
        # Trainer would otherwise leave n_gpu > 1.
        model = self.model
        is_kbit = bool(
            getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)
        )
        has_device_map = getattr(model, "hf_device_map", None) is not None
        if is_kbit or has_device_map:
            self.is_model_parallel = True
            self.place_model_on_device = False
            self.args._n_gpu = 1

    def _unwrap_eda_model(self, model=None) -> EDAModel:
        candidate = model if model is not None else self.model
        candidate = self.accelerator.unwrap_model(candidate)
        while hasattr(candidate, "module"):
            candidate = candidate.module
        if not isinstance(candidate, EDAModel):
            raise TypeError(
                f"expected EDAModel, received {type(candidate).__name__}"
            )
        return candidate

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save compact, reloadable EDA checkpoints.

        Trainer otherwise treats EDAModel as an arbitrary nn.Module and saves
        the complete quantized base model.  Checkpoints only need the LoRA
        adapter plus ALS/ESM state; optimizer/scheduler state is handled by
        Trainer separately.
        """

        del state_dict
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self._unwrap_eda_model()
        model.base_model.save_pretrained(
            output_dir,
            safe_serialization=self.args.save_safetensors,
        )

        eda_state = {}
        for module_name in (
            "adaptive_layer_selection",
            "elastic_shift_module",
        ):
            module = getattr(model, module_name, None)
            if module is None:
                continue
            for key, value in module.state_dict().items():
                eda_state[f"{module_name}.{key}"] = value.detach().cpu()
        torch.save(
            eda_state,
            os.path.join(output_dir, EDA_MODULE_WEIGHTS_NAME),
        )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_eda_checkpoint(self, checkpoint: str, model=None) -> None:
        eda_model = self._unwrap_eda_model(model)
        adapter_state = load_peft_weights(checkpoint, device="cpu")
        active_adapter = getattr(
            eda_model.base_model, "active_adapter", "default"
        )
        if not isinstance(active_adapter, str):
            active_adapter = "default"
        set_peft_model_state_dict(
            eda_model.base_model,
            adapter_state,
            adapter_name=active_adapter,
        )

        eda_weights_path = os.path.join(
            checkpoint, EDA_MODULE_WEIGHTS_NAME
        )
        if not os.path.isfile(eda_weights_path):
            raise ValueError(
                f"checkpoint is missing {EDA_MODULE_WEIGHTS_NAME}: "
                f"{checkpoint}"
            )
        eda_state = torch.load(
            eda_weights_path,
            map_location="cpu",
            weights_only=True,
        )
        for module_name in (
            "adaptive_layer_selection",
            "elastic_shift_module",
        ):
            module = getattr(eda_model, module_name, None)
            prefix = f"{module_name}."
            module_state = {
                key[len(prefix) :]: value
                for key, value in eda_state.items()
                if key.startswith(prefix)
            }
            if module is not None and module_state:
                module.load_state_dict(module_state, strict=True)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self._load_eda_checkpoint(resume_from_checkpoint, model=model)

    def _load_best_model(self):
        if self.state.best_model_checkpoint is None:
            return
        self._load_eda_checkpoint(self.state.best_model_checkpoint)

    def _build_prompt_only_subset(
        self,
        model,
        input_ids,
        attention_mask,
        labels,
        selected_indices,
        is_source: bool,
        prompt_lengths: Optional[torch.Tensor] = None,
    ):
        del labels
        return _build_prompt_only_subset(
            model,
            input_ids,
            attention_mask,
            selected_indices,
            is_source,
            prompt_lengths,
        )

    def get_train_dataloader(self):
        if (
            self.source_train_dataset is None
            or self.target_train_dataset is None
        ):
            return super().get_train_dataloader()

        balanced_dataset = BalancedDomainBatchDataset(
            self.source_train_dataset,
            self.target_train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            seed=self.args.seed,
            drop_last=True,
        )
        if len(balanced_dataset) == 0:
            raise ValueError(
                "balanced training dataloader is empty; each domain needs "
                "at least train_batch_size / 2 records"
            )
        dataloader = DataLoader(
            balanced_dataset,
            batch_size=None,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return self.accelerator.prepare(dataloader)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys=None,
        **gen_kwargs,
    ):
        """Generate answers without treating answer-only references as LM labels.

        Evaluation records store the gold answer by itself in ``labels``.
        Passing those labels to a decoder-only causal LM alongside the much
        longer prompt is both shape-invalid and conceptually wrong.  Delegate
        generation/padding to Seq2SeqTrainer with labels removed, then return
        the references solely to the metric pipeline.
        """

        if not self.args.predict_with_generate:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
                **gen_kwargs,
            )
        if prediction_loss_only:
            return None, None, None

        reference_labels = inputs.get("labels")
        generation_inputs = dict(inputs)
        generation_inputs.pop("labels", None)
        _, generated_tokens, _ = super().prediction_step(
            model,
            generation_inputs,
            False,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )
        labels = (
            self._prepare_input(reference_labels)
            if reference_labels is not None
            else None
        )
        return None, generated_tokens, labels

    def training_step(
        self,
        model: nn.Module,
        inputs: dict,
        num_items: Optional[int] = None,
    ):
        del num_items
        model.train()
        inputs = self._prepare_inputs(inputs)

        unwrapped = self._unwrap_eda_model(model)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        domain_labels = inputs["domain_labels"]
        domain_ids = inputs["domain_ids"]

        source_indices = torch.nonzero(
            domain_labels == 0, as_tuple=False
        ).flatten()
        target_indices = torch.nonzero(
            domain_labels == 1, as_tuple=False
        ).flatten()
        if source_indices.numel() == 0:
            raise RuntimeError("EDA training requires source examples in every batch")

        lm_output = model(
            input_ids=input_ids.index_select(0, source_indices),
            attention_mask=attention_mask.index_select(0, source_indices),
            labels=labels.index_select(0, source_indices),
            domain_labels=None,
        )
        lm_loss = lm_output.get("lm_loss", None)
        if lm_loss is None:
            lm_loss = lm_output["loss"]
        if isinstance(lm_loss, torch.Tensor) and lm_loss.dim() > 0:
            lm_loss = lm_loss.mean()

        prompt_lengths = inputs["prompt_lens"]
        source_ids = source_mask = target_ids = target_mask = None
        if source_indices.numel() > 0:
            source_ids, source_mask = self._build_prompt_only_subset(
                unwrapped,
                input_ids,
                attention_mask,
                labels,
                source_indices,
                is_source=True,
                prompt_lengths=prompt_lengths,
            )
        if target_indices.numel() > 0:
            target_ids, target_mask = self._build_prompt_only_subset(
                unwrapped,
                input_ids,
                attention_mask,
                labels,
                target_indices,
                is_source=False,
                prompt_lengths=prompt_lengths,
            )

        domain_loss = None
        if source_ids is not None and target_ids is not None:
            target_domain_ids = domain_ids.index_select(0, target_indices)
            domain_loss = unwrapped.compute_alignment_loss(
                source_ids,
                source_mask,
                target_ids,
                target_mask,
                target_domain_ids,
            )

        total_loss = lm_loss
        if domain_loss is not None:
            total_loss = (
                total_loss + unwrapped.domain_loss_weight * domain_loss
            )

        completed_step = self.state.global_step + 1
        if self.accelerator.sync_gradients:
            domain_loss_text = (
                "None"
                if domain_loss is None
                else f"{float(domain_loss.detach()):.6f}"
            )
            print(
                f"[step {completed_step}] "
                f"src={int(source_indices.numel())}, "
                f"tgt={int(target_indices.numel())}, "
                f"total_loss={float(total_loss.detach()):.6f}, "
                f"lm_loss={float(lm_loss.detach()):.6f}, "
                f"domain_loss={domain_loss_text}, "
                f"alpha={unwrapped.compute_grl_alpha():.4f}"
            )

        try:
            unwrapped._last_total_loss = float(
                total_loss.detach().float().mean().item()
            )
            unwrapped._last_lm_loss = float(
                lm_loss.detach().float().mean().item()
            )
            unwrapped._last_domain_loss = (
                float(domain_loss.detach().float().mean().item())
                if domain_loss is not None
                else None
            )
        except Exception:
            pass

        # Transformers 4.56 performs gradient accumulation in Trainer's outer
        # loop and configures Accelerator with accumulation_steps=1. Match the
        # base Trainer by normalizing every micro-batch loss explicitly.
        accumulation_steps = max(
            1, int(self.current_gradient_accumulation_steps)
        )
        loss_for_backward = total_loss / accumulation_steps
        self.accelerator.backward(loss_for_backward)
        return loss_for_backward.detach()

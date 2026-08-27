# Elastic Domain Adaptation

Official Llama 3.1-8B-Instruct implementation of **Elastic Domain Adaptation
(EDA)** for improving reasoning robustness under question perturbations.

EDA contains two main components:

- **Adaptive Layer Selection (ALS):** dynamically selects Transformer layers
  for domain alignment.
- **Elastic Shift Module (ESM):** applies perturbation-specific feature shifts
  and multi-domain adversarial learning.

## Structure

```text
.
├── train.py
├── evaluate_gsmplus.py
├── eda/                         # ALS, ESM, model, data, and trainer modules
├── data/
│   ├── train/                   # source and target training splits
│   └── test/                    # GSM-Plus evaluation split
├── scripts/
│   └── train_llama31_gsmplus.sh
└── requirements.txt
```

## Installation

Python 3.10+ and a CUDA-capable environment are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default experiment uses QLoRA and requires access to the gated
`meta-llama/Meta-Llama-3.1-8B-Instruct` model on Hugging Face.

## Data

The repository includes the splits used by the provided training and
evaluation commands:

| File | Records | Description |
| --- | ---: | --- |
| `data/train/source8k_random10p_seed123.jsonl` | 5,976 | labeled original questions |
| `data/train/target8k_random10p_seed123.jsonl` | 5,976 | unlabeled perturbed questions |
| `data/test/gsm-plus-testmini.jsonl` | 2,400 | GSM-Plus evaluation subset |

Each file uses JSON Lines. Training examples follow this format:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}],"variant_id":0}
```

Source examples use `variant_id: 0`. Target examples have an empty assistant
answer and use perturbation IDs `1` through `8`. 

## Training

Run the reproduced Llama 3.1-8B experiment:

```bash
bash scripts/train_llama31_gsmplus.sh
```

The script reads the bundled data and writes the LoRA adapter and tokenizer to
`outputs/eda-llama31-gsmplus-seed123`. Local relative paths can be supplied
without editing the script:

```bash
MODEL_PATH=models/Meta-Llama-3.1-8B-Instruct \
OUTPUT_DIR=outputs/my-eda-run \
bash scripts/train_llama31_gsmplus.sh
```

Run `python train.py --help` for all training options.

## Evaluation

Evaluate the trained adapter on the bundled GSM-Plus subset:

```bash
python evaluate_gsmplus.py \
  --adapter-path outputs/eda-llama31-gsmplus-seed123
```

To use another test file:

```bash
python evaluate_gsmplus.py \
  --adapter-path outputs/my-eda-run \
  --test-file data/test/gsm-plus-test.jsonl
```

The evaluator infers the base model from `adapter_config.json`. Results are
written under the adapter directory. Run `python evaluate_gsmplus.py --help`
for all options.

## Code mapping

| Component | Implementation |
| --- | --- |
| EDA | `eda.model.EDAModel` |
| ALS | `eda.als.AdaptiveLayerSelection` |
| ESM | `eda.esm.ElasticShiftModule` |
| Elastic shift adapter | `eda.esm.ElasticShiftAdapter` |
| Domain discriminator | `eda.esm.DomainDiscriminator` |

This release currently targets Llama 3.1-8B-Instruct. ALS and ESM are used
during training; the final export contains the LoRA adapter and tokenizer.

## License

Licensed under the [Apache License 2.0](LICENSE).

<div align="center">

# 🐺 Fenrir

### LLM Fine-Tuning & Evaluation Framework

<p>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/LoRA-8B5CF6?style=flat"/>
<img src="https://img.shields.io/badge/Transformers-FF6F00?style=flat"/>
</p>

A framework for fine-tuning and evaluating large language models using LoRA adapters,<br/>with comprehensive benchmarking to compare base vs. fine-tuned performance.

</div>

---

## Overview

Fenrir provides an end-to-end pipeline for LLM fine-tuning and evaluation. It enables you to take a base language model, fine-tune it using parameter-efficient LoRA adapters on custom datasets, and then rigorously compare the fine-tuned model against the original using structured evaluation benchmarks.

The framework produces detailed evaluation reports in Markdown, making it straightforward to document and compare model performance across runs.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Fenrir Pipeline                    │
│                                                       │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐  │
│  │  Custom   │───▶│  LoRA Fine-  │───▶│  Fine-Tuned│  │
│  │  Dataset  │    │   Tuning     │    │   Model    │  │
│  │  (data/)  │    │  (train/)    │    │ (models/)  │  │
│  └──────────┘    └──────────────┘    └─────┬──────┘  │
│                                            │          │
│                                            ▼          │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐  │
│  │   Base   │───▶│  Evaluation  │───▶│  Report    │  │
│  │  Model   │    │   Suite      │    │  (.md)     │  │
│  └──────────┘    └──────────────┘    └────────────┘  │
│                                                       │
│  evaluate_base.py ←→ evaluate_lora.py                │
│  eval_base.md     ←→ eval_lora.md                    │
└─────────────────────────────────────────────────────┘
```

## Features

- **LoRA Fine-Tuning** — Parameter-efficient fine-tuning using Low-Rank Adaptation, minimizing GPU memory requirements while preserving model quality
- **Base vs. Fine-Tuned Evaluation** — Side-by-side comparison with structured evaluation scripts (`evaluate_base.py` / `evaluate_lora.py`)
- **Evaluation Reports** — Auto-generated Markdown reports (`eval_base.md` / `eval_lora.md`) documenting model responses and quality metrics
- **Custom Dataset Support** — Bring your own training data in the `data/` directory
- **Agent-Based Interaction** — `agent.py` provides an interactive interface for testing fine-tuned models
- **Prompt Management** — Centralized prompt templates in `prompts.py` for consistent evaluation

## Tech Stack

| Component | Technology |
|:---|:---|
| **Language** | Python |
| **Deep Learning** | PyTorch |
| **LLM Framework** | HuggingFace Transformers |
| **Fine-Tuning** | LoRA + QLoRA (PEFT) |
| **Model Format** | Adapter weights (models/) |

## Training Configuration

Details from the actual fine-tuning run documented in [`report.md`](report.md):

| Parameter | Value |
|:---|:---|
| **Base Model** | TinyLlama-1.1B-Chat-v1.0 |
| **Method** | LoRA + QLoRA |
| **LoRA Config** | r=8, alpha=16, dropout=0.05 |
| **Target Layers** | q_proj, v_proj |
| **Dataset** | 150+ CLI Q&A pairs (Git, Bash, Python, grep, tar) |
| **Epochs** | 1 |
| **Token Limit** | 512 |
| **Batch Size** | 2 |
| **Hardware** | RTX 3050 6GB GPU (~20 min training) |

## Project Structure

```
fenrir/
├── agent.py              # Interactive agent for model testing
├── prompts.py            # Prompt templates for evaluation
├── evaluate_base.py      # Evaluate base (pre-fine-tuning) model
├── evaluate_lora.py      # Evaluate LoRA fine-tuned model
├── eval_base.md          # Base model evaluation results
├── eval_lora.md          # Fine-tuned model evaluation results
├── report.md             # Comprehensive comparison report
├── test.py               # Test scripts
├── test2.py              # Additional test scripts
├── data/                 # Training datasets
├── train/                # Training scripts and configs
└── models/               # Saved model checkpoints & LoRA adapters
```

## Usage

```bash
# Clone the repository
git clone https://github.com/rafay-byte/fenrir.git
cd fenrir

# Fine-tune a model with LoRA
# (configure your base model and dataset in train/)
python train/train_lora.py

# Evaluate base model
python evaluate_base.py

# Evaluate fine-tuned model
python evaluate_lora.py

# Interactive testing with the agent
python agent.py
```

## Evaluation

The framework generates side-by-side evaluation reports:

- **[`eval_base.md`](eval_base.md)** — Baseline model responses and metrics
- **[`eval_lora.md`](eval_lora.md)** — Fine-tuned model responses and metrics
- **[`report.md`](report.md)** — Comprehensive comparison report (TinyLlama-1.1B, LoRA+QLoRA, RTX 3050)

This enables direct, reproducible comparison of how fine-tuning affects model behavior on specific tasks.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page or submit a pull request.

## License

This project is open source and available under the standard MIT-compatible terms for educational and research purposes.

---

<div align="center">
<sub>Developed by <a href="https://github.com/rafay-byte">Abdul Rafay Khalid</a> • BS AI Student @ PAF-IAST</sub>
</div>


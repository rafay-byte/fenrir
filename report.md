# Mini CLI Agent – Fine-Tuning Report

## A. Data Collection

I collected and formatted over 150 Q&A pairs focusing on common command-line topics including:
- Git (e.g., branching, commits)
- Bash shell usage
- File compression with tar/gzip
- Text processing with grep and find
- Python virtual environments
- Log file operations

Each entry followed a simple instruction-response format to simulate user inputs and assistant replies.

---

## B. Model and Training

**Model used:** TinyLlama-1.1B-Chat-v1.0  
This open-source model has under 2 billion parameters and is well-suited for low-resource fine-tuning.

**Fine-Tuning Approach:**
- Method: LoRA + QLoRA
- LoRA configuration: r=8, alpha=16, dropout=0.05
- Target layers: q_proj and v_proj
- Epochs: 1
- Token limit: 512
- Batch size: 2
- Hardware: Trained on a local RTX 3050 6GB GPU (approx. 20 minutes)

---

## C. CLI Agent Functionality

The `agent.py` script was built to:
- Accept a natural-language terminal prompt
- Use the fine-tuned model to generate a step-by-step plan
- Dry-run the first command if it matches a shell command (via `echo`)

---

## D. Evaluation Results

Seven prompts were used to compare outputs from the base and LoRA-tuned model.

| Prompt | Base Model Output | Fine-Tuned Model Output |
|--------|--------------------|--------------------------|
| Create a new Git branch | Irrelevant | Correct and structured |
| Compress folder to tar.gz | Off-topic | Correct command |
| Recursively list Python files | Partially correct | Partial but cleaner |
| Set up venv and install requests | Irrelevant | Correct |
| First ten lines of output.log | Confused with tail | Correct use of head |
| Delete all .tmp files | Unrelated suggestions | Somewhat relevant |
| Check disk usage of /home | Off-topic | Partially relevant |

**Summary:**  
The fine-tuned model consistently outperformed the base model, showing better understanding of shell tasks, especially for Git and Python commands. Some outputs still required refinement for structure or clarity.

---

## E. Recommendations for Improvement

1. Expand dataset with edge-case prompts and multi-step instructions.
2. Use evaluation metrics or human feedback loops to refine answer quality and structure.


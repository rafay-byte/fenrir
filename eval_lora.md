# LoRA Fine-Tuned Model Evaluation

| Prompt | LoRA Output Summary | Score (0–2) | Notes |
|--------|----------------------|-------------|-------|
| Create Git branch | git checkout -b + git status shown | 2 | Accurate commands |
| Compress reports/ | Mixes tar and S3 upload | 1 | No tar -czf shown |
| List Python files | Just repeats prompt | 0 | No find command or logic |
| Setup venv & install | Mentions pip, script | 1 | Missing python -m venv |
| Head 10 lines | Shows head -10 command | 2 | Correct and concise |
| Delete .tmp files | Misunderstood task | 0 | Irrelevant steps |
| Disk usage /home | Describes, no commands | 1 | Missing du -sh /home/* |

Total Score: 7 / 14

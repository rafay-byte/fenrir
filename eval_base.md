# Base Model Evaluation

| Prompt | Base Output Summary              | Score (0–2) | Notes |
|--------|----------------------------------|-------------|-------|
| Create Git branch | Talks about git pull             | 0 | Wrong command |
| Compress reports/ | Talks about uploading/generation | 0 | No tar or gzip used |
| List Python files | Starts Python logic              | 1 | No CLI method like find |
| Setup venv & install | Bad content about scores         | 0 | Not relevant |
| Head 10 lines | Shows tail -10 command           | 1 | Wrong direction |
| Delete .tmp files | Talks about backup/cloud         | 0 | Not about .tmp cleanup |
| Disk usage /home | Talks about LDAP                 | 0 | Wrong topic |

Total Score: 2 / 14

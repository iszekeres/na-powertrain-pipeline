Non-pipeline tools
-------------------

This folder is for helper scripts and one-off utilities used by ChatGPT or local tooling that are **not** part of the production NA_Trans LOG-FIRST / MODE-B / SMOOTH / XGEAR pipeline.

Constraints:
- Do not reference these scripts from NA_TRANS__LOGFIRST_MODEB__RUN_CMD.txt.
- Keep production pipeline logic in the repo root scripts only.
- It is safe to add, change, or remove files under tools/ without affecting the main run.

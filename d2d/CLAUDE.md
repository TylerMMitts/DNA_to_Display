# d2d: DNA to Display pipeline app

This folder is a self-contained app. It never writes outside `d2d/workspace/` and
never edits the project's code, data or models (they are read only).

- Commands run from this folder: `uv run python run.py <command>` (status, tasks,
  task-done, query, approve, run, dashboard). From the repository root:
  `python -m d2d <command>`.
- `/d2d` processes the agent tasks waiting in the workspace with this session
  (the user's Claude subscription), through the `d2d-<stage>` agents in
  `.claude/agents/`.
- Training never runs on this laptop; the training stage writes SLURM scripts.
- Keep token use low: query reports by JSON pointer, read only the listed figures.

---
name: d2d
description: Process the DNA to Display pipeline tasks waiting for an AI decision (dataset intake, image preprocessing, traits, genetic encoding, training setup, quality control, analysis), using this Claude Code session instead of API credits. Use when the user runs /d2d or asks to handle pending d2d tasks.
---

# Process pending d2d tasks

The d2d pipeline (this folder) runs every analysis itself and asks an agent only for
decisions. With the `claude-code` backend it writes each question to a task file and
waits. This skill answers those questions with this session's Claude subscription.

Keep token use low: read only the fields named below, never whole reports or images
that are not listed, and fetch extra detail with the small query commands.

## Steps

1. List what is waiting (run from this `d2d` folder):

   ```
   uv run python run.py tasks
   ```

   If nothing is pending, say so and stop.

2. For each pending task, hand it to its agent with the Agent tool (`subagent_type`
   is the task's `agent`, e.g. `d2d-images`), one task at a time, with this prompt:

   > Decide the d2d task in `<task.json path>`. From it read only `prompt`, `schema`,
   > `default_decision`, `images` and `decision_path`. Look at each file in `images`
   > with the Read tool. For anything else use
   > `uv run python run.py query <dataset> <stage> report <json-pointer>`,
   > `... artifacts`, `... table <name> <n>` or `... figure <name>` (then Read that
   > path), asking only for what you need. Write your decision as JSON matching
   > `schema` exactly to `decision_path`, then run
   > `uv run python run.py task-done <dataset> <stage>` and fix the file if it reports
   > schema errors. Finish with your `notes`.

3. After each agent returns, tell the user in one or two lines what it decided. The
   autopilot applies the decision and carries on by itself; nothing needs approving.
   (A dataset run without the autopilot, from the command line, still waits for
   `uv run python run.py approve <dataset> <stage>`.)

## Rules

- Never edit files outside `d2d/workspace/`, never change the project's code, data or
  models, and never start training.
- If a task's images or report are missing, run `task-done` anyway so the
  stage falls back to its rule-based decision, and tell the user.

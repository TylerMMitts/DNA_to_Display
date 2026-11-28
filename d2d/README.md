# d2d: DNA to Display for any dataset

Give it a folder of images plus genotypes (and traits, if you have them), and agents
organise the data, prepare the images, choose what to measure, pick the genetic
encoding, size the models, write the training jobs, watch training for failures,
and score the generated images. You follow and approve each step in a dashboard.

Everything d2d writes goes into `d2d/workspace/`. It never changes the project's own
`dataset/`, `results/`, `models/` or `code/`; it reads the trainers and the existing
models, and runs the trainers through its own launcher.

## Quick start

From the repository root:

```
uv run python -m d2d demo                 # a small synthetic dataset, to try everything
uv run python -m d2d run demo all --auto  # every stage with the rule-based agent
uv run python -m d2d dashboard            # http://127.0.0.1:8765
```

Your own data:

```
uv run python -m d2d new mydata --source D:\path\to\folder --cluster-host you@cluster --cluster-root /path/to/repo
uv run python -m d2d run mydata intake
```

Then work through the stages in the dashboard: run, read the agent's decision and
notes, edit it if you want, approve.

## The stages and their agents

| Stage | What the code measures | What the agent decides |
|---|---|---|
| intake | files, folders, table columns, which name patterns match which ID lists | image-to-line mapping, genotype file and format, trait table, split |
| images | per-image background, subject size, shape, orientation, colour; flags images unlike the rest | canvas and centring, invert or flatten the background, colour, augmentation, exclusions |
| traits | a broad trial of measurements on 300 images, how each varies and tracks imported traits, overlays | which measurements to keep and how generated images are scored |
| genetics | missingness, allele frequencies, sparsity, LD decay, PCA spectrum, cross-validated trait prediction per encoding | PCA, windows or direct columns, and their settings |
| training | training images and genotype conditions, image variety | model size(s), epochs, cluster job scripts |
| qc | training histories (spikes, collapse, divergence, memorisation, overfitting) and generated images (flat, washed out, collapsed, copies) | what went wrong and what to do |
| analysis | generates validation lines, measures them, correlates with the truth and a genomic-prediction baseline | a plain summary; test lines only on request |

Every decision is plain JSON in `workspace/results/<dataset>/d2d/<stage>/decision.json`
and applied by code, so a run is reproducible and you can always edit a decision.

## Choosing the agent: rules, API or your Claude subscription

Pick it in the dashboard's Agent menu or with `--backend`:

- **rules**: no AI. Every stage has a rule-based answer. Free, offline, and the fallback
  for the other two.
- **api**: the Claude API with an API key (`ANTHROPIC_API_KEY`). Fully automatic.
- **claude-code**: your Claude Pro or Max subscription through Claude Code, no API
  credits. Anthropic does not allow third-party apps to offer claude.ai sign-in, so d2d
  never signs in for you. Instead it leaves each question as a task file. Open Claude
  Code in this `d2d/` folder and run `/d2d`. The skill hands each task to its agent
  (`.claude/agents/d2d-<stage>.md`), which writes the decision back, and the dashboard
  picks it up.

## Keeping token use low

- An agent is asked only when the rules are unsure. Clear-cut stages (a sparse
  genotype design, a dataset with no flagged images, model sizing) skip the call.
  `--force-agent` asks anyway.
- The agent gets a short digest (often under 1,000 tokens), not the report. It pulls
  more with small retrieval tools: one report section by JSON pointer, a few table
  rows, one figure.
- Identical questions are answered from a cache (`workspace/.d2d_cache/`).
- The agent's instructions and tools form a stable prefix marked for prompt caching.
- Figures are sent at no more than 1,024 px, and only the ones a stage lists.
- Effort is set per stage: high only for intake, low for sizing and statistics.
- Every API call's tokens are logged (`usage.jsonl`, dashboard: Activity & usage).

## Training

The training stage writes `train_litevae.sbatch`, `train_diffusion.sbatch` and
`copy_to_cluster.txt` in `workspace/results/<dataset>/d2d/training/`. On the cluster,
pull the repository, copy the listed folders, and submit from the repository root. The
jobs run the project's unmodified trainers through `python -m d2d.launch_trainer`,
which points them at the workspace. Copy `workspace/results/training/<run>/` back to
watch a run in the dashboard. QC checks run on every refresh.

## Layout

```
d2d/
  cli.py, project.py, llm.py, retrieval.py   commands, workspace, agent backends, retrieval
  stages/                                    one module per stage
  imaging.py, measure.py, qc_checks.py       image survey, trait measurement, QC checks
  launch_trainer.py, generate.py             run the project's trainers / sampling on the workspace
  dashboard/                                 the web app
  .claude/agents/, .claude/skills/d2d/       the Claude Code agents and the /d2d skill
  workspace/                                 everything d2d writes (git-ignored)
```

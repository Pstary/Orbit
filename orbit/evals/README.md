# Orbit Eval Runners

## SWE-bench Lite

Install eval dependencies:

```bash
python -m pip install -e ".[eval]"
```

Run one SWE-bench Lite instance:

```bash
orbit-swe-bench-lite \
  --limit 1 \
  --force \
  --sandbox docker \
  --permission-mode full_auto
```

Compare Orbit agent mode with a direct prompt-only LLM baseline:

```bash
orbit-swe-bench-lite \
  --limit 5 \
  --force \
  --eval-mode both \
  --post-test-command "python -m pytest -q"
```

The summary reports success rate by mode:

```text
agent:  success_rate=...
direct: success_rate=...
```

`agent` mode lets Orbit inspect/edit/run tools through the harness. `direct`
mode calls the LLM once with only the SWE-bench prompt fields and expects a
unified diff patch; it does not let the model inspect the repository.

Run selected instances:

```bash
orbit-swe-bench-lite \
  --instance-id django__django-11099 \
  --force
```

Use a local JSONL file instead of Hugging Face:

```bash
orbit-swe-bench-lite \
  --source ./sample_swe_lite.jsonl \
  --force
```

Outputs are written under:

```text
orbit/evals/runs/YYYYMMDD/
  summary.json
  results.jsonl
  agent/<instance_id>/      # when --eval-mode both
  direct/<instance_id>/     # when --eval-mode both
  <instance_id>/            # when running a single mode
    repo/
    model.patch
    result.json
    trace/trace-*.json
    test_logs/
```

This runner evaluates Orbit's agent loop and harness execution path. It does
not replace the official SWE-bench grader. For leaderboard-compatible scoring,
submit `model.patch` files to the official SWE-bench evaluation harness.

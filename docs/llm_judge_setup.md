# Local LLM setup

## Setup

Follow the steps below to set up the local LLM (gpt-oss-20b). This is required to evaluate any report generation or red-team tasks.

Use two terminals: keep the judge server running in terminal 1, and run NeuroBench evaluations from terminal 2. Replace `/path/to/neuro_bench` with your local repository path in every command.

### 1. Check the machine

You need Python 3.12 with `venv` support, an NVIDIA GPU supported by vLLM's GPT-OSS implementation, a compatible NVIDIA driver, and internet access to download packages and model weights. A100 and H100 are examples of supported GPUs. See the [vLLM GPT-OSS hardware guide](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html).

In terminal 1, check that the GPU is visible:

```bash
nvidia-smi
python3.12 --version
```


### 2. Install vLLM and NeuroBench

Create a fresh environment in `neuro_bench/`:

```bash
cd /path/to/neuro_bench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip uv
uv pip install vllm --torch-backend=auto
uv pip install -e .
```

This uses vLLM's [recommended installation with automatic PyTorch backend selection](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/). Install into the fresh environment so vLLM can select compatible dependencies.

### 3. Start GPT-OSS-20B in terminal 1

With that environment still active, launch one model instance on one GPU:

```bash
vllm serve openai/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --host 127.0.0.1 \
  --port 8000
```

vLLM downloads the model on its first launch and exposes an OpenAI-compatible endpoint. Initial downloading and initialization can take several minutes. Leave this command running while you evaluate. See the [OpenAI GPT-OSS serving guide](https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm) for background.

### 4. Verify the endpoint in terminal 2

Open another terminal on the same machine and activate the same environment:

```bash
cd /path/to/neuro_bench
source .venv/bin/activate
curl --fail --silent --show-error http://localhost:8000/v1/models
```

Wait until this returns a model list containing `gpt-oss-20b`. Then check that the server can generate a response using the judge's decoding settings:

```bash
curl --fail --silent --show-error http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Return only the JSON object {\"ready\": true}."}],
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens": 5120,
    "seed": 42,
    "reasoning_effort": "low",
    "stream": false
  }'
```

The response should contain a nonempty `choices[0].message.content` with the requested JSON and a `finish_reason` of `stop`. Report and red-team evaluators accept plain JSON or JSON enclosed in one complete Markdown code fence; required fields and allowed labels are validated identically. The server may also return a separate reasoning field. Once this works, leave terminal 1 running and continue in terminal 2.

### 5. Check the NeuroBench configuration

The bundled [default_config.json](../neurobench/llm/default_config.json) already matches the server we set up:

| Setting | Default |
|---|---|
| Endpoint | `http://localhost:8000/v1` |
| Served model | `gpt-oss-20b` |
| Reasoning effort | `low` |
| Temperature / top-p | `0.0` / `0.95` |
| Maximum output tokens | `5120` |
| Base seed | `42` |

Report-generation and red-team evaluations share these settings. The file also controls request concurrency, timeout, and retry budgets. If a single GPU cannot keep up and requests time out, lower `execution.max_concurrent` or increase `execution.timeout_seconds` in a separate configuration file, supplied with `--llm_config_path PATH`.

### 6. Prepare your CSV and run an evaluation

Follow [Prepare responses](../README.md#3-prepare-responses), then use the complete command in the [report-generation](evaluation_guides/report_generation.md#run), [fabrication-resistance](evaluation_guides/fabrication_resistance.md#run), [misdiagnosis-resistance](evaluation_guides/misdiagnosis_resistance.md#run), or [mislocalization-resistance](evaluation_guides/mislocalization_resistance.md#run) guide. These examples assume the server above is running at `localhost:8000` and use the bundled judge configuration.

If you want to use a different LLM as judge, use `--llm_config_path PATH` with settings supported by that model and server. For an endpoint that requires authentication, `--api_key_path PATH` reads a token from a text file and sends it as a Bearer token. Please note that this pipeline has only been tested with gpt-oss-20b, so quality of the scores cannot be guaranteed if using a different LLM.

## Check completion

The progress bar counts judge requests, with separate successful and failed totals, elapsed time, and an estimated remaining time once requests finish. Retries do not count as additional requests; a failed request advances the bar only after its attempts are exhausted.

Every evaluation requires a new or empty output directory. If `--output_dir` already contains any files or subdirectories, evaluation stops with an error and leaves those contents unchanged. Interrupted or failed runs cannot be resumed; rerun the full evaluation in a fresh directory. Retries within a run preserve each vote's seed and decoding settings.

Alongside the scores and figure, each run saves these records:

| File or directory | Purpose |
|---|---|
| `config.json` | Effective settings, completion status, and coverage counts |
| `judge_votes.csv` | One row per planned judgment, with common [audit fields and statuses](evaluation_guides/auditing.md#judge-votes) and task-specific parsed fields |
| `evaluation_issues.csv` | Missing responses, judge failures, and every affected exclusion, with explicit actions |
| `llm_job/` | Saved requests, responses, and attempts |

Check coverage alongside the scores. Report generation and red-team summaries retain only complete, matched questions or studies. Missing responses and exhausted judge failures exclude the corresponding question or study across all compared models within that task. Figures display retained/total counts, and a terminal warning lists issue counts. See [evaluation issues](evaluation_guides/auditing.md) and the individual scoring guides for details.

Exit codes are `0` when no judge failures remain, `1` for exhausted judge failures, `2` for invalid inputs/configuration, and `130` for interruption. Exit code `0` does not establish that every model supplied a response; missing responses and their exclusions are recorded in `evaluation_issues.csv`.

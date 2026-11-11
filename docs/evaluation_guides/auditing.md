# Evaluation auditing

Every evaluator writes `evaluation_issues.csv` to record problems and their effect on scoring.

## Missing responses and exclusions

| Evaluation | Handling |
|---|---|
| Closed-ended | Blank or unparseable responses count as **incorrect** and remain in the summary |
| Classification | Incomplete or mismatched prediction CSVs **abort** evaluation |
| Report generation | A missing report or failed judgment **excludes the question across every model pair**. Questions are identified by `--id_columns`, which identify studies by default |
| Red-team | A missing response or failed judgment **excludes the study across all models and conditions within the selected task** |

Judge failures cause exclusion only after retries are exhausted. Red-team exclusions never carry over to another task, including presumptive versus suggestive prompts within the same family.

For classification, labels without the reference examples needed for a metric have blank scores and are omitted from macro averages. Their counts appear in `summary.csv` and an informational note, without issue rows or warnings. Successful classification runs have a header-only `evaluation_issues.csv`.

## Evaluation issues

Each affected response (closed-ended/red-team) or model-pair comparison (report generation) has an issue row, including successful results excluded alongside a missing or failed result.

| Fields | Meaning |
|---|---|
| `task`, `input_row_index`, `identifiers` | Task, one-based input data-row index, and identifying columns as JSON |
| `model`, `candidate_1_model`, `candidate_2_model` | Affected model or compared pair |
| `reason`, `details`, `request_ids` | Cause, explanation, and failed judge request IDs where applicable |
| `action`, `included_in_summary`, `scope` | Scoring action, summary inclusion, and extent of the exclusion |

Terminal warnings count affected responses or comparisons by issue type, not unique studies or retry attempts.

## Judge votes

Report-generation and red-team evaluations write `judge_votes.csv`, with one row per planned judge request, including failed or missing replies. Blank responses generate no judge requests and are recorded in `results.csv` and `evaluation_issues.csv`. The vote table records each request's final outcome. Retry attempts remain in `llm_job/attempts.jsonl`.

| Common columns | Meaning |
|---|---|
| `task`, `input_row_index` | Task and one-based input data-row index |
| `request_id` | Links to `id` in `llm_job/requests.jsonl` and `responses.jsonl`, and `request_id` in `attempts.jsonl` |
| `vote`, `seed` | One-based repetition index and requested seed (blank when unconfigured) |
| `status`, `error` | `success`, `failed_api_request`, `failed_validation`, or `missing_response`, with failure details |
| `raw_response` | Unmodified judge response text when available |

Report votes also identify the comparison, A/B model assignments, `winner`, `winner_model`, and `rationale`. Red-team votes identify the study, prompt condition `label`, model, response, component, and parsed rubric fields. The judge's category is named `judge_label` to distinguish it from the prompt condition.

Failed votes retain any raw response but leave parsed rubric fields blank. Successful votes can still be excluded from summaries. Check `pairwise.csv` for report inclusion and `results.csv` for red-team inclusion.

Both pipelines use `success`, `missing_model_response`, `failed_api_request`, or `failed_judge_response` in `results.csv`'s `evaluation_status`. Invalid or absent replies produce `failed_judge_response`. API failures take precedence when both occur. The vote table preserves individual failure statuses. The status in `config.json` describes the whole run.
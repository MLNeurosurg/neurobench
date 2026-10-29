# Closed-ended VQA evaluation

## Overview

This evaluator scores multiple-choice and yes/no answers against the CSV's reference answers. The primary metric is accuracy (0 to 1).

## Inputs

Paths below are relative to `visual_question_answering/closed_ended/`. The 11 CSVs contain 28,105 questions across 8,225 distinct studies; study counts are deduplicated across each cohort's files.

| Cohort | Question CSVs | Studies | Questions |
|---|---|---:|---:|
| MR-RATE | `mr_rate/*.csv` | 1,276 | 2,803 |
| CTRG-Brain-263K | `ctrg_brain_263k/*.csv` | 1,383 | 3,288 |
| CQ500 | `cq500/diagnose.csv` | 488 | 5,368 |
| RSNA | `rsna/diagnose.csv` | 2,124 | 10,620 |
| OpenBrain | `open_brain/*.csv` | 2,954 | 6,026 |

For MR-RATE, CTRG-Brain-263K, and OpenBrain, the files are `diagnose.csv`, `describe.csv`, and `localize.csv`.

Keep the original question CSV columns and add your model-response columns. Required columns are `question`, `answer`, `dataset`, and the candidate columns supplied in the command. MCQ questions must list their options on separate lines, such as `A. ...` and `B. ...`.

Use `--question_column` or `--reference_answer_column` for alternative column names. By default, summaries are separate for each model and dataset. `--group_by dataset label` additionally separates question labels when that column exists. Each CSV row has equal weight; studies with more questions contribute more rows. No combined score across different tasks or datasets is produced.

## Run

After [installing NeuroBench and preparing responses](../../README.md), run:

```bash
neurobench evaluate \
  --task closed_ended \
  --csv /path/to/diagnose_with_outputs.csv \
  --candidate_answer model_output "My Model" \
  --output_dir results/diagnose
```

Repeat `--candidate_answer COLUMN MODEL` for additional models. Open `summary.csv` for scores and `results.csv` to inspect extracted answers.

## Scoring

Accuracy is the number of correct extracted answers divided by **all input rows** in the group. *Blank and unparseable responses count as incorrect.* Parse coverage is the proportion of rows with a recognized answer, whether correct or incorrect.

### Answer extraction

The parser recognizes bare option letters, parenthesized letters, leading option markers, and explicit statements such as `The answer is B.` Simple Markdown formatting and serialized response cells are supported. Complete `<think>...</think>` blocks are removed before parsing; an unfinished block is unparseable.

| Model response | Extracted answer |
|---|---|
| `B`, `(B)`, `**B**`, `B. Infarction` | B |
| `The correct answer is B.` | B |
| `Based on the imaging, the answer is C.` | C |
| `A focal lesion is present.` | Unparseable |
| `A or B` | Unparseable |
| `Yes, hemorrhage is present.` | Yes |
| `No evidence of hemorrhage.` | No, for a yes/no question |
| `Cannot determine.` | Unparseable |

Only options present in the question are accepted. Conflicting recognized answer declarations are unparseable. The parser does not infer a diagnosis from free text, search for matching option descriptions, randomly guess an answer, or call an LLM.

## Outputs

| File | Contents |
|---|---|
| `summary.csv` | Accuracy, parse coverage, and counts of correct, parsed, blank, and unparseable responses by model and grouping columns |
| `results.csv` | Original columns plus model name, normalized response, extracted/reference answers, parsing status, and binary correctness |
| `evaluation_issues.csv` | Information on missing/unparseable responses; see the [evaluation auditing guide](auditing.md) for details |
| `config.json` | Code version, input path, parser policy, grouping, and candidate columns |

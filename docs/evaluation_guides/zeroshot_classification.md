# Zero-shot classification evaluation

## Overview

The NeuroBench classification evaluation is focused on **study-level radiologic diagnoses**. OpenBrain is a single-label classification task across six classes, with the primary metric being six-class recall. All other datasets are multi-label classification tasks evaluated with AUROC.

This benchmark was designed for zero-shot classification specifically, although you should feel free to use the labels in whatever way you like. See below for guidelines on our CLIP zero-shot classification protocol.

## Inputs

Download the reference CSVs with `neurobench download --output_dir neurobench_data`. They are under `neurobench_data/zeroshot_classification/` with the default download directory. See the [data guide](../data.md) for instructions on how to download the associated images.

Paths below are relative to `zeroshot_classification/`. Each row in a reference CSV represents one study.

| Cohort | Reference CSV | Studies | Scored labels |
|---|---|---:|---:|
| MR-RATE | `mr_rate/mr_rate_test_74.csv` | 5,550 | 74 |
| CTRG-Brain-263K | `ctrg_brain_263k/ctrg_brain_263k_abnormal_82.csv` | 6,001 | 82 |
| CQ500 | `cq500/cq500.csv` | 488 | 11 |
| RSNA | `rsna/rsna.csv` | 21,744 | 6 |
| OpenBrain | `open_brain/open_brain.csv` | 6,336 | 6 (single-label) |
| OpenBrain-Large | `open_brain/open_brain_large.csv` | 2,954 | 6 (single-label) |

The 74-label MRI ontology used for MR-RATE and the 82-label CT ontology used for CTRG-Brain-263K are based on the diagnostic ontologies defined in [*Health system learning enables generalist neuroimaging models*](https://www.nature.com/articles/s41591-026-04497-1).

The reference CSV contains a column for each label. Keep this reference CSV unchanged, and prepare a separate prediction CSV per model you want to evaluate, preserving the identifier columns and using the scored diagnosis column names for model scores. The identifiers are `dataset`, `patient`, and `study` for OpenBrain, and `study` for the other cohorts. Supply one row per reference study and one finite score per scored diagnosis. Scores may be logits, probabilities, or similarity scores. Higher scores must indicate stronger evidence of a diagnosis. Match rows by identifiers, rather than row order.

## Run

After [installing NeuroBench](../../README.md), run:

```bash
neurobench evaluate \
  --task classification \
  --dataset mr_rate \
  --csv neurobench_data/zeroshot_classification/mr_rate/mr_rate_test_74.csv \
  --candidate_csv /path/to/predictions.csv "My Model" \
  --output_dir results/classification/mr_rate
```

Repeat `--candidate_csv CSV MODEL` to evaluate additional models. `--dataset` selects the scoring protocol: `mr_rate`, `ctrg_brain_263k`, `cq500`, `rsna`, `open_brain`, or `open_brain_large`. Each run evaluates one reference CSV, with any number of models and arbitrary model names. Use a new or empty output directory for every run.

Invalid or incomplete prediction CSVs stop evaluation with an error. See the [evaluation auditing guide](auditing.md) for details.

## Scoring

**MR-RATE, CTRG, CQ500, and RSNA:** the evaluator computes AUROC separately for each diagnosis, then takes the unweighted mean across evaluable diagnoses. For MR-RATE and CTRG, it additionally reports a macro average over evaluable diagnoses with at least 32 positive cases. It uses the submitted scores directly, without thresholding or applying softmax.

**OpenBrain:** the evaluator selects the highest-scoring of `stroke`, `glioma`, `metastasis`, `meningioma`, `schwannoma`, and `multiple sclerosis`, in that order. Ties choose the first class. It reports recall per pathology and its unweighted mean (**balanced accuracy**), plus accuracy per source dataset and its unweighted mean. OpenBrain-Large is a subset of OpenBrain and is evaluated in a separate run.

> [!NOTE]
> Because AUROC requires both positive and negative reference cases, diagnoses lacking either are excluded from macro averages, with counts reported in `summary.csv`. There are **4 MR-RATE** and **37 CTRG-Brain-263K** diagnoses that have no positive cases, leaving **70/74** and **45/82** diagnoses with an available AUROC, respectively.

## Outputs

| File | Contents |
|---|---|
| `summary.csv` | Per-label and aggregate metrics, group sizes, coverage, and evaluable-label counts. OpenBrain also has rows per source dataset |
| `results.csv` | One row per reference study/model, with identifiers, `reference_<label>` and `score_<label>` columns. OpenBrain adds `reference_label`, `predicted_label`, and `correct` |
| `evaluation_issues.csv` | Input-validation errors. Successful runs contain only the header |
| `config.json` | Code version, dataset, input paths, model names, identifier columns, scored labels, and scoring settings |

Summary metrics are `auroc`, `macro_auroc`, and, for MR-RATE/CTRG, `macro_auroc_min_32_positive`. OpenBrain uses `recall`, `balanced_accuracy`, `accuracy` per `source_dataset`, and `macro_source_accuracy`. `n_studies` is the number of studies in that row's group. Per-pathology OpenBrain recall uses only studies positive for that pathology. `n_labels` and `n_evaluable_labels` identify each label average's denominator.

## Zero-shot classification guidelines

The following is a reference protocol for CLIP-style zero-shot classification on our benchmarks.

1. **Prepare the study.** Use all available scans with preprocessing appropriate for your model. Obtain one study-level representation through whatever aggregation you prefer (e.g. aggregating individual slice and scan embeddings).
2. **Describe each diagnosis.** Map each scored CSV label to a diagnostic text description, expanding abbreviations where appropriate (for example, `ICH` to `intracranial hemorrhage`). Apply the 18 templates below to each description, keeping the descriptions and templates fixed across studies.
3. **Build the text representations.** Encode each prompt, L2-normalize its embedding, average the 18 embeddings for each diagnosis, and L2-normalize that average. This produces one text representation per diagnosis.
4. **Compute the scores.** Take the dot product of the study representation with each diagnosis representation. For an L2-normalized study representation, this is cosine similarity. Create a copy of the reference label CSV and replace the ground truth labels with your model scores.

<details>
<summary>18 prompt templates</summary>

Replace `{diagnosis}` with the diagnosis description.

```text
This study shows: {diagnosis}.
This study shows: {diagnosis} identified.
This study shows: {diagnosis} noted.
This study shows: {diagnosis} seen.
This study shows: new {diagnosis}.
This study shows: known {diagnosis}.
This study shows: prominent {diagnosis}.
This study shows: likely {diagnosis}.
This study shows: possibly {diagnosis}.
This study shows: indicating {diagnosis}.
This study shows: reflecting {diagnosis}.
This study shows: representing {diagnosis}.
This study shows: suggesting {diagnosis}.
This study shows: indicative of {diagnosis}.
This study shows: suggestive of {diagnosis}.
This study shows: related to {diagnosis}.
This study shows: consistent with {diagnosis}.
This study shows: compatible with {diagnosis}.
```

</details>

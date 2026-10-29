# How to download NeuroBench

You can download NeuroBench directly from [Hugging Face](https://huggingface.co/datasets/mlinslab/neurobench) without installing the evaluator, or use the NeuroBench CLI described here to select datasets and record the dataset revision and file hashes.

## Install

Follow [Install NeuroBench](../README.md#1-install), then run the commands below in that environment. `python -m neurobench` is equivalent to `neurobench`.

## Download CSVs

Download all question CSVs, classification references, evaluation metadata, and Neuro-VL baseline reports:

```bash
neurobench download --output_dir neurobench_data
```

The CSV-only download is enough to inspect the benchmark or score predictions you have already generated. Generating new predictions also requires the relevant images.

Every download command includes the complete CSV collection. Save model responses and other edits in separate working copies so that the downloaded originals can be updated safely.

## Download images

Images are available both as NIfTI files and as preprocessed PyTorch tensors compatible with [Neuro-VL](https://github.com/MLNeurosurg/neuro-vl). Availability varies by dataset. See the [Neuro-VL demo script](https://github.com/MLNeurosurg/neuro-vl/blob/master/demo.py) for details on the image preprocessing workflow. Users evaluating their own models may choose their own preprocessing.

Use `--image_format tensor`, `nifti`, or `both` with `--images`. The default is `both`.

```bash
neurobench download --images mr_rate --image_format tensor --output_dir neurobench_data
neurobench download --images mr_rate --image_format nifti --output_dir neurobench_data
neurobench download --images mr_rate --image_format both --output_dir neurobench_data
```

Each command includes all benchmark CSVs and the available license files for the selected datasets.

### Selected datasets

Select one or more source datasets:

```bash
neurobench download --images mr_rate --output_dir neurobench_data
neurobench download --images mr_rate ctrg_brain_263k --output_dir neurobench_data
```

Dataset names identify source collections, such as `mr_rate` or `ctrg_brain_263k`. A CSV's `dataset` field may include a subdirectory, such as `ctrg_brain_263k/abnormal`.

## Preview a download

Use `--dry_run` with a download command to list the selected files without transferring them:

```bash
neurobench download --images all --output_dir neurobench_data --dry_run
```

## Output directory

`--output_dir` sets the download root. The default is `neurobench_data/`. Files follow the [Hugging Face repository layout](https://huggingface.co/datasets/mlinslab/neurobench), except that RSNA's batch-directory level is automatically removed in both image formats. The CLI writes `download_manifest.json` at that root. It records the HF repository, dataset commit, downloader version, hashes of managed files, and the mapping from HF source paths to local destinations.

## Update an existing download

Rerun the download command with the same output directory to synchronize managed files with the current dataset. Previously downloaded image files are also refreshed, even if the new command requests only CSVs or a different image format; omitting `--images` does not remove those images. Files previously managed by the downloader but removed from HF are removed locally when their contents are unchanged.

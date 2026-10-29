"""Provide a universal NeuroBench CLI for downloading data and evaluating supplied model responses."""

import argparse
import csv
from importlib import import_module
from pathlib import Path
import sys

from neurobench import __version__
from neurobench.data.download import REPO_ID

ROOT = Path(__file__).resolve().parent
TASK_FAMILIES = {
    'classification': 'classification',
    'closed_ended': 'closed_ended',
    'report_generation': 'report_generation',
    'fabrication_resistance/presumptive_prompt': 'fabrication_resistance',
    'fabrication_resistance/suggestive_prompt': 'fabrication_resistance',
    'misdiagnosis_resistance/presumptive_prompt': 'misdiagnosis_resistance',
    'misdiagnosis_resistance/suggestive_prompt': 'misdiagnosis_resistance',
    'mislocalization_resistance/presumptive_prompt': 'mislocalization_resistance',
}


def positive_integer(text):
    """Validate the positive repetition count used for pairwise report judgments."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError('Must be a positive integer')
    return value


def build_parser(task=None):
    """Build download and evaluation commands with options for the selected task."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('download', help='Download all CSVs and optional images',
                                   description='Download all question, reference, and Neuro-VL report CSVs. Add --images to include imaging datasets.',
                                   allow_abbrev=False)
    download.add_argument('--images', nargs='+', metavar='DATASET',
                          help='Also download images for these datasets, or all (for example: --images mr_rate ctrg_brain_263k)')
    download.add_argument('--image_format', choices=('tensor', 'nifti', 'both'),
                          help='Format for newly requested images: preprocessed tensors, NIfTI, or both (default: both; requires --images). Previously managed files are also refreshed.')
    download.add_argument('--output_dir', type=Path, default=Path('neurobench_data'))
    download.add_argument('--repo_id', default=REPO_ID)
    download.add_argument('--dry_run', action='store_true', help='List selected files without downloading them')
    evaluate = commands.add_parser('evaluate', help='Score a CSV containing model predictions', allow_abbrev=False)
    evaluate.add_argument('--task', choices=tuple(TASK_FAMILIES), required=True)
    evaluate.add_argument('--csv', type=Path, required=True)
    if task != 'classification':
        evaluate.add_argument('--candidate_answer', nargs=2, action='append', required=task is not None,
                              metavar=('COLUMN', 'MODEL'), help='Response column and model name; repeat to compare models')
    evaluate.add_argument('--output_dir', type=Path, required=True)
    if task not in ('closed_ended', 'classification'):
        evaluate.add_argument('--llm_config_path', type=Path, default=ROOT / 'llm/default_config.json')
        evaluate.add_argument('--api_key_path', type=Path, help='Optional authentication token file for the judge server')
    if task is None or task in ('closed_ended', 'report_generation'):
        evaluate.add_argument('--reference_answer_column', default='answer')
    if task is None or task == 'closed_ended':
        group = evaluate.add_argument_group('closed-ended options')
        group.add_argument('--question_column', default='question')
        group.add_argument('--group_by', nargs='+', default=['dataset'], help='Columns defining separate summary groups')
    if task is None or task == 'classification':
        from neurobench.evaluation.classification.runner import DATASETS
        group = evaluate.add_argument_group('classification options')
        group.add_argument('--dataset', choices=tuple(DATASETS), required=task == 'classification',
                           help='Dataset defining the classification labels and scoring protocol')
        group.add_argument('--candidate_csv', nargs=2, action='append', required=task == 'classification',
                           metavar=('CSV', 'MODEL'), help='Prediction CSV and model name; repeat to compare models')
    if task is None or task == 'report_generation':
        group = evaluate.add_argument_group('report-generation options')
        group.add_argument('--id_columns', nargs='+', default=['dataset', 'patient', 'study'])
        group.add_argument('--prompt_path', type=Path, default=ROOT / 'evaluation/report_generation/prompts/report.txt')
        group.add_argument('--n_evals_per_pair', type=positive_integer, default=4)
    return parser


def parse_args(argv=None):
    """Select task-specific CLI options and parse the complete command."""
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument('--task')
    selected, _ = selector.parse_known_args(argv)
    parser = build_parser(selected.task)
    args = parser.parse_args(argv)
    if args.command == 'download':
        if args.image_format is not None and not args.images:
            parser.error('--image_format requires --images')
        args.image_format = args.image_format or 'both'
    if getattr(args, 'task', None) == 'classification':
        args.candidate_csv = [(Path(path), model) for path, model in args.candidate_csv]
    return args


def execute(args):
    """Run a download or evaluation command and record rejected evaluation inputs."""
    if args.command == 'download':
        from neurobench.data.download import execute as download
        return download(args)
    try:
        return evaluate(args)
    except (ValueError, OSError, csv.Error, RuntimeError) as exc:
        from neurobench.evaluation.issues import record_abort
        record_abort(args, exc)
        raise


def evaluate(args):
    """Dispatch supplied model responses to the selected task evaluator."""
    family = TASK_FAMILIES[args.task]
    if family == 'classification':
        from neurobench.evaluation.classification import runner
        return runner.execute(args)
    if family == 'closed_ended':
        from neurobench.evaluation.closed_ended import runner
        return runner.execute(args)
    if family == 'report_generation':
        from neurobench.evaluation.report_generation import runner
        return runner.execute(args)
    from neurobench.evaluation.red_team import runner
    evaluation = import_module(f'neurobench.evaluation.red_team.{family}.evaluation')
    return runner.execute(args, args.task, evaluation)


def main(argv=None):
    """Run the CLI and return its evaluation, input-error, or interruption exit status."""
    try:
        return execute(parse_args(argv))
    except (ValueError, OSError, csv.Error, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('Interrupted.', file=sys.stderr)
        return 130

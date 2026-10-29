#!/usr/bin/env python3
"""Run pairwise report evaluation and write scores, judge audits, and the summary figure."""

from __future__ import annotations

from datetime import datetime, timezone
import sys

from neurobench import __version__
from neurobench.data.io import directory_lock, require_empty_directory, save_csv, write_json
from neurobench.llm import LocalLLMClient, load_config
from neurobench.evaluation.report_generation.evaluation import VOTE_FIELDS, build_comparisons, load_inputs, validate_response
from neurobench.evaluation.report_generation.aggregation import evaluate_results
from neurobench.evaluation.report_generation.plotting import plot_comparison
from neurobench.evaluation import issues

def execute(args):
    """Evaluate supplied reports pairwise and save outcomes, coverage, judge audits, and the figure."""
    require_empty_directory(args.output_dir)
    header, rows = load_inputs(args)
    prompt = args.prompt_path.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("Prompt file is empty")
    config = load_config(args.llm_config_path)
    client = LocalLLMClient(config, args.api_key_path)
    items, pairs, requests = build_comparisons(rows, args, prompt, config)
    missing = sum(bool(item["blank_models"]) for item in items)
    if args.n_evals_per_pair % 2:
        print(
            "WARNING: odd n_evals_per_pair cannot balance A/B positions exactly within each pair",
            file=sys.stderr,
        )
    with directory_lock(args.output_dir) as output:
        require_empty_directory(output)
        manifest_path = output / 'config.json'
        manifest = {
            'code_version': __version__, 'task': 'report_generation',
            'llm_config': config, 'id_columns': args.id_columns,
            'reference_answer_column': args.reference_answer_column,
            'candidates': args.candidate_answer, 'n_evals_per_pair': args.n_evals_per_pair,
            'aggregation': 'plurality_with_decisive_tiebreak',
            'csv_path': str(args.csv.resolve()), 'prompt_path': str(args.prompt_path.resolve()),
            'llm_config_path': str(args.llm_config_path.resolve()), 'prompt_text': prompt,
            'status': 'running', 'updated_at': datetime.now(timezone.utc).isoformat(),
            'n_questions': len(items), 'n_questions_with_blank_candidates': missing,
            'n_requests': len(requests),
        }
        write_json(manifest_path, manifest)
        n_models = len(args.candidate_answer)
        comparisons_per_question = n_models * (n_models - 1) // 2
        comparison_word = "comparison" if comparisons_per_question == 1 else "comparisons"
        vote_word = "vote" if args.n_evals_per_pair == 1 else "votes"
        print(
            f"Evaluating {len(items):,} report-generation questions across {n_models:,} models.\n"
            f"{comparisons_per_question:,} {comparison_word} per question → {len(pairs):,} total comparisons.\n"
            f"{args.n_evals_per_pair:,} judge {vote_word} per comparison with two nonblank reports → {len(requests):,} total judge requests.",
            flush=True,
        )
        responses = client.run(
            requests,
            output / "llm_job",
            validator=validate_response,
        )
        results, pair_rows, summary, diagnostics, audit = evaluate_results(
            items, pairs, responses, args
        )
        result_columns = header + [
            "input_row_index",
            "comparison_id",
            "evaluation_status",
            "evaluation_error",
        ]
        pair_columns = [
            "input_row_index",
            "comparison_id",
            *args.id_columns,
            "pair_id",
            "candidate_1_model",
            "candidate_2_model",
            "n_evals",
            "n_valid",
            "n_failed",
            "candidate_1_votes",
            "candidate_2_votes",
            "tie_votes",
            "pair_outcome",
            "winner_model",
            "aggregation_reason",
            "included_in_summary",
        ]
        save_csv(output / "results.csv", results, result_columns)
        save_csv(output / "pairwise.csv", pair_rows, pair_columns)
        save_csv(output / "judge_votes.csv", audit, [*VOTE_FIELDS, *args.id_columns])
        save_csv(output / "summary.csv", summary, list(summary[0]))
        issues.save_issues(output, diagnostics)
        unit = 'studies' if args.id_columns == ['dataset', 'patient', 'study'] else 'questions'
        plot_comparison(summary, output / "results.png", unit=unit)
        failed = sum(row['evaluation_status'].startswith('failed_') for row in results)
        complete = summary[0]['n_complete']
        manifest.update(
            status="incomplete" if failed else "complete",
            updated_at=datetime.now(timezone.utc).isoformat(),
            n_complete=complete,
            n_excluded=summary[0]['n_excluded'],
            n_failed=failed,
        )
        write_json(manifest_path, manifest)
        print(
            f"Saved {output}: {complete}/{len(items)} {unit} retained; {len(items) - complete} excluded "
            f"({missing} with missing reports, {failed} with judge failures; these counts can overlap).",
            flush=True,
        )
        issues.warn_issues(output, diagnostics, 'comparison')
        return 1 if failed else 0

"""Provide common judge-vote auditing and record scoring issues and exclusions."""

from collections import Counter
import csv
import json
import sys

from neurobench.data.io import directory_lock, save_csv

VOTE_FIELDS = ('task', 'input_row_index', 'request_id', 'vote', 'seed',
               'status', 'error', 'raw_response')

FILENAME = 'evaluation_issues.csv'
FIELDS = ('task', 'input_row_index', 'identifiers', 'model',
          'candidate_1_model', 'candidate_2_model', 'reason', 'action',
          'included_in_summary', 'scope', 'request_ids', 'details')
REASONS = {
    'missing_model_response': 'missing model response',
    'unparseable_model_response': 'unparseable model response',
    'judge_failure': 'judge failure',
    'excluded_due_to_other_issue': 'excluded because another result is missing or failed',
}


def judge_response(response, parser, *parser_args):
    """Parse a final judge reply while retaining raw text and detailed failure status."""
    raw = response.get('response') or ''
    status = response.get('status', 'missing_response')
    error = response.get('error') or ''
    parsed = None
    if status == 'success':
        try:
            parsed = parser(raw, *parser_args)
        except (ValueError, TypeError) as exc:
            status, error = 'failed_validation', str(exc)
    if status != 'success' and not error:
        error = ('Request missing from client results' if status == 'missing_response'
                 else 'Unsuccessful judge request')
    return {'status': status, 'error': error, 'raw_response': raw}, parsed


def issue(task, row_index, identifiers, *, reason, action, included, scope,
          details, model='', candidate_1_model='', candidate_2_model='', request_ids=()):
    """Describe one scoring issue, its affected response or pair, and its inclusion outcome."""
    return dict(task=task, input_row_index=row_index,
                identifiers=json.dumps(identifiers, ensure_ascii=False, sort_keys=True),
                model=model, candidate_1_model=candidate_1_model,
                candidate_2_model=candidate_2_model, reason=reason, action=action,
                included_in_summary=included, scope=scope,
                request_ids='; '.join(request_ids), details=details)


def save_issues(output, records):
    """Write evaluation_issues.csv, including its header when no issues occurred."""
    save_csv(output / FILENAME, records, FIELDS)


def warn_issues(output, records, unit):
    """Print issue counts for affected responses or comparisons and point to the CSV audit."""
    if not records:
        return
    counts = Counter(row['reason'] for row in records)
    print(f'WARNING: evaluation issues ({unit} counts):', file=sys.stderr)
    for reason, count in sorted(counts.items()):
        print(f'  {REASONS.get(reason, reason)}: {count}', file=sys.stderr)
    print(f'  See {output / FILENAME} for actions and final inclusion.', file=sys.stderr)


def record_abort(args, error):
    """Record an invalid evaluation in an unused output directory without replacing run artifacts."""
    try:
        with directory_lock(args.output_dir) as output:
            if any(output.iterdir()):
                return
            save_issues(output, [issue(
                args.task, '', {'input_csv': str(args.csv)}, reason='invalid_input_or_configuration',
                action='evaluation_aborted', included=False, scope='run', details=str(error))])
        print(f'Evaluation aborted; details saved in {args.output_dir / FILENAME}.', file=sys.stderr)
    except (OSError, ValueError, RuntimeError, csv.Error):
        # The original error is still reported by the CLI if the output cannot be used safely.
        pass

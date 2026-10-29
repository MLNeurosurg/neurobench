"""Aggregate report judgments into win, tie, and loss summaries for complete questions."""

from collections import Counter
from itertools import combinations

from neurobench.evaluation.report_generation.evaluation import parse_response
from neurobench.evaluation import issues


def aggregate_votes(left: int, right: int, ties: int):
    """Resolve a report pair's model/tie vote counts and return the outcome and decision reason."""
    if ties > max(left, right):
        return "tie", "explicit_tie_strict_plurality"
    if left > right:
        return (
            "candidate_1",
            "candidate_1_decisive_tiebreak"
            if left == ties
            else "candidate_1_plurality",
        )
    if right > left:
        return (
            "candidate_2",
            "candidate_2_decisive_tiebreak"
            if right == ties
            else "candidate_2_plurality",
        )
    return "tie", "candidate_deadlock"


def evaluate_results(items, pairs, responses, args):
    """Aggregate report votes into question results, pair outcomes, summaries, issues, and vote audits.

    Include a question in summaries only when every candidate report and pair judgment is complete.
    """
    by_id = {response["id"]: response for response in responses}
    models = [model for _, model in args.candidate_answer]
    pair_rows, results, diagnostics, audit = [], [], [], []
    for pair in pairs:
        votes = Counter()
        failures = []
        for evaluation in pair["evaluations"]:
            request_id = evaluation["request_id"]
            judgment, parsed = issues.judge_response(by_id.get(request_id, {}), parse_response)
            winner_model = ''
            if parsed is not None:
                winner = parsed['winner']
                if winner == 'tie':
                    outcome = 'tie'
                else:
                    winner_model = evaluation[f'candidate_{winner.lower()}_model']
                    outcome = ('candidate_1' if winner_model == pair['candidate_1_model']
                               else 'candidate_2')
                # Count fixed candidate slots so a model called "tie" cannot collide with tied votes.
                votes[outcome] += 1
            else:
                failures.append({
                    'request_id': request_id,
                    'status': ('failed_api_request' if judgment['status'] == 'failed_api_request'
                               else 'failed_judge_response'),
                    'error': judgment['error'],
                })
            audit.append({
                'task': 'report_generation', 'input_row_index': pair['input_row_index'],
                **{field: pair[field] for field in args.id_columns},
                **{field: pair[field] for field in ('comparison_id', 'pair_id',
                                                  'candidate_1_model', 'candidate_2_model')},
                **evaluation, **judgment, 'winner_model': winner_model,
                'winner': parsed['winner'] if parsed is not None else '',
                'rationale': parsed['rationale'] if parsed is not None else '',
            })
        left, right, ties = votes['candidate_1'], votes['candidate_2'], votes['tie']
        if pair['blank_models']:
            outcome, reason = 'unresolved', 'missing_model_response'
        else:
            outcome, reason = (
                ("unresolved", "incomplete_votes")
                if failures
                else aggregate_votes(left, right, ties)
            )
        record = {key: value for key, value in pair.items()
                  if key not in ("evaluations", "blank_models")}
        record.update(
            n_evals=len(pair['evaluations']),
            n_valid=sum(votes.values()),
            n_failed=len(failures),
            candidate_1_votes=left,
            candidate_2_votes=right,
            tie_votes=ties,
            pair_outcome=outcome,
            winner_model=pair[outcome + "_model"]
            if outcome.startswith("candidate_")
            else "",
            aggregation_reason=reason,
            included_in_summary=False,
        )
        pair["result"] = record
        pair["failures"] = failures
        pair_rows.append(record)
    for item in items:
        identity = item["identity"]
        failed_pairs = [pair for pair in item["pairs"] if pair["failures"]]
        failures = [failure for pair in failed_pairs for failure in pair["failures"]]
        status, error = "success", ""
        if failures:
            status = (
                "failed_api_request"
                if any(
                    failure["status"] == "failed_api_request" for failure in failures
                )
                else "failed_judge_response"
            )
            error = " | ".join(
                f"{failure['request_id']}: {failure['status']}: {failure['error']}"
                for failure in failures
            )
        if item['blank_models']:
            if not failures:
                status = 'missing_model_response'
            blank_message = 'Blank candidate answer(s): ' + ', '.join(item['blank_models'])
            error = ' | '.join(filter(None, (error, blank_message)))
        result = {
            **item["row"],
            **identity,
            "evaluation_status": status,
            "evaluation_error": error,
        }
        if status == 'success':
            for pair in item["pairs"]:
                pair["result"]["included_in_summary"] = True
        else:
            for pair in item['pairs']:
                if pair['blank_models']:
                    reason = 'missing_model_response'
                    details = 'Blank report(s): ' + ', '.join(pair['blank_models']) + '. No win, loss, or tie assigned.'
                elif pair['failures']:
                    reason = 'judge_failure'
                    details = ' | '.join(f"{f['request_id']}: {f['status']}: {f['error']}" for f in pair['failures'])
                else:
                    reason = 'excluded_due_to_other_issue'
                    details = 'This pair was judged successfully, but another report is missing or another pair failed. ' + error
                diagnostics.append(issues.issue(
                    'report_generation', identity['input_row_index'],
                    {field: identity[field] for field in args.id_columns},
                    candidate_1_model=pair['candidate_1_model'], candidate_2_model=pair['candidate_2_model'],
                    reason=reason, action='excluded', included=False, scope='question_all_pairs',
                    request_ids=[f['request_id'] for f in pair['failures']],
                    details=details + ' Question excluded from every model pair summary.'))
        results.append(result)
    # Exclude a question from every summary pair if any report is missing or pair failed. Keep its
    # successful pair outcomes in pairwise.csv for audit.
    counts = {pair: Counter() for pair in combinations(models, 2)}
    for row in pair_rows:
        if row["included_in_summary"]:
            counts[row["candidate_1_model"], row["candidate_2_model"]][row["pair_outcome"]] += 1
    n_excluded = sum(bool(item['blank_models']) or any(pair['failures'] for pair in item['pairs']) for item in items)
    n_complete = len(items) - n_excluded
    summary_rows = []
    for (left, right), outcomes in counts.items():
        summary_rows.append({
            "candidate_1_model": left,
            "candidate_2_model": right,
            "n_questions": len(items),
            "n_complete": n_complete,
            "n_excluded": n_excluded,
            "candidate_1_wins": outcomes["candidate_1"],
            "ties": outcomes["tie"],
            "candidate_2_wins": outcomes["candidate_2"],
            "candidate_1_win_rate": outcomes["candidate_1"] / n_complete if n_complete else "",
            "tie_rate": outcomes["tie"] / n_complete if n_complete else "",
            "candidate_2_win_rate": outcomes["candidate_2"] / n_complete if n_complete else "",
        })
    return results, pair_rows, summary_rows, diagnostics, audit

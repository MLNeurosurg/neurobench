"""Run red-team evaluators and write response scores, study summaries, and judge audits."""

from __future__ import annotations

import ast
from collections import defaultdict
from datetime import datetime, timezone
import json
import statistics

from neurobench import __version__
from neurobench.data.io import (
    digest,
    directory_lock,
    normalize_answer,
    read_csv,
    require_empty_directory,
    save_csv,
    write_json,
)
from neurobench.llm import LocalLLMClient, Request, load_config
from neurobench.evaluation.red_team import plotting
from neurobench.evaluation import issues

STUDY_FIELDS = ('dataset', 'patient', 'study')
ID_FIELDS = (*STUDY_FIELDS, 'label')
BASE_FIELDS = (*STUDY_FIELDS, 'files', 'types', 'label', 'question', 'answer')
RESULT_FIELDS = ('task', 'input_row_index', 'response_id', 'model', 'candidate_answer',
                 'evaluation_status', 'evaluation_error', 'included_in_summary', 'n_valid_votes')
SUMMARY_FIELDS = ('task', 'model', 'condition', 'endpoint', 'n_studies',
                  'n_complete', 'n_excluded', 'score')
VOTE_FIELDS = (*issues.VOTE_FIELDS, *ID_FIELDS, 'model', 'response_id', 'component')


def judge_columns(evaluator):
    """Keep rubric fields distinct from audit fields, including the input condition label."""
    return tuple(f'judge_{field}' if field in VOTE_FIELDS else field for field in evaluator.JUDGE_FIELDS)


def require_votes(votes, count):
    """Require the full set of repeated judge votes before scoring a candidate response."""
    if len(votes) != count:
        raise ValueError(f'The evaluator requires exactly {count} votes per response')


def validate_matched_conditions(rows, task, conditions):
    """Require all task conditions and identical image metadata within one study."""
    if {row['label'] for row in rows} != set(conditions):
        raise ValueError(f'Study must contain every condition for {task}')
    if len({(row['files'], row['types']) for row in rows}) != 1:
        raise ValueError('Study must use identical images across conditions')


def image_list(text, field, index):
    """Parse a CSV image-metadata list and report its row and field when invalid."""
    try:
        values = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f'CSV record {index}: {field} must be a serialized list') from exc
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)):
        raise ValueError(f'CSV record {index}: {field} needs a nonempty list of strings')
    return values


def read_inputs(args, task, evaluator, spec):
    """Load red-team responses and validate study identities, conditions, and image metadata."""
    columns, models = zip(*args.candidate_answer)
    for values, name in ((columns, 'candidate columns'), (models, 'model labels')):
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError(f'Blank or duplicate {name}')
    required = {*BASE_FIELDS, *spec['metadata'], *columns}
    if set(columns) & set(BASE_FIELDS + spec['metadata']):
        raise ValueError('Candidate columns must be distinct from benchmark columns')
    header, rows = read_csv(args.csv, required)
    if set(header) & set(RESULT_FIELDS + spec['score_fields']):
        raise ValueError('Input contains reserved evaluation output columns')
    seen, cases = set(), defaultdict(list)
    for index, row in enumerate(rows, 1):
        identity = tuple(row[field] for field in ID_FIELDS)
        if any(not row[field].strip() for field in (*ID_FIELDS, 'question', *spec['metadata'])):
            raise ValueError(f'CSV record {index}: blank identifier, question, or metadata')
        if identity in seen:
            raise ValueError(f'Duplicate study/condition at record {index}: {identity}')
        seen.add(identity)
        if row['label'] not in spec['conditions']:
            raise ValueError(f'CSV record {index}: unexpected condition {row["label"]!r}')
        if len(image_list(row['files'], 'files', index)) != len(image_list(row['types'], 'types', index)):
            raise ValueError(f'CSV record {index}: files/types lengths differ')
        cases[identity[:3]].append(row)
    for identity, group in cases.items():
        try:
            evaluator.validate_case(group, task)
        except ValueError as exc:
            raise ValueError(f'Study {dict(zip(STUDY_FIELDS, identity))}: {exc}') from exc
    return header, rows


def build_requests(rows, args, task, evaluator, spec, config):
    """Build seeded judge requests for nonblank responses and retain their vote-audit identities."""
    items, requests, request_info = [], [], {}
    base_seed = config['api']['request_args'].get('seed')
    for index, row in enumerate(rows, 1):
        for column, model in args.candidate_answer:
            candidate = normalize_answer(row[column]).strip()
            response_id = digest([task, *(row[field] for field in ID_FIELDS), model])[:24]
            item = {'row': row, 'input_row_index': index, 'model': model,
                    'candidate_answer': candidate, 'response_id': response_id, 'requests': []}
            items.append(item)
            if not candidate:
                continue
            for vote in range(evaluator.N_VOTES):
                for component in evaluator.components(row):
                    request_id = f'{response_id}__{component}__vote_{vote + 1}'
                    seed = None if base_seed is None else base_seed + vote
                    prompt = spec['prompts'][evaluator.prompt_key(row, component)]
                    requests.append(Request(
                        id=request_id,
                        messages=[{'role': 'system', 'content': prompt},
                                  {'role': 'user', 'content': json.dumps(
                                      evaluator.payload(row, candidate, component, task), ensure_ascii=False)}],
                        request_args={} if seed is None else {'seed': seed}))
                    info = {'request_id': request_id, 'component': component,
                            'vote': vote + 1, 'seed': '' if seed is None else seed}
                    request_info[request_id] = info
                    item['requests'].append(info)
    return items, requests, request_info


def evaluate(items, responses, task, evaluator):
    """Return per-response scores and vote audits, leaving missing or failed judgments unscored."""
    by_id = {response['id']: response for response in responses}
    results, audit = [], []
    for item in items:
        row = item['row']
        record = {**row, **{field: item[field] for field in
                  ('input_row_index', 'model', 'candidate_answer', 'response_id')},
                  'task': task, 'evaluation_status': 'success', 'evaluation_error': '',
                  'included_in_summary': False, 'n_valid_votes': 0}
        if not item['candidate_answer']:
            record.update(evaluation_status='missing_model_response',
                          evaluation_error='Candidate response is blank; study excluded within this task')
            results.append(record)
            continue
        votes = [dict() for _ in range(evaluator.N_VOTES)]
        errors, failure_statuses = [], set()
        for info in item['requests']:
            judgment, parsed = issues.judge_response(
                by_id.get(info['request_id'], {}), evaluator.parse_response, info['component'])
            if parsed is not None:
                votes[info['vote'] - 1][info['component']] = parsed
            else:
                failure_statuses.add(judgment['status'])
                errors.append(f'{info["request_id"]}: {judgment["status"]}: {judgment["error"]}')
            audit.append({
                'task': task, 'input_row_index': item['input_row_index'],
                **{field: row[field] for field in ID_FIELDS},
                'model': item['model'], 'response_id': item['response_id'], **info,
                **judgment,
                **{column: '' if parsed is None else parsed[field]
                   for field, column in zip(evaluator.JUDGE_FIELDS, judge_columns(evaluator))},
            })
        expected = set(evaluator.components(row))
        record['n_valid_votes'] = sum(set(vote) == expected for vote in votes)
        if errors:
            record.update(evaluation_status=('failed_api_request' if 'failed_api_request' in failure_statuses
                                             else 'failed_judge_response'),
                          evaluation_error=' | '.join(errors))
        else:
            record.update(evaluator.score(row, votes))
        results.append(record)
    return results, audit


def summarize(results, args, task, spec):
    """Mark response inclusion and summarize studies complete for every condition and model."""
    cases = defaultdict(list)
    for row in results:
        cases[tuple(row[field] for field in STUDY_FIELDS)].append(row)
    complete = {key for key, rows in cases.items()
                if all(row['evaluation_status'] == 'success' for row in rows)}
    for row in results:
        row['included_in_summary'] = tuple(row[field] for field in STUDY_FIELDS) in complete
    lookup = {(tuple(row[field] for field in STUDY_FIELDS), row['model'], row['label']): row
              for row in results}
    scores, summaries = [], []
    for _, model in args.candidate_answer:
        for endpoint in spec['endpoints']:
            for condition, members in spec['groups'].items():
                values = []
                for case in sorted(complete):
                    value = statistics.mean(lookup[(case, model, member)][endpoint] for member in members)
                    values.append(value)
                    scores.append({'task': task, **dict(zip(STUDY_FIELDS, case)),
                                   'model': model, 'endpoint': endpoint,
                                   'condition': condition, 'score': value})
                summaries.append({
                    'task': task, 'model': model, 'condition': condition, 'endpoint': endpoint,
                    'n_studies': len(cases), 'n_complete': len(complete),
                    'n_excluded': len(cases) - len(complete),
                    'score': statistics.mean(values) if values else '',
                })
    return scores, summaries


def execute(args, task, evaluator):
    """Evaluate one red-team task and save response scores, study summaries, audits, and the figure."""
    require_empty_directory(args.output_dir)
    spec = evaluator.task_spec(task)
    header, rows = read_inputs(args, task, evaluator, spec)
    config = load_config(args.llm_config_path)
    items, requests, request_info = build_requests(rows, args, task, evaluator, spec, config)
    with directory_lock(args.output_dir) as output:
        require_empty_directory(output)
        manifest_path = output / 'config.json'
        manifest = {
            'code_version': __version__, 'task': task,
            'prompts': spec['prompts'], 'llm_config': config,
            'candidates': args.candidate_answer, 'n_votes': evaluator.N_VOTES,
            'groups': spec['groups'], 'endpoints': spec['endpoints'],
            'cohort_policy': evaluator.COHORT_POLICY, 'status': 'running',
            'csv_path': str(args.csv.resolve()), 'n_questions': len(rows),
            'n_requests': len(requests), 'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        write_json(manifest_path, manifest)
        print(f'{task}: {len(rows)} questions, {len(requests)} planned judge requests.',
              flush=True)

        def validate(request, text):
            """Apply the current criterion's rubric as the shared judge client's response validator."""
            evaluator.parse_response(text, request_info[request.id]['component'])

        responses = LocalLLMClient(config, args.api_key_path).run(
            requests, output / 'llm_job', validator=validate)
        results, audit = evaluate(items, responses, task, evaluator)
        scores, summary = evaluator.summarize(results, args, task, spec)
        save_csv(output / 'results.csv', results, [*header, *RESULT_FIELDS, *spec['score_fields']])
        save_csv(output / 'judge_votes.csv', audit,
                 [*VOTE_FIELDS, *judge_columns(evaluator)])
        save_csv(output / 'study_scores.csv', scores,
                 ['task', *STUDY_FIELDS, 'model', 'condition', 'endpoint', 'score'])
        save_csv(output / 'summary.csv', summary, SUMMARY_FIELDS)
        plotting.plot_summary(summary, [model for _, model in args.candidate_answer],
                              evaluator.figure_spec(task), output)
        failed_requests = defaultdict(list)
        for vote in audit:
            if vote['status'] != 'success':
                failed_requests[vote['response_id']].append(vote['request_id'])
        diagnostics = []
        for row in results:
            if row['included_in_summary']:
                continue
            reason = {'success': 'excluded_due_to_other_issue',
                      'missing_model_response': 'missing_model_response',
                      'failed_judge_response': 'judge_failure',
                      'failed_api_request': 'judge_failure'}[row['evaluation_status']]
            details = row['evaluation_error'] or 'Another condition or model has a missing response or failed judgment in this study.'
            diagnostics.append(issues.issue(
                task, row['input_row_index'], {field: row[field] for field in ID_FIELDS},
                model=row['model'], reason=reason, action='excluded', included=False,
                scope='study_within_experiment', request_ids=failed_requests[row['response_id']],
                details=details + ' Excluded across all conditions and models in this task only; no score enters the summary.'))
        issues.save_issues(output, diagnostics)
        failures = sum(row['evaluation_status'].startswith('failed_') for row in results)
        manifest.update(status='incomplete' if failures else 'complete',
                        n_complete_studies=summary[0]['n_complete'],
                        n_excluded_studies=summary[0]['n_excluded'],
                        n_failed_responses=failures,
                        n_blank_responses=sum(row['evaluation_status'] == 'missing_model_response'
                                              for row in results),
                        updated_at=datetime.now(timezone.utc).isoformat())
        write_json(manifest_path, manifest)
        print(f'Saved {output}: {summary[0]["n_complete"]}/{summary[0]["n_studies"]} studies retained; '
              f'{summary[0]["n_excluded"]} excluded.', flush=True)
        issues.warn_issues(output, diagnostics, 'response')
        return 1 if failures else 0

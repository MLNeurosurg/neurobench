"""Evaluate closed-ended responses and record accuracy and answer-parse coverage."""

from collections import defaultdict
from datetime import datetime, timezone
import re

from neurobench import __version__
from neurobench.data.io import normalize_answer, read_csv, require_empty_directory, save_csv, write_json
from neurobench.evaluation import issues

GENERATED = ('input_row_index', 'model', 'candidate_answer', 'parsed_answer',
             'reference_answer', 'parse_status', 'correct')
SUMMARY = ('model', 'n_questions', 'n_correct', 'n_parsed', 'n_blank',
           'n_unparseable', 'accuracy', 'parse_coverage')


def answer_spec(question, reference):
    """Identify the allowed yes/no or option-letter answers and validate the reference answer."""
    gold = normalize_answer(reference).strip().strip('.*_()[] ').upper()
    if gold in ('YES', 'NO'):
        return ('YES', 'NO'), gold
    options = tuple(re.findall(r'^\s*(?:\(([A-Z])\)|([A-Z])[.)])\s+', question, flags=re.M))
    choices = tuple(a or b for a, b in options)
    if len(choices) < 2 or len(set(choices)) != len(choices):
        raise ValueError('MCQ questions must list distinct options on separate lines, such as A. ... and B. ...')
    if gold not in choices:
        raise ValueError(f'Reference answer {gold!r} is not one of {choices}')
    return choices, gold


def parse_answer(value, choices):
    """Extract one unambiguous allowed answer and classify blank or unparseable responses."""
    text = normalize_answer(value).strip()
    if not text:
        return '', 'blank'
    # Reasoning blocks are not final answers; an unfinished block is not scored.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S | re.I).strip()
    if re.search(r'<think>', text, flags=re.I):
        return '', 'unparseable'
    text = re.sub(r'^```(?:\w+)?\s*\n|\n?```$', '', text).strip()
    text = text.replace('**', '').replace('__', '')
    binary = choices == ('YES', 'NO')
    token = r'yes|no' if binary else r'[A-Z]'
    explicit = rf'\b(?:final\s+answer|(?:correct\s+)?answer|(?:correct\s+)?option|(?:my\s+)?choice)\s*(?:is\s*|:\s*|=\s*)[\[(]?({token})\b[\])]?(?!\w)'
    declared = [m.group(1).upper() for m in re.finditer(explicit, text, flags=re.I)]
    leading = re.match(rf'^\s*[\[(]?({token})[\])]?(?=$|[\s.,:;!?)])', text, flags=re.I)
    candidates = declared[:]
    if leading:
        # A sentence beginning "A lesion ..." does not select option A.
        suffix = text[leading.end():]
        marked = (binary or bool(re.match(r'^(?:[\[(]|[A-Za-z][.):])', text))
                  or not suffix.strip() or suffix.startswith(('.', ')', ':', '-', '\n')))
        if marked:
            candidates.append(leading.group(1).upper())
    # Explicit alternatives such as "A or B" and "yes/no" are not single answers.
    alternatives = rf'\b({token})\b\s*(?:/|\bor\b|\band\b)\s*\b({token})\b'
    if re.search(alternatives, text, flags=re.I):
        return '', 'unparseable'
    unique = set(candidates)
    if len(unique) == 1 and unique <= set(choices):
        return next(iter(unique)), 'parsed'
    return '', 'unparseable'


def load_inputs(args):
    """Load closed-ended responses and validate reference answers, model columns, and grouping."""
    columns, models = zip(*args.candidate_answer)
    for name, values in [('candidate columns', columns), ('model names', models)]:
        if len(set(values)) != len(values) or any(not value.strip() for value in values):
            raise ValueError(f'Blank or duplicate {name}')
    metadata = {args.question_column, args.reference_answer_column, 'dataset'}
    if set(columns) & metadata or args.question_column == args.reference_answer_column:
        raise ValueError('Candidate columns must differ from question, reference, and dataset columns')
    header, rows = read_csv(args.csv, metadata | set(columns))
    if set(header) & set(GENERATED):
        raise ValueError(f'Input columns collide with output fields: {sorted(set(header) & set(GENERATED))}')
    specs = []
    for index, row in enumerate(rows, 1):
        if any(not row[key].strip() for key in [args.question_column, 'dataset']):
            raise ValueError(f'CSV record {index}: blank question or dataset value')
        try:
            specs.append(answer_spec(row[args.question_column], row[args.reference_answer_column]))
        except ValueError as exc:
            raise ValueError(f'CSV record {index}: {exc}') from exc
    return header, rows, specs


def execute(args):
    """Score every supplied response and save accuracy, parse coverage, and scoring issues."""
    require_empty_directory(args.output_dir)
    header, rows, specs = load_inputs(args)
    output = args.output_dir
    results, groups = [], defaultdict(list)
    for index, (row, (choices, gold)) in enumerate(zip(rows, specs), 1):
        for column, model in args.candidate_answer:
            parsed, status = parse_answer(row[column], choices)
            record = {**row, 'input_row_index': index, 'model': model,
                      'candidate_answer': normalize_answer(row[column]), 'parsed_answer': parsed,
                      'reference_answer': gold, 'parse_status': status, 'correct': int(parsed == gold)}
            results.append(record)
            groups[(model, row['dataset'])].append(record)
    summary = []
    for (model, dataset), records in groups.items():
        n = len(records); correct = sum(row['correct'] for row in records)
        parsed = sum(row['parse_status'] == 'parsed' for row in records)
        blank = sum(row['parse_status'] == 'blank' for row in records)
        summary.append({'model': model, 'dataset': dataset, 'n_questions': n,
                        'n_correct': correct, 'n_parsed': parsed, 'n_blank': blank,
                        'n_unparseable': n - parsed - blank, 'accuracy': correct / n, 'parse_coverage': parsed / n})
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / 'results.csv', results, [*header, *GENERATED])
    diagnostics = [issues.issue(
        'closed_ended', row['input_row_index'],
        {field: row[field] for field in ('dataset', 'patient', 'study', 'label')
         if field in row}, model=row['model'],
        reason='missing_model_response' if row['parse_status'] == 'blank' else 'unparseable_model_response',
        action='scored_incorrect', included=True, scope='response',
        details='Response is blank; counted as incorrect (score 0).'
        if row['parse_status'] == 'blank' else 'No unambiguous valid answer could be extracted; counted as incorrect (score 0).')
        for row in results if row['parse_status'] != 'parsed']
    issues.save_issues(output, diagnostics)
    save_csv(output / 'summary.csv', summary, ['model', 'dataset', *SUMMARY[1:]])
    config = {'code_version': __version__, 'task': 'closed_ended',
              'parser': 'conservative_answer_markers', 'candidates': args.candidate_answer,
              'question_column': args.question_column, 'reference_answer_column': args.reference_answer_column,
              'group_by': ['dataset'], 'denominator': 'all input questions in each group',
              'csv_path': str(args.csv.resolve()),
              'n_questions': len(rows), 'status': 'complete', 'updated_at': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'config.json', config)
    n_models = len(args.candidate_answer)
    model_word = 'model' if n_models == 1 else 'models'
    print(f'Saved {output}: {len(rows)} questions, {n_models} {model_word}.')
    for row in summary:
        print(f'{row["model"]} (dataset={row["dataset"]}): accuracy {row["accuracy"]:.1%}, parse coverage {row["parse_coverage"]:.1%}')
    issues.warn_issues(output, diagnostics, 'response')
    return 0

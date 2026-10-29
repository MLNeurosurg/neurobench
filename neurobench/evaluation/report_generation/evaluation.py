"""Validate report inputs and construct blinded comparisons for the shared judge client."""

from __future__ import annotations

import hashlib
import itertools

from neurobench.data.io import digest, normalize_answer, parse_judgment, read_csv
from neurobench.llm import Request
from neurobench.evaluation import issues

GENERATED_COLUMNS = {
    "input_row_index",
    "comparison_id",
    "evaluation_status",
    "evaluation_error",
}
PAIR_FIELDS = {
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
}


VOTE_FIELDS = (*issues.VOTE_FIELDS, 'comparison_id', 'pair_id',
               'candidate_1_model', 'candidate_2_model', 'candidate_a_model', 'candidate_b_model',
               'winner', 'winner_model', 'rationale')


def load_inputs(args):
    """Load report responses and validate unique question identifiers and candidate columns."""
    specs = args.candidate_answer
    if len(specs) < 2:
        raise ValueError(
            "Supply at least two --candidate_answer COLUMN MODEL arguments"
        )
    columns, models = zip(*specs)
    for label, values in [
        ("model names", models),
        ("candidate columns", columns),
        ("ID columns", args.id_columns),
    ]:
        if any(not value.strip() for value in values) or len(set(values)) != len(
            values
        ):
            raise ValueError(f"Blank or duplicate {label}: {values}")
    if set(columns) & set(args.id_columns) or args.reference_answer_column in {
        *columns,
        *args.id_columns,
    }:
        raise ValueError("ID, reference, and candidate columns must be distinct")
    if set(args.id_columns) & (GENERATED_COLUMNS | PAIR_FIELDS | set(VOTE_FIELDS)):
        raise ValueError(
            "An ID column collides with an evaluator output column; rename that input column"
        )
    required = {*args.id_columns, args.reference_answer_column, *columns}
    header, rows = read_csv(args.csv, required)
    if conflicts := set(header) & GENERATED_COLUMNS:
        raise ValueError(f"Input columns collide with generated outputs: {sorted(conflicts)}")
    seen = {}
    for record_index, row in enumerate(rows, 1):
        identity = tuple(row[column] for column in args.id_columns)
        if any(not value.strip() for value in identity):
            raise ValueError(
                f"CSV record {record_index}: blank ID in {dict(zip(args.id_columns, identity))}"
            )
        if identity in seen:
            raise ValueError(
                f"Duplicate IDs at CSV records {seen[identity]} and {record_index}: {dict(zip(args.id_columns, identity))}; add a question ID column to --id_columns if needed"
            )
        seen[identity] = record_index
        if not normalize_answer(row[args.reference_answer_column]).strip():
            raise ValueError(f"CSV record {record_index}, IDs {identity}: blank reference answer")
    return header, rows


def parse_response(text: str) -> dict:
    """Parse the report judge rationale and its A, B, or tie preference."""
    return parse_judgment(text, 'winner', ('A', 'B', 'tie'))


def validate_response(request: Request, text: str) -> None:
    """Apply report-rubric parsing as the shared judge client's response validator."""
    parse_response(text)


def build_comparisons(rows, args, prompt, config):
    """Build seeded report-pair requests with alternating anonymous A/B order and audit identities."""
    items, pairs, requests = [], [], []
    seed = config["api"]["request_args"].get("seed")
    for index, row in enumerate(rows, 1):
        comparison_id = digest([(column, row[column]) for column in args.id_columns])[:24]
        identity = {
            "input_row_index": index,
            "comparison_id": comparison_id,
            **{column: row[column] for column in args.id_columns},
        }
        blank = [
            model
            for column, model in args.candidate_answer
            if not normalize_answer(row[column]).strip()
        ]
        item = {"identity": identity, "row": row, "blank_models": blank, "pairs": []}
        items.append(item)
        for (column1, model1), (column2, model2) in itertools.combinations(
            args.candidate_answer, 2
        ):
            pair_id = digest(
                [
                    ("comparison_id", comparison_id),
                    ("candidate_1", model1),
                    ("candidate_2", model2),
                ]
            )[:24]
            pair = {
                **identity,
                "pair_id": pair_id,
                "candidate_1_model": model1,
                "candidate_2_model": model2,
                "blank_models": [model for model in (model1, model2) if model in blank],
                "evaluations": [],
            }
            pairs.append(pair)
            item["pairs"].append(pair)
            if pair["blank_models"]:
                continue
            payload = (
                f"{seed if seed is not None else 42}|{comparison_id}|{model1}|{model2}"
            )
            initial_swap = hashlib.sha256(payload.encode()).digest()[0] % 2 == 1
            for offset in range(args.n_evals_per_pair):
                swapped = initial_swap ^ (offset % 2 == 1)
                a_column, b_column = (
                    (column2, column1) if swapped else (column1, column2)
                )
                a_model, b_model = (model2, model1) if swapped else (model1, model2)
                request_id = f"{pair_id}__eval_{offset + 1}"
                user = f"Reference Report:\n{normalize_answer(row[args.reference_answer_column])}\n\nCandidate A:\n{normalize_answer(row[a_column])}\n\nCandidate B:\n{normalize_answer(row[b_column])}"
                requests.append(
                    Request(
                        request_id,
                        [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": user},
                        ],
                        {"seed": seed + offset} if seed is not None else {},
                    )
                )
                pair["evaluations"].append(
                    {
                        "request_id": request_id,
                        "vote": offset + 1,
                        "seed": seed + offset if seed is not None else "",
                        "candidate_a_model": a_model,
                        "candidate_b_model": b_model,
                    }
                )
    ids = [item["identity"]["comparison_id"] for item in items]
    if len(ids) != len(set(ids)) or len({request.id for request in requests}) != len(
        requests
    ):
        raise ValueError("Generated ID collision")
    return items, pairs, requests

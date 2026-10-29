"""Score study-level classification predictions and report metrics, coverage, and reference label availability."""

from collections import defaultdict
from datetime import datetime, timezone
from math import isfinite
from statistics import mean

from neurobench import __version__
from neurobench.data.io import read_csv, require_empty_directory, save_csv, write_json
from neurobench.evaluation import issues

# Fixed diagnosis columns keep scoring independent of filenames and prediction-column order.
DATASETS = {
    'mr_rate': (
        'subdural_hematoma', 'epidural_hematoma', 'brain_contusion', 'diffuse_axonal_injury',
        'cerebral_aneurysm', 'arteriovenous_malformation', 'cavernous_malformation_cavernoma',
        'venous_sinus_thrombosis', 'acute_ischemic_stroke', 'chronic_ischemic_stroke', 'moya_moya_disease',
        'small_vessel_ischemic_disease', 'lacunar_stroke', 'intracranial_hemorrhage',
        'intraparenchymal_hemorrhage', 'intraventricular_hemorrhage', 'aneurysmal_subarachnoid_hemorrhage',
        'traumatic_subarachnoid_hemorrhage', 'multiple_sclerosis', 'neuromyelitis_optica',
        'neurosarcoidosis', 'brain_abscess', 'viral_encephalitis', 'arachnoid_cyst', 'pineal_cyst',
        'epidermoid_cyst', 'colloid_cyst', 'brain_tumor', 'orbital_tumor', 'head_neck_tumor',
        'spine_tumor', 'nerve_tumor', 'spinal_degenerative_changes', 'glioma', 'high_grade_glioma',
        'low_grade_glioma', 'brain_metastasis', 'meningioma', 'schwannoma', 'lymphoma', 'pineal_tumor',
        'intraventricular_tumor', 'pediatric_posterior_fossa_tumor', 'brainstem_glioma', 'pituitary_tumor',
        'rathkes_cleft_cyst', 'craniopharyngioma', 'chordoma_chondrosarcoma', 'chiari_malformation',
        'cerebral_atrophy', 'brain_herniation', 'brain_mass_effect', 'brain_midline_shift', 'cephaloceles',
        'encephalomalacia_gliosis', 'cerebral_edema', 'cavum_septum_pellucidum',
        'dandy_walker_malformation', 'mega_cisterna_magna', 'heterotopia', 'pachygyria_lissencephaly',
        'schizencephaly', 'dysgenesis_corpus_callosum', 'ventriculomegaly', 'hydrocephalus_ex_vacuo',
        'obstructive_hydrocephalus', 'subdural_hygroma_effusions', 'intracranial_hypotension',
        'craniotomy_craniectomy', 'tumor_resection_cavity', 'postsurgical_changes', 'pneumocephalus',
        'catheter', 'spine_syrinx',
    ),
    'ctrg_brain_263k': (
        'acute ischemic stroke', 'acute subdural hematoma', 'airway obstruction', 'aneurysm clip',
        'aneurysm coil', 'aneurysmal subarachnoid hemorrhage', 'arachnoid cyst', 'arterial dissection',
        'basal ganglia calcification', 'brain contusion', 'brain herniation', 'brain lead',
        'brain mass effect', 'burr hole', 'carotid cavernous fistula', 'catheter',
        'cavernous malformation cavernoma', 'cerebral abscess', 'cerebral arteriovenous malformation',
        'cerebral atrophy', 'cerebral edema', 'cerebral foreign body', 'cerebral subdural empyema',
        'cerebral venous sinus thrombosis', 'chiari malformation', 'cholesteatoma', 'colloid cyst',
        'craniofacial injury', 'cranioplasty implant', 'craniotomy', 'dandy walker malformation',
        'diffuse axonal injury', 'displaced skull fracture', 'dysgenesis corpus callosum',
        'embolization material', 'encephalomalacia gliosis', 'epidural hematoma',
        'extra axial brain tumor', 'head neck enlarged lymph node', 'head neck tumor',
        'intra axial brain tumor', 'intracranial aneurysm', 'intracranial atherosclerosis',
        'intracranial epidermoid dermoid cyst', 'intracranial hemorrhage', 'intracranial hypotension',
        'intracranial pressure monitor', 'intraparenchymal hemorrhage', 'intraventricular hemorrhage',
        'large vessel occlusion', 'mastoid effusion', 'mega cisterna magna', 'midline shift',
        'nondisplaced skull fracture', 'obstructive hydrocephalus', 'orbital emphysema', 'orbital trauma',
        'orbital tumor', 'otitis media', 'parathyroid nodule', 'peritonsillar abscess',
        'pharyngeal abscess', 'pineal tumor', 'pituitary tumor', 'pneumocephalus',
        'posterior fossa tumors', 'postsurgical changes', 'resection cavity', 'scalp hemorrhage hematoma',
        'sinusitis', 'skull base fracture', 'skull bone tumor', 'slit ventricle',
        'small vessel ischemic disease', 'spinal degenerative changes', 'spine tumor',
        'subacute chronic subdural hematoma', 'subdural hygroma effusions', 'thyroid nodule',
        'transependymal flow', 'traumatic subarachnoid hemorrhage', 'ventriculomegaly',
    ),
    'cq500': (
        'ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH', 'BleedLocation-Left', 'BleedLocation-Right', 'Fracture',
        'MassEffect', 'MidlineShift',
    ),
    'rsna': (
        'Any', 'Epidural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Subdural',
    ),
    'open_brain': (
        'stroke', 'glioma', 'metastasis', 'meningioma', 'schwannoma', 'multiple sclerosis',
    ),
}
DATASETS['open_brain_large'] = DATASETS['open_brain']
OPENBRAIN = ('open_brain', 'open_brain_large')
COMMON_LABEL_DATASETS = ('mr_rate', 'ctrg_brain_263k')
SUMMARY = ('model', 'dataset', 'metric', 'label', 'source_dataset', 'score',
           'n_studies', 'n_complete', 'n_excluded', 'coverage', 'n_positive',
           'n_negative', 'n_correct', 'n_labels', 'n_evaluable_labels')


def load_table(path, id_columns, labels, reference=False):
    """Read unique study identifiers and validate binary references or finite prediction scores."""
    _, rows = read_csv(path, (*id_columns, *labels))
    table = {}
    for index, row in enumerate(rows, 1):
        key = tuple(row[column].strip() for column in id_columns)
        if any(not value for value in key):
            raise ValueError(f'{path}: CSV record {index}: blank study identifier')
        if key in table:
            raise ValueError(f'{path}: CSV record {index}: duplicate study identifier')
        values = []
        for label in labels:
            try:
                value = float(row[label])
            except ValueError as exc:
                raise ValueError(f'{path}: CSV record {index}: {label!r} must contain a numerical value') from exc
            if not isfinite(value):
                raise ValueError(f'{path}: CSV record {index}: {label!r} must contain a finite value')
            if reference and value not in (0, 1):
                raise ValueError(f'{path}: CSV record {index}: reference {label!r} must be 0 or 1')
            values.append(int(value) if reference else value)
        table[key] = tuple(values)
    return table


def load_inputs(args):
    """Load reference labels and align complete model score tables using the selected dataset's identifiers."""
    labels = DATASETS[args.dataset]
    id_columns = ('dataset', 'patient', 'study') if args.dataset in OPENBRAIN else ('study',)
    names = [model for _, model in args.candidate_csv]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError('Blank or duplicate model names')
    reference = load_table(args.csv, id_columns, labels, reference=True)
    if args.dataset in OPENBRAIN:
        for index, values in enumerate(reference.values(), 1):
            if sum(values) != 1:
                raise ValueError(f'CSV record {index}: OpenBrain requires exactly one positive reference label')
    candidates = []
    for path, model in args.candidate_csv:
        if path.resolve() == args.csv.resolve():
            raise ValueError('Candidate predictions must use a separate CSV from the reference labels')
        predictions = load_table(path, id_columns, labels)
        missing = reference.keys() - predictions.keys()
        extra = predictions.keys() - reference.keys()
        if missing or extra:
            raise ValueError(f'{model}: prediction identifiers differ from the reference CSV '
                             f'({len(missing)} missing, {len(extra)} unexpected studies)')
        candidates.append((model, predictions))
    return id_columns, labels, reference, candidates


def summary_record(dataset, model, metric, score, n_studies, **details):
    """Describe a classification metric and its group size, complete coverage, and label counts."""
    return {'model': model, 'dataset': dataset, 'metric': metric, 'score': score,
            'n_studies': n_studies, 'n_complete': n_studies, 'n_excluded': 0,
            'coverage': 1.0 if n_studies else None, **details}


def score_multilabel(dataset, model, labels, reference, predictions):
    """Compute per-diagnosis AUROC and the dataset's unweighted label summaries without transforming scores."""
    from sklearn.metrics import roc_auc_score

    records, aucs, positives = [], [], []
    n = len(reference)
    for index, label in enumerate(labels):
        truth = [values[index] for values in reference.values()]
        n_positive = sum(truth)
        n_negative = n - n_positive
        score = None
        if n_positive and n_negative:
            scores = [predictions[key][index] for key in reference]
            score = float(roc_auc_score(truth, scores))
        records.append(summary_record(dataset, model, 'auroc', score, n,
                                      label=label, n_positive=n_positive, n_negative=n_negative,
                                      n_labels=1, n_evaluable_labels=int(score is not None)))
        aucs.append(score)
        positives.append(n_positive)
    valid = [score for score in aucs if score is not None]
    records.append(summary_record(dataset, model, 'macro_auroc', mean(valid) if valid else None, n,
                                  n_labels=len(labels), n_evaluable_labels=len(valid)))
    if dataset in COMMON_LABEL_DATASETS:
        common = [score for score, count in zip(aucs, positives) if count >= 32]
        valid_common = [score for score in common if score is not None]
        records.append(summary_record(
            dataset, model, 'macro_auroc_min_32_positive', mean(valid_common) if valid_common else None, n,
            n_labels=len(common), n_evaluable_labels=len(valid_common)))
    return records, {}


def score_openbrain(dataset, model, labels, reference, predictions):
    """Compute pathology recall and source-dataset accuracy from six-class predictions with stable tie handling."""
    outcomes, by_label, by_source = {}, defaultdict(list), defaultdict(list)
    for key, truth in reference.items():
        expected = labels[truth.index(1)]
        # max selects the first declared class when scores tie.
        predicted = labels[max(range(len(labels)), key=predictions[key].__getitem__)]
        correct = int(expected == predicted)
        outcomes[key] = (expected, predicted, correct)
        by_label[expected].append(correct)
        by_source[key[0]].append(correct)
    records, recalls = [], []
    for label in labels:
        values = by_label[label]
        score = mean(values) if values else None
        if score is not None:
            recalls.append(score)
        records.append(summary_record(dataset, model, 'recall', score, len(values), label=label,
                                      n_positive=len(values), n_negative=0, n_correct=sum(values),
                                      n_labels=1, n_evaluable_labels=int(score is not None)))
    records.append(summary_record(dataset, model, 'balanced_accuracy', mean(recalls), len(reference),
                                  n_labels=len(labels), n_evaluable_labels=len(recalls)))
    source_scores = []
    for source, values in sorted(by_source.items()):
        score = mean(values)
        source_scores.append(score)
        records.append(summary_record(dataset, model, 'accuracy', score, len(values),
                                      source_dataset=source, n_correct=sum(values)))
    records.append(summary_record(dataset, model, 'macro_source_accuracy', mean(source_scores), len(reference)))
    return records, outcomes


def result_records(id_columns, labels, reference, candidates, outcomes):
    """Yield matched reference labels and model scores once per input study and candidate model."""
    for index, (key, truth) in enumerate(reference.items(), 1):
        for model, predictions in candidates:
            record = {'input_row_index': index, **dict(zip(id_columns, key)), 'model': model}
            record.update((f'reference_{label}', value) for label, value in zip(labels, truth))
            record.update((f'score_{label}', value) for label, value in zip(labels, predictions[key]))
            if outcomes[model]:
                expected, predicted, correct = outcomes[model][key]
                record.update(reference_label=expected, predicted_label=predicted, correct=correct)
            yield record


def execute(args):
    """Validate and score supplied classification CSVs and save metrics, matched results, and run settings."""
    require_empty_directory(args.output_dir)
    id_columns, labels, reference, candidates = load_inputs(args)
    summary, outcomes = [], {}
    scorer = score_openbrain if args.dataset in OPENBRAIN else score_multilabel
    for model, predictions in candidates:
        records, model_outcomes = scorer(args.dataset, model, labels, reference, predictions)
        summary.extend(records)
        outcomes[model] = model_outcomes
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    columns = ['input_row_index', *id_columns, 'model',
               *(f'reference_{label}' for label in labels), *(f'score_{label}' for label in labels)]
    if args.dataset in OPENBRAIN:
        columns.extend(('reference_label', 'predicted_label', 'correct'))
    save_csv(output / 'results.csv', result_records(id_columns, labels, reference, candidates, outcomes), columns)
    save_csv(output / 'summary.csv', summary, SUMMARY)
    issues.save_issues(output, [])
    config = {'code_version': __version__, 'task': 'classification', 'dataset': args.dataset,
              'csv_path': str(args.csv.resolve()),
              'candidates': [[str(path.resolve()), model] for path, model in args.candidate_csv],
              'id_columns': list(id_columns), 'labels': list(labels), 'score_transform': 'none',
              'metrics': sorted({row['metric'] for row in summary}),
              'coverage_policy': 'require_all_reference_studies_and_scored_labels',
              'n_studies': len(reference), 'status': 'complete',
              'updated_at': datetime.now(timezone.utc).isoformat()}
    if args.dataset in COMMON_LABEL_DATASETS:
        config['macro_positive_case_threshold'] = 32
    if args.dataset in OPENBRAIN:
        config['tie_break'] = 'first_declared_class'
    write_json(output / 'config.json', config)
    print(f'Saved {output}: {len(reference)} studies, {len(candidates)} models; prediction coverage 100%.')
    label_metric = 'recall' if args.dataset in OPENBRAIN else 'auroc'
    unavailable = [row for row in summary if row['model'] == candidates[0][0]
                   and row['metric'] == label_metric and row['score'] is None]
    if unavailable:
        metric_name = 'recall' if args.dataset in OPENBRAIN else 'AUROC'
        dataset_name = {'mr_rate': 'MR-RATE', 'ctrg_brain_263k': 'CTRG-Brain-263K',
                        'open_brain': 'OpenBrain', 'open_brain_large': 'OpenBrain-Large'}.get(
                            args.dataset, args.dataset.upper())
        reasons = []
        case_fields = [('n_positive', 'positive')]
        if label_metric == 'auroc':
            case_fields.append(('n_negative', 'negative'))
        for count_field, case_type in case_fields:
            count = sum(row[count_field] == 0 for row in unavailable)
            if count:
                subject = '1 diagnosis has' if count == 1 else f'{count} diagnoses have'
                reasons.append(f'{subject} no {case_type} cases')
        print(f'INFO: {dataset_name} {metric_name} is reported for {len(labels) - len(unavailable)}/{len(labels)} diagnoses; '
              f'{" and ".join(reasons)} in the reference CSV.')
    for row in summary:
        if not row.get('label') and not row.get('source_dataset'):
            score = 'unavailable' if row['score'] is None else f'{row["score"]:.4f}'
            print(f'{row["model"]}: {row["metric"]} {score}')
    return 0

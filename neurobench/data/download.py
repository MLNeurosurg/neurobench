"""Download benchmark CSVs and selected image datasets while protecting local edits."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from neurobench import __version__
from neurobench.data.io import directory_lock, write_json

REPO_ID = 'mlinslab/neurobench'
CSV_ROOTS = ('visual_question_answering/', 'zeroshot_classification/', 'neuro_vl_assets/')
IMAGE_FORMATS = ('tensor', 'nifti')


def image_location(name):
    """Return the format and source dataset for a file under either image root."""
    parts = PurePosixPath(name).parts
    if len(parts) >= 4 and parts[0] == 'data' and parts[1] in IMAGE_FORMATS:
        return parts[1], parts[2]
    return None


def select_files(filenames, images=None, image_format='both'):
    """Select all benchmark CSVs plus the requested datasets and image formats."""
    if image_format not in (*IMAGE_FORMATS, 'both'):
        raise ValueError(f'Unknown image format: {image_format}')
    selected = {name for name in filenames if name.endswith('.csv') and name.startswith(CSV_ROOTS)}
    if not selected:
        raise ValueError('No benchmark CSVs found in the dataset')
    if images:
        available = {location[1] for name in filenames if (location := image_location(name))}
        if 'all' in images:
            if len(images) != 1:
                raise ValueError('Use --images all by itself, or list individual datasets')
            datasets = available
        else:
            unknown = set(images) - available
            if unknown:
                raise ValueError('Unknown image datasets: ' + ', '.join(sorted(unknown))
                                 + '. Available: ' + ', '.join(sorted(available)))
            datasets = set(images)
        formats = IMAGE_FORMATS if image_format == 'both' else (image_format,)
        prefixes = tuple(f'data/{format_name}/{dataset}/'
                         for format_name in formats for dataset in sorted(datasets))
        selected.update(name for name in filenames if name.startswith(prefixes)
                        and PurePosixPath(name).name != '.gitkeep')
    for name in selected:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Unsafe dataset path: {name}')
    return sorted(selected)


def describe_selection(selected, images, image_format='both', filenames=None):
    """Report selected files and missing image formats without implying full coverage."""
    csv_count = sum(name.startswith(CSV_ROOTS) for name in selected)
    print(f'{csv_count} benchmark CSVs (questions, reference answers, and Neuro-VL reports).')
    image_files = [name for name in selected if image_location(name)]
    if images or image_files:
        print(f'{len(image_files)} files from image dataset folders (including licenses).')
    if images:
        available = {location[1] for name in (filenames if filenames is not None else selected)
                     if (location := image_location(name))}
        datasets = available if images == ['all'] else set(images)
        formats = IMAGE_FORMATS if image_format == 'both' else (image_format,)
        locations = {image_location(name) for name in image_files
                     if name.endswith(('.pt', '.nii', '.nii.gz'))}
        for dataset in sorted(datasets):
            for format_name in formats:
                if (format_name, dataset) not in locations:
                    print(f'No {format_name} images are published for {dataset}; '
                          'only available metadata for that format will be downloaded.')
        if not image_files and images == ['all']:
            print('No image files are published.')


def sha256(path):
    """Hash a downloaded file to record its contents and detect subsequent local edits."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def destination_path(destination, name):
    """Resolve a managed download path and reject paths outside the output directory."""
    path = PurePosixPath(name)
    target = destination / name
    if path.is_absolute() or '..' in path.parts or not target.resolve().is_relative_to(destination):
        raise ValueError(f'Unsafe download destination: {name}')
    return target


def local_name(name):
    """Keep published paths except for RSNA's numeric batch directory level."""
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe dataset path: {name}')
    parts = path.parts
    if (len(parts) >= 5 and image_location(name) in {('tensor', 'rsna'), ('nifti', 'rsna')}
            and parts[3].startswith('batch_') and parts[3][6:].isdigit()):
        return str(PurePosixPath(*parts[:3], *parts[4:]))
    return str(path)


def plan_paths(selected, names, destination, previous):
    """Map active and previously managed sources to safe, unambiguous destinations."""
    managed = previous.get('files', {})
    selected = sorted(set(selected) | (managed.keys() & names))
    paths = {name: local_name(name) for name in selected}
    # Older manifests used the HF source path as the local path.
    previous_paths = {name: previous.get('local_paths', {}).get(name, name) for name in managed}
    for mapping in (previous_paths, paths):
        owners = {}
        for name, relative in mapping.items():
            target = destination_path(destination, relative).resolve()
            if target in owners:
                raise ValueError(f'Download path collision: {owners[target]} and {name} -> {relative}')
            owners[target] = name
    return selected, paths, previous_paths


def copy_file(source, target):
    """Install a downloaded file atomically, creating its parent directories as needed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.download-', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def sync_files(args, info, selected, destination, previous):
    """Update managed files from one dataset snapshot after checking for local edits."""
    from huggingface_hub import hf_hub_download

    names = {item.rfilename for item in info.siblings}
    previous_hashes = previous.get('files', {})
    selected, paths, previous_paths = plan_paths(selected, names, destination, previous)
    previous_by_path = {destination_path(destination, relative).resolve(): previous_hashes[name]
                        for name, relative in previous_paths.items()}
    hashes, copies, removals = {}, [], []
    for name in selected:
        target = destination_path(destination, paths[name])
        source = Path(hf_hub_download(args.repo_id, filename=name, repo_type='dataset', revision=info.sha))
        checksum = sha256(source)
        current = sha256(target) if target.exists() else None
        if current is not None and current not in (
                checksum, previous_hashes.get(name), previous_by_path.get(target.resolve())):
            raise ValueError(f'Refusing to replace a locally edited or unmanaged file: {target}; move it or choose a new --output_dir')
        hashes[name] = checksum
        if current != checksum:
            copies.append((source, target))
    active_paths = {destination_path(destination, relative).resolve() for relative in paths.values()}
    for name, relative in sorted(previous_paths.items()):
        # A renamed HF source may still own the same local file; do not delete it.
        target = destination_path(destination, relative)
        if target.resolve() in active_paths:
            continue
        if target.exists():
            if sha256(target) != previous_hashes[name]:
                raise ValueError(f'Refusing to remove a locally edited file: {target}; move it or choose a new --output_dir')
            removals.append(target)
    # Downloads and checks must all succeed before changing any dataset files.
    for source, target in copies:
        copy_file(source, target)
    for target in removals:
        target.unlink()
        # Remove empty batch directories after migration, preserving other files.
        for parent in target.parents:
            if parent == destination:
                break
            try:
                parent.rmdir()
            except OSError:
                break
    manifest = {'code_version': __version__, 'repo_id': args.repo_id,
                'revision': info.sha, 'files': hashes, 'local_paths': paths}
    write_json(destination / 'download_manifest.json', manifest)


def execute(args):
    """Download the current dataset snapshot and record its revision and file hashes."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    try:
        info = HfApi().repo_info(args.repo_id, repo_type='dataset')
        names = [item.rfilename for item in info.siblings]
        selected = select_files(names, args.images, args.image_format)
        destination = args.output_dir.resolve()
        if args.dry_run:
            preview(args, info, selected, destination)
            return 0
        with directory_lock(destination):
            previous = preview(args, info, selected, destination)
            sync_files(args, info, selected, destination, previous)
    except HfHubHTTPError as exc:
        raise RuntimeError(f'Dataset download failed: {exc}') from exc
    print(f'Saved to {destination}.')
    return 0


def preview(args, info, selected, destination):
    """Print the download selection and return the existing compatible download manifest."""
    manifest_path = destination / 'download_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous and previous.get('repo_id') != args.repo_id:
        raise ValueError('Download directory belongs to another repository; choose a new --output_dir')
    names = {item.rfilename for item in info.siblings}
    selected, paths, previous_paths = plan_paths(selected, names, destination, previous)
    print(f'{args.repo_id}: {len(selected)} files')
    describe_selection(selected, args.images, args.image_format, names)
    if args.dry_run:
        for name in selected:
            print(name if paths[name] == name else f'{name} -> {paths[name]}')
        active_paths = {destination_path(destination, relative).resolve() for relative in paths.values()}
        for name, relative in sorted(previous_paths.items()):
            if destination_path(destination, relative).resolve() not in active_paths:
                print(f'Remove previously downloaded file: {relative}')
    return previous

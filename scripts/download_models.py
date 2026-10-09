#!/usr/bin/env python3
"""Download only the three pinned public files, resume .part files, verify SHA-256."""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

CHUNK = 8 * 1024 * 1024
MARGIN = 2 * 1024**3


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or not data.get('files'):
        raise ValueError('Expected schema version 1 and one or more model files.')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', data['repo_id']):
        raise ValueError('Invalid public Hugging Face repo identifier.')
    if not re.fullmatch(r'[a-f0-9]{40}', data['revision']):
        raise ValueError('Models must be pinned to an immutable 40-character revision.')
    targets = set()
    for item in data['files']:
        for key in ('remote_path', 'target_path'):
            part = PurePosixPath(item[key])
            if part.is_absolute() or '..' in part.parts or '\\' in item[key] or not part.parts:
                raise ValueError('Unsafe model path.')
        if item['target_path'] in targets:
            raise ValueError('Duplicate model target.')
        targets.add(item['target_path'])
        if not re.fullmatch(r'[a-f0-9]{64}', item['sha256']):
            raise ValueError('Each model requires a verified metadata SHA-256 value.')
        if type(item['size_bytes']) is not int or item['size_bytes'] <= 0:
            raise ValueError('Invalid expected size.')
    if sum(x['size_bytes'] for x in data['files']) != data['total_size_bytes']:
        raise ValueError('Manifest total does not match the files.')
    return data


def safe_target(root, relative):
    target = root / relative
    # Refuse symlink escapes, including .part files, rather than writing elsewhere.
    if not target.resolve().is_relative_to(root):
        raise ValueError('Model target escapes MODEL_DIR through a symlink.')
    part = target.with_name(target.name + '.part')
    if not part.resolve().is_relative_to(root):
        raise ValueError('Partial-file target escapes MODEL_DIR through a symlink.')
    return target, part


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b''):
            result.update(chunk)
    return result.hexdigest()


def verify(path, item):
    if path.stat().st_size != item['size_bytes']:
        raise ValueError(f"Wrong size for {item['target_path']}; preserve/remove the invalid file manually before retrying.")
    if digest(path) != item['sha256']:
        raise ValueError(f"SHA-256 mismatch for {item['target_path']}; the file has not been accepted.")


def public_url(manifest, item):
    path = urllib.parse.quote(item['remote_path'], safe='/')
    return f"https://huggingface.co/{manifest['repo_id']}/resolve/{manifest['revision']}/{path}"


def stream_download(url, partial, expected_size):
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError('Partial file is larger than the manifest; inspect/remove it before retrying.')
    if offset == expected_size:
        return
    headers = {'User-Agent': 'comfyui-reusable-setup/1', 'Accept-Encoding': 'identity'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        status = response.status
        mode = 'wb'
        if status == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if not match or int(match[1]) != offset or int(match[3]) != expected_size:
                raise ValueError('Server returned an inconsistent byte range.')
            mode = 'ab' if offset else 'wb'
        elif status == 200:
            # Some CDNs ignore Range. Restart safely; do not append duplicate bytes.
            offset = 0
        else:
            raise ValueError(f'Unexpected download status {status}.')
        read_bytes, last_report = offset, time.monotonic()
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                if read_bytes + len(chunk) > expected_size:
                    raise ValueError('Download exceeded expected size.')
                handle.write(chunk)
                read_bytes += len(chunk)
                if time.monotonic() - last_report >= 10:
                    print(f'  {read_bytes / expected_size:.1%} ({read_bytes:,}/{expected_size:,} bytes)', flush=True)
                    last_report = time.monotonic()
            handle.flush()
            os.fsync(handle.fileno())
    if read_bytes != expected_size:
        raise OSError('Download ended early; the partial file will be resumed.')


def ensure_file(root, manifest, item, verify_only=False):
    target, partial = safe_target(root, item['target_path'])
    if target.exists():
        print(f"Verifying existing {item['target_path']} …", flush=True)
        verify(target, item)
        return
    if verify_only:
        raise FileNotFoundError(f"Missing model: {item['target_path']}; DOWNLOAD_MODELS=0 forbids downloading.")
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {item['target_path']} ({item['size_bytes']:,} bytes) …", flush=True)
    for attempt in range(1, 6):
        try:
            stream_download(public_url(manifest, item), partial, item['size_bytes'])
            break
        except urllib.error.HTTPError as exc:
            # Never log redirected signed URLs or full exception strings.
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 5:
                raise RuntimeError(f'Public model download failed with HTTP {exc.code}.') from None
            print(f'  HTTP {exc.code}; retry {attempt}/5 …', flush=True)
        except (OSError, urllib.error.URLError) as exc:
            if attempt == 5:
                raise RuntimeError(f'Download failed after five attempts ({type(exc).__name__}); rerun to resume.') from None
            print(f'  Interrupted ({type(exc).__name__}); retry {attempt}/5 …', flush=True)
        time.sleep(min(attempt * 3, 15))
    verify(partial, item)
    partial.replace(target)
    print(f"Verified SHA-256: {item['target_path']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    data = load_manifest(args.manifest)
    root = args.model_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    remaining = 0
    for item in data['files']:
        target, partial = safe_target(root, item['target_path'])
        if not target.exists():
            remaining += max(0, item['size_bytes'] - (partial.stat().st_size if partial.exists() else 0))
    if not args.verify_only and remaining and shutil.disk_usage(root).free < remaining + MARGIN:
        raise RuntimeError(f'Not enough disk space: need {remaining + MARGIN:,} bytes including 2 GiB safety margin.')
    for item in data['files']:
        ensure_file(root, data, item, args.verify_only)
    print('All manifest files are present and SHA-256 verified.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)

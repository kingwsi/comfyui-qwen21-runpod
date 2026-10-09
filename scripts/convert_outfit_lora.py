#!/usr/bin/env python3
"""Convert only the SHA-pinned OutfitSwap LoRA for the pinned split Qwen 2.1 GGUF.

Standard library only: BF16 bytes are copied, never decoded or requantized.
The original file is read-only. The separately named output and its JSON manifest
are published from fsynced temporary files; an incomplete/corrupt pair is refused.
This proves layout compatibility, not GPU execution or image quality.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import tempfile

SOURCE_SHA256 = '4ddfeac5695adaed1f0f78b23fda40919fd2ed6a4af790ced6c1e043dc4ae1f0'
SOURCE_BYTES = 159436488
OUTPUT_NAME = 'OutfitSwap-LoRA-GGUF-compatible.safetensors'
CONVERTER_VERSION = '1'
BLOCKS = 32
RANK = 32
HEADER_LIMIT = 1024 * 1024
CHUNK = 8 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique_object)


def tensor_contract(split=False):
    """Exact key/dtype/shape contract for this one adapter, without alpha tensors."""
    modules = {
        'attn.to_k': (4096, 4096),
        'attn.to_out.0': (4096, 4096),
        'attn.to_q': (4096, 4096),
        'attn.to_v': (4096, 4096),
        'img_mlp.out': (4096, 12288),
    }
    if split:
        modules.update({'img_mlp.gate_layer': (12288, 4096),
                        'img_mlp.proj': (12288, 4096)})
    else:
        modules['img_mlp.gate_up'] = (24576, 4096)
    result = {}
    for block in range(BLOCKS):
        for module, (rows, cols) in modules.items():
            stem = f'diffusion_model.transformer_blocks.{block}.{module}'
            result[stem + '.lora_A.weight'] = [RANK, cols]
            result[stem + '.lora_B.weight'] = [rows, RANK]
    return result


def read_safetensors(raw):
    """Strict, bounded BF16 safetensors reader; data views retain original bytes."""
    require(len(raw) >= 8, 'Truncated safetensors prefix')
    size = struct.unpack('<Q', raw[:8])[0]
    require(2 <= size <= HEADER_LIMIT and 8 + size <= len(raw),
            'Invalid/truncated safetensors header')
    header = _json(bytes(raw[8:8 + size]))
    require(isinstance(header, dict), 'Safetensors header must be an object')
    metadata = header.pop('__metadata__', {})
    require(isinstance(metadata, dict) and all(isinstance(k, str) and
            isinstance(v, str) for k, v in metadata.items()), 'Invalid metadata')
    data = memoryview(raw)[8 + size:]
    tensors, spans = {}, []
    for key, value in header.items():
        require(isinstance(value, dict) and set(value) == {'dtype', 'shape', 'data_offsets'},
                f'Invalid tensor descriptor: {key}')
        shape, offsets = value['shape'], value['data_offsets']
        require(value['dtype'] == 'BF16', f'Unexpected dtype: {key}')
        require(isinstance(shape, list) and len(shape) == 2 and
                all(type(n) is int and n > 0 for n in shape), f'Invalid shape: {key}')
        require(isinstance(offsets, list) and len(offsets) == 2 and
                all(type(n) is int for n in offsets), f'Invalid offsets: {key}')
        start, end = offsets
        require(0 <= start < end <= len(data) and end - start == math.prod(shape) * 2,
                f'Invalid tensor byte span: {key}')
        spans.append((start, end))
        tensors[key] = {'dtype': 'BF16', 'shape': shape, 'data': data[start:end]}
    cursor = 0
    for start, end in sorted(spans):
        require(start == cursor, 'Overlapping tensors or a gap in safetensors data')
        cursor = end
    require(cursor == len(data), 'Unclaimed/truncated safetensors data')
    return tensors, metadata


def validate_contract(tensors, split=False):
    expected = tensor_contract(split)
    require(set(tensors) == set(expected),
            'Unexpected tensor keys; missing, extra, alpha, or already-split adapters are refused')
    for key, shape in expected.items():
        tensor = tensors[key]
        require(tensor['dtype'] == 'BF16' and tensor['shape'] == shape and
                len(tensor['data']) == math.prod(shape) * 2, f'Unexpected tensor shape/dtype: {key}')


def read_gguf_header(path):
    """Inspect only the first MiB; the pinned GGUF tensor directory fits there."""
    with Path(path).open('rb') as handle:
        stream = io.BytesIO(handle.read(HEADER_LIMIT))

    def number(fmt):
        length = struct.calcsize('<' + fmt)
        raw = stream.read(length)
        require(len(raw) == length, 'Truncated GGUF header')
        return struct.unpack('<' + fmt, raw)[0]

    def string():
        length = number('Q')
        require(length <= HEADER_LIMIT, 'Oversized GGUF string')
        raw = stream.read(length)
        require(len(raw) == length, 'Truncated GGUF string')
        return raw.decode('utf-8')

    def value(kind, depth=0):
        require(depth < 4, 'Nested GGUF metadata is too deep')
        kinds = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if kind == 8:
            return string()
        if kind == 9:
            subtype, count = number('I'), number('Q')
            require(count <= 100000, 'Oversized GGUF metadata array')
            return [value(subtype, depth + 1) for _ in range(count)]
        require(kind in kinds, 'Unknown GGUF metadata type')
        return number(kinds[kind])

    require(stream.read(4) == b'GGUF', 'Invalid GGUF magic')
    version, count, nmeta = number('I'), number('Q'), number('Q')
    require(version == 3 and count <= 10000 and nmeta <= 10000, 'Unexpected GGUF header')
    metadata = {}
    for _ in range(nmeta):
        key = string()
        require(key not in metadata, 'Duplicate GGUF metadata key')
        metadata[key] = value(number('I'))
    tensors = {}
    for _ in range(count):
        key, ndim = string(), number('I')
        require(key not in tensors and 0 < ndim <= 8, 'Invalid GGUF tensor descriptor')
        shape = [number('Q') for _ in range(ndim)]
        require(all(shape), 'Invalid GGUF tensor shape')
        tensors[key] = {'shape': list(reversed(shape)), 'ggml_type': number('I'),
                        'offset': number('Q')}
    return {'metadata': metadata, 'tensors': tensors, 'tensor_count': count}


def validate_base(base):
    require(base['metadata'].get('general.architecture') == 'qwen_image21',
            'Wrong base architecture; expected qwen_image21')
    require(base['tensor_count'] == 297, 'Unexpected pinned base tensor count')
    tensors = base['tensors']
    for block in range(BLOCKS):
        require(f'transformer_blocks.{block}.img_mlp.gate_up.weight' not in tensors,
                'Base already has fused gate_up weights')
    expected = tensor_contract(split=True)
    for key, shape in expected.items():
        if not key.endswith('.lora_A.weight'):
            continue
        stem = key[:-len('.lora_A.weight')]
        target = stem.removeprefix('diffusion_model.') + '.weight'
        output_shape = [expected[stem + '.lora_B.weight'][0], shape[1]]
        require(target in tensors and tensors[target]['shape'] == output_shape and
                tensors[target]['ggml_type'] == 8,
                f'Missing/incompatible Q8_0 split base target: {target}')


def split_tensors(tensors):
    validate_contract(tensors)
    result = {key: value for key, value in tensors.items() if '.img_mlp.gate_up.' not in key}
    for block in range(BLOCKS):
        prefix = f'diffusion_model.transformer_blocks.{block}.img_mlp.'
        a = tensors[prefix + 'gate_up.lora_A.weight']
        b = tensors[prefix + 'gate_up.lora_B.weight']
        midpoint = 12288 * RANK * 2
        for part, start, end in [('gate_layer', 0, midpoint),
                                 ('proj', midpoint, len(b['data']))]:
            result[prefix + part + '.lora_A.weight'] = a
            result[prefix + part + '.lora_B.weight'] = {
                'dtype': 'BF16', 'shape': [12288, RANK], 'data': b['data'][start:end]}
    validate_contract(result, split=True)
    # Rank 32 stays rank 32. No source alpha exists, so ComfyUI's default scale
    # remains 1.0. B_gate @ A and B_proj @ A are the original delta's row slices
    # at every external LoRA strength, including zero and negative strengths.
    return result


def serialized_chunks(tensors, metadata):
    header, cursor = {'__metadata__': metadata}, 0
    for key in sorted(tensors):
        tensor = tensors[key]
        end = cursor + len(tensor['data'])
        header[key] = {'dtype': tensor['dtype'], 'shape': tensor['shape'],
                       'data_offsets': [cursor, end]}
        cursor = end
    encoded = json.dumps(header, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    encoded += b' ' * (-len(encoded) % 8)
    yield struct.pack('<Q', len(encoded))
    yield encoded
    for key in sorted(tensors):
        yield tensors[key]['data']


def _digest_chunks(chunks):
    digest, size = hashlib.sha256(), 0
    for chunk in chunks:
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def file_digest(path):
    with Path(path).open('rb') as handle:
        return _digest_chunks(iter(lambda: handle.read(CHUNK), b''))


def manifest_path(output):
    return Path(str(output) + '.manifest.json')


def _write_temporary(parent, chunks):
    fd, name = tempfile.mkstemp(prefix='.outfit-lora-', suffix='.tmp', dir=parent)
    path = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            for chunk in chunks:
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _verify_existing(output, sidecar, expected):
    require(output.is_file() and sidecar.is_file(),
            'Incomplete conversion: output and manifest must both exist; inspect manually')
    require(not output.is_symlink() and not sidecar.is_symlink(), 'Symlinked output/manifest refused')
    require(sidecar.stat().st_size <= 16384, 'Invalid conversion manifest size')
    actual = _json(sidecar.read_bytes())
    require(actual == expected, 'Conversion manifest mismatch; refusing existing output')
    digest, size = file_digest(output)
    require(digest == expected['output_sha256'] and size == expected['output_bytes'],
            'Existing output SHA-256/size mismatch; refusing corrupt output')


def convert(source, base_header, output):
    source, output = Path(source), Path(output)
    sidecar = manifest_path(output)
    require(output.name == OUTPUT_NAME, f'Output must be named {OUTPUT_NAME}')
    require(source.resolve() != output.resolve() and source.resolve() != sidecar.resolve(),
            'Refusing in-place conversion or source overwrite')
    require(not output.is_symlink() and not sidecar.is_symlink(), 'Symlinked output/manifest refused')
    require(source.stat().st_size == SOURCE_BYTES, 'Unexpected source size')
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    require(source_hash == SOURCE_SHA256,
            'Unexpected source SHA-256; this converter is deliberately file-specific')
    tensors, metadata = read_safetensors(raw)
    validate_contract(tensors)
    validate_base(read_gguf_header(base_header))
    converted = split_tensors(tensors)
    metadata = {key: value for key, value in metadata.items() if not key.startswith('sshs_')}
    metadata.update({
        'compatibility_conversion': 'gate_up -> gate_layer/proj; gate rows first; no rescaling',
        'compatibility_source_sha256': source_hash,
        'compatibility_tool_version': CONVERTER_VERSION,
    })
    output_hash, output_size = _digest_chunks(serialized_chunks(converted, metadata))
    manifest = {
        'schema_version': 1, 'converter_version': CONVERTER_VERSION,
        'input_sha256': source_hash, 'input_bytes': len(raw),
        'output_sha256': output_hash, 'output_bytes': output_size,
        'output_filename': OUTPUT_NAME, 'input_tensors': 384, 'output_tensors': 448,
        'output_adapters': 224, 'converted_blocks': BLOCKS,
        'mapping': 'gate_up rows [0:12288] -> gate_layer; [12288:24576] -> proj',
        'rank': RANK, 'rescaled': False,
    }
    # The expected hash is independently regenerated from the verified source on
    # EVERY run. An attacker/corruption changing both output and manifest cannot
    # make an altered tensor accepted merely by updating the sidecar's hash.
    if os.path.lexists(output) or os.path.lexists(sidecar):
        _verify_existing(output, sidecar, manifest)
        return {**manifest, 'status': 'verified-existing'}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = temporary_manifest = None
    try:
        temporary_output = _write_temporary(output.parent, serialized_chunks(converted, metadata))
        temporary_manifest = _write_temporary(output.parent, [
            (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()])
        require(file_digest(temporary_output) == (output_hash, output_size), 'Temporary output verification failed')
        # Atomic exclusive publication: never replace a file created concurrently.
        # A crash between the two links leaves an incomplete pair which is refused
        # on restart. The launcher starts ComfyUI only after this command succeeds.
        os.link(temporary_output, output)
        os.link(temporary_manifest, sidecar)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        for temporary in (temporary_output, temporary_manifest):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    _verify_existing(output, sidecar, manifest)
    return {**manifest, 'status': 'created'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--base-header', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = convert(args.source, args.base_header, args.output)
    except (OSError, ValueError, UnicodeError, struct.error) as error:
        parser.exit(1, f'OutfitSwap conversion refused: {error}\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

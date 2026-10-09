"""Dependency-free regression tests; no downloaded weights or torch required.

Synthetic tensors use the full production key/shape contract. Only the tests
patch the pinned source fingerprint; there is no runtime hash override.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('convert_outfit_lora', ROOT / 'scripts/convert_outfit_lora.py')
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def bf16(number):
    return struct.pack('<f', number)[2:]


def make_tensors():
    result, buffers = {}, {}
    for key, shape in converter.tensor_contract().items():
        dims = tuple(shape)
        if dims not in buffers:
            if shape == [24576, 32]:
                # Unequal halves catch gate/proj reversal; unequal rank values
                # catch row-vs-column splitting and unintended rescaling.
                pattern = b''.join(bf16(float(i + 1)) for i in range(32))
                buffers[dims] = pattern * 12288 + bf16(0.5) * (12288 * 32)
            else:
                pattern = b''.join(bf16(float(i % 4 + 1)) for i in range(32))
                buffers[dims] = pattern * (math.prod(shape) // 32)
        result[key] = {'dtype': 'BF16', 'shape': shape, 'data': memoryview(buffers[dims])}
    return result


def make_base():
    result = {}
    contract = converter.tensor_contract(split=True)
    for key, shape in contract.items():
        if key.endswith('.lora_A.weight'):
            stem = key[:-len('.lora_A.weight')]
            target = stem.removeprefix('diffusion_model.') + '.weight'
            result[target] = {'shape': [contract[stem + '.lora_B.weight'][0], shape[1]],
                              'ggml_type': 8, 'offset': 0}
    for index in range(297 - len(result)):
        result[f'fixture_extra_{index}'] = {'shape': [1], 'ggml_type': 0, 'offset': 0}
    return {'metadata': {'general.architecture': 'qwen_image21'},
            'tensors': result, 'tensor_count': len(result)}


def write_base(path, base):
    def string(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw
    encoded = b'GGUF' + struct.pack('<IQQ', 3, len(base['tensors']), 1)
    encoded += string('general.architecture') + struct.pack('<I', 8)
    encoded += string(base['metadata']['general.architecture'])
    for key, tensor in base['tensors'].items():
        shape = list(reversed(tensor['shape']))
        encoded += string(key) + struct.pack('<I', len(shape))
        encoded += struct.pack('<' + 'Q' * len(shape), *shape)
        encoded += struct.pack('<IQ', tensor['ggml_type'], tensor['offset'])
    path.write_bytes(encoded)


def encode_raw(header, payload):
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    return struct.pack('<Q', len(raw)) + raw + payload


class TensorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensors = make_tensors()

    def test_exact_pin_and_separate_output_name(self):
        self.assertEqual(converter.SOURCE_SHA256,
                         '4ddfeac5695adaed1f0f78b23fda40919fd2ed6a4af790ced6c1e043dc4ae1f0')
        self.assertEqual(converter.SOURCE_BYTES, 159436488)
        self.assertEqual(converter.OUTPUT_NAME, 'OutfitSwap-LoRA-GGUF-compatible.safetensors')

    def test_all_32_blocks_exact_byte_preservation(self):
        output = converter.split_tensors(self.tensors)
        self.assertEqual(len(self.tensors), 384)
        self.assertEqual(len(output), 448)
        self.assertEqual(sum(key.endswith('.lora_A.weight') for key in output), 224)
        self.assertFalse(any('gate_up' in key or 'alpha' in key for key in output))
        unchanged = 0
        for key, original in self.tensors.items():
            if '.gate_up.' not in key:
                self.assertIs(output[key], original)
                unchanged += 1
        self.assertEqual(unchanged, 320)
        for block in range(32):
            prefix = f'diffusion_model.transformer_blocks.{block}.img_mlp.'
            a = self.tensors[prefix + 'gate_up.lora_A.weight']
            b = self.tensors[prefix + 'gate_up.lora_B.weight']['data']
            gate = output[prefix + 'gate_layer.lora_B.weight']
            proj = output[prefix + 'proj.lora_B.weight']
            self.assertEqual(gate['shape'], [12288, 32])
            self.assertEqual(proj['shape'], [12288, 32])
            self.assertEqual(bytes(gate['data']) + bytes(proj['data']), bytes(b))
            self.assertIs(output[prefix + 'gate_layer.lora_A.weight'], a)
            self.assertIs(output[prefix + 'proj.lora_A.weight'], a)

    def test_delta_and_strength_preserved_without_numpy(self):
        output = converter.split_tensors(self.tensors)
        prefix = 'diffusion_model.transformer_blocks.0.img_mlp.'

        def value(tensor, row, col):
            index = (row * tensor['shape'][1] + col) * 2
            return struct.unpack('<f', b'\0\0' + bytes(tensor['data'][index:index + 2]))[0]

        def delta(a, b, row, col):
            return sum(value(b, row, rank) * value(a, rank, col) for rank in range(32))

        a = self.tensors[prefix + 'gate_up.lora_A.weight']
        b = self.tensors[prefix + 'gate_up.lora_B.weight']
        for part, offset in [('gate_layer', 0), ('proj', 12288)]:
            ca = output[prefix + part + '.lora_A.weight']
            cb = output[prefix + part + '.lora_B.weight']
            for row in [0, 6144, 12287]:
                for col in [0, 2048, 4095]:
                    for strength in [0, 0.5, 1, 1.5, -0.5]:
                        self.assertEqual(delta(a, b, offset + row, col) * strength,
                                         delta(ca, cb, row, col) * strength)

    def test_refuses_missing_extra_alpha_and_split_keys(self):
        key = next(iter(self.tensors))
        for operation in ['missing', 'extra', 'alpha', 'split']:
            with self.subTest(operation=operation):
                changed = dict(self.tensors)
                if operation == 'missing':
                    changed.pop(key)
                else:
                    changed[{'extra': 'unknown.weight', 'alpha': 'anything.alpha',
                             'split': 'diffusion_model.transformer_blocks.0.img_mlp.proj.lora_A.weight'}[operation]] = changed[key]
                with self.assertRaisesRegex(ValueError, 'Unexpected tensor keys'):
                    converter.split_tensors(changed)

    def test_refuses_wrong_shape_dtype_and_rank(self):
        key = 'diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight'
        for changes in [{'shape': [16, 8192]}, {'dtype': 'F16'}, {'data': b'bad'}]:
            with self.subTest(changes=changes):
                changed = dict(self.tensors)
                changed[key] = {**changed[key], **changes}
                with self.assertRaisesRegex(ValueError, 'Unexpected tensor shape/dtype'):
                    converter.split_tensors(changed)

    def test_base_all_224_mapping_targets_and_guards(self):
        converter.validate_base(make_base())
        for problem in ['architecture', 'count', 'fused', 'missing', 'shape', 'quantization']:
            with self.subTest(problem=problem):
                base = make_base()
                target = 'transformer_blocks.31.img_mlp.proj.weight'
                if problem == 'architecture':
                    base['metadata']['general.architecture'] = 'qwen_image'
                elif problem == 'count':
                    base['tensor_count'] = 296
                elif problem == 'fused':
                    base['tensors']['transformer_blocks.0.img_mlp.gate_up.weight'] = {}
                elif problem == 'missing':
                    base['tensors'].pop(target)
                elif problem == 'shape':
                    base['tensors'][target]['shape'] = [24576, 4096]
                else:
                    base['tensors'][target]['ggml_type'] = 0
                with self.assertRaises(ValueError):
                    converter.validate_base(base)


class FormatTests(unittest.TestCase):
    def test_small_bf16_roundtrip(self):
        tensors = {'x': {'dtype': 'BF16', 'shape': [1, 2], 'data': bf16(1) + bf16(2)}}
        raw = b''.join(converter.serialized_chunks(tensors, {'format': 'pt'}))
        actual, metadata = converter.read_safetensors(raw)
        self.assertEqual(actual['x']['data'], tensors['x']['data'])
        self.assertEqual(metadata, {'format': 'pt'})

    def test_refuses_malformed_safetensors(self):
        valid = {'x': {'dtype': 'BF16', 'shape': [1, 2], 'data_offsets': [0, 4]}}
        samples = [b'', b'bad', struct.pack('<Q', 2**63),
                   encode_raw(valid, b'\0' * 3), encode_raw(valid, b'\0' * 5),
                   encode_raw({'__metadata__': {'x': 1}, **valid}, b'\0' * 4)]
        for problem in ['overlap', 'gap', 'dtype', 'shape', 'offsets']:
            header = json.loads(json.dumps(valid))
            if problem == 'overlap':
                header['y'] = dict(header['x'])
            elif problem == 'gap':
                header['x']['data_offsets'] = [1, 5]
            elif problem == 'dtype':
                header['x']['dtype'] = 'F16'
            elif problem == 'shape':
                header['x']['shape'] = [True, 2]
            else:
                header['x']['data_offsets'] = [False, 4]
            samples.append(encode_raw(header, b'\0' * 5))
        duplicate = b'{"x":{},"x":{}}'
        samples.append(struct.pack('<Q', len(duplicate)) + duplicate)
        for raw in samples:
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                converter.read_safetensors(raw)

    def test_gguf_header_roundtrip_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.gguf'
            base = make_base()
            write_base(path, base)
            self.assertEqual(converter.read_gguf_header(path), base)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                converter.read_gguf_header(path)
            path.write_bytes(b'nope')
            with self.assertRaisesRegex(ValueError, 'magic'):
                converter.read_gguf_header(path)


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.root = Path(cls.workspace.name)
        cls.source = cls.root / 'original.safetensors'
        cls.base = cls.root / 'base.gguf'
        cls.tensors = make_tensors()
        with cls.source.open('wb') as handle:
            for chunk in converter.serialized_chunks(cls.tensors, {
                    'format': 'pt', 'name': 'synthetic fixture', 'sshs_model_hash': 'stale'}):
                handle.write(chunk)
        cls.source_hash, cls.source_size = converter.file_digest(cls.source)
        write_base(cls.base, make_base())

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def setUp(self):
        self.case = tempfile.TemporaryDirectory(dir=self.root)
        self.output = Path(self.case.name) / converter.OUTPUT_NAME
        self.sidecar = converter.manifest_path(self.output)
        self.pins = mock.patch.multiple(converter, SOURCE_SHA256=self.source_hash,
                                       SOURCE_BYTES=self.source_size)
        self.pins.start()

    def tearDown(self):
        self.pins.stop()
        self.case.cleanup()

    def convert(self):
        return converter.convert(self.source, self.base, self.output)

    def test_conversion_idempotence_and_full_corruption_checks(self):
        result = self.convert()
        self.assertEqual(result['status'], 'created')
        manifest = json.loads(self.sidecar.read_text())
        self.assertEqual(manifest['input_sha256'], self.source_hash)
        self.assertEqual(manifest['output_tensors'], 448)
        self.assertEqual(manifest['converter_version'], converter.CONVERTER_VERSION)
        self.assertEqual(converter.file_digest(self.output),
                         (manifest['output_sha256'], manifest['output_bytes']))
        tensors, metadata = converter.read_safetensors(self.output.read_bytes())
        converter.validate_contract(tensors, split=True)
        self.assertNotIn('sshs_model_hash', metadata)
        self.assertEqual(metadata['name'], 'synthetic fixture')
        self.assertEqual(metadata['compatibility_source_sha256'], self.source_hash)
        del tensors
        before = (self.output.stat().st_mtime_ns, self.sidecar.stat().st_mtime_ns)
        self.assertEqual(self.convert()['status'], 'verified-existing')
        self.assertEqual(before, (self.output.stat().st_mtime_ns, self.sidecar.stat().st_mtime_ns))
        # Corrupt a data byte while keeping size and metadata unchanged.
        with self.output.open('r+b') as handle:
            handle.seek(-1, os.SEEK_END)
            original = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([original[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, 'SHA-256/size mismatch'):
            self.convert()
        # Even updating the sidecar to match the corruption must not pass.
        changed = {**manifest, 'output_sha256': converter.file_digest(self.output)[0]}
        self.sidecar.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
            self.convert()
        self.sidecar.write_text(json.dumps(manifest))
        with self.output.open('r+b') as handle:
            handle.seek(-1, os.SEEK_END)
            handle.write(original)
        self.assertEqual(self.convert()['status'], 'verified-existing')
        self.sidecar.write_text(json.dumps({**manifest, 'converter_version': '0'}))
        with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
            self.convert()
        self.sidecar.unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete conversion'):
            self.convert()
        self.assertEqual(converter.file_digest(self.source), (self.source_hash, self.source_size))

    def test_refuses_wrong_input_hash_even_with_existing_output(self):
        self.output.write_bytes(b'not valid')
        with mock.patch.object(converter, 'SOURCE_SHA256', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'source SHA-256'):
                self.convert()
        self.assertEqual(self.output.read_bytes(), b'not valid')

    def test_refuses_wrong_input_size(self):
        with mock.patch.object(converter, 'SOURCE_BYTES', 1):
            with self.assertRaisesRegex(ValueError, 'source size'):
                self.convert()
        self.assertFalse(self.output.exists())

    def test_refuses_wrong_output_name_and_in_place(self):
        with self.assertRaisesRegex(ValueError, 'must be named'):
            converter.convert(self.source, self.base, Path(self.case.name) / 'wrong.safetensors')
        alias = Path(self.case.name) / converter.OUTPUT_NAME
        alias.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, 'in-place'):
            converter.convert(self.source, self.base, alias)

    def test_refuses_symlinked_sidecar(self):
        self.sidecar.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, 'source overwrite|Symlinked'):
            self.convert()

    def test_refuses_manifest_without_output(self):
        self.sidecar.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Incomplete conversion'):
            self.convert()
        self.assertFalse(self.output.exists())

    def test_failure_before_publication_leaves_no_output_or_temp_files(self):
        original_writer = converter._write_temporary
        count = 0

        def fail_second(parent, chunks):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError('simulated manifest write failure')
            return original_writer(parent, chunks)

        with mock.patch.object(converter, '_write_temporary', side_effect=fail_second):
            with self.assertRaisesRegex(OSError, 'simulated'):
                self.convert()
        self.assertEqual(list(Path(self.case.name).iterdir()), [])

    def test_interrupted_pair_is_refused_on_restart(self):
        original_link = os.link
        count = 0

        def fail_second(source, target):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError('simulated interruption')
            return original_link(source, target)

        with mock.patch.object(converter.os, 'link', side_effect=fail_second):
            with self.assertRaisesRegex(OSError, 'simulated'):
                self.convert()
        self.assertTrue(self.output.is_file())
        self.assertFalse(self.sidecar.exists())
        self.assertEqual(list(Path(self.case.name).glob('.outfit-lora-*')), [])
        with self.assertRaisesRegex(ValueError, 'Incomplete conversion'):
            self.convert()

    def test_exclusive_publication_never_overwrites_concurrent_output(self):
        def concurrent_write(source, target):
            Path(target).write_bytes(b'another process created this')
            raise FileExistsError('simulated concurrent creation')

        with mock.patch.object(converter.os, 'link', side_effect=concurrent_write):
            with self.assertRaises(FileExistsError):
                self.convert()
        self.assertEqual(self.output.read_bytes(), b'another process created this')
        self.assertFalse(self.sidecar.exists())
        self.assertEqual(list(Path(self.case.name).glob('.outfit-lora-*')), [])


if __name__ == '__main__':
    unittest.main()

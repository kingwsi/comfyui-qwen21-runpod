"""Regression coverage for legacy/V3 object_info combo serialization."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cpu_smoke',ROOT/'scripts/cpu_smoke.py')
smoke=importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def runtime_info(samplers):
    return {
        'CLIPLoaderGGUF': {'input': {'required': {'clip_name': [[],{}], 'type': [['qwen_image'],{}]}}, 'output': ['CLIP']},
        'SamplerCustom': {'input': {'required': {key: [] for key in ('model','add_noise','noise_seed','cfg','positive','negative','sampler','sigmas','latent_image')}}, 'output': ['LATENT','LATENT']},
        'ManualSigmas': {'input': {'required': {'sigmas': ['STRING',{}]}}, 'output': ['SIGMAS']},
        'KSamplerSelect': {'input': {'required': {'sampler_name': samplers}}, 'output': ['SAMPLER']},
    }


class SmokeRuntime(unittest.TestCase):
    def test_legacy_and_v3_sampler_choices_are_accepted(self):
        for schema in [[['euler','heun'],{}],['COMBO',{'options':['euler','heun']}]]:
            smoke.check_turbo_runtime(runtime_info(schema))

    def test_missing_euler_is_still_rejected(self):
        for schema in [[['heun'],{}],['COMBO',{'options':['heun']}]]:
            with self.assertRaises(AssertionError):
                smoke.check_turbo_runtime(runtime_info(schema))

    def test_malformed_choices_cannot_pass_by_substring(self):
        for schema in [['euler'],['COMBO',{'options':'euler'}]]:
            with self.assertRaises(AssertionError):
                smoke.combo_options(schema)

    def test_missing_input_is_still_rejected(self):
        info=runtime_info(['COMBO',{'options':['euler']}])
        del info['SamplerCustom']['input']['required']['sigmas']
        with self.assertRaises(AssertionError):
            smoke.check_turbo_runtime(info)


if __name__=='__main__':unittest.main()

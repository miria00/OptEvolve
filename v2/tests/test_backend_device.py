"""Device detection / TFLOPS probe tests. The CPU probe runs in-process
(about 2 seconds, cached); the GPU subprocess probe is opt-in via
ATLAS_GPU_PROBE=1 so the default suite never touches the vLLM-occupied GPU."""
import json
import os

import pytest

from atlas.backend import device as dv


REQUIRED_KEYS = {"kind", "platform", "memory_gb", "tflops_f32", "tflops_f64",
                 "probe"}


def test_cpu_record_structure_and_values():
    rec = dv.detect_device(prefer_gpu=False)
    assert REQUIRED_KEYS <= set(rec)
    assert rec["platform"] == "cpu"
    assert rec["memory_gb"] > 0
    assert rec["tflops_f32"] > 0
    assert rec["tflops_f64"] > 0
    json.dumps(rec)  # JSON-serializable, card-ready


def test_probe_is_cached():
    first = dv.detect_device(prefer_gpu=False)
    again = dv.detect_device(prefer_gpu=False)
    assert again is first  # cached object, no re-probe


@pytest.mark.skipif(os.environ.get("ATLAS_GPU_PROBE") != "1",
                    reason="GPU probe is opt-in (set ATLAS_GPU_PROBE=1); "
                           "the LOCAL GPU is shared with the vLLM server")
def test_gpu_record_via_subprocess():
    rec = dv.detect_device(prefer_gpu=True, force_refresh=True)
    assert rec["platform"] != "cpu"
    assert rec["tflops_f32"] > rec["tflops_f64"]  # every GPU we own

"""Header-only structural coverage for the real external DENSE TTS checkpoint.

The user's ``VibeVoice-1.5B-bf16.safetensors`` (5,408,306,126 bytes) is the
first external dense TTS file this pack has ever been asked to load, so its
shape is worth pinning. It is 5.4 GB, so the standing rule — never load a huge
model in an automated test — applies absolutely: every assertion here reads the
safetensors HEADER only, via ``safe_open(...).get_slice(key).get_shape()``.
``get_slice`` parses the header and never materialises tensor data, so the
test costs milliseconds and no measurable memory regardless of file size.

These tests SKIP when the file is absent, so the suite stays green on a machine
that does not have it (CI, other developers). They never fail for a missing
file; they exist to catch a re-export that changes the checkpoint's structure
under the assumptions the external dense path makes of it.
"""

import os

import pytest
import torch

from ComfyUI_VibeVoice.modules.config_detect import (
    _ASR_ONLY_KEY_PREFIXES,
    _EMBEDDING_KEY_CANDIDATES,
    _FAMILY_SIGNATURES,
    fingerprint_weights,
)

# The first external dense TTS checkpoint in play. Opt-in: the file is 5.4 GB
# and lived in one developer's models directory, so there is no portable
# default and no fallback path. Set VIBEVOICE_TEST_DENSE_CHECKPOINT to run
# these; the always-runs coverage of the same header-reading code lives in
# test_config_detect.py and test_external_loader.py.
FIXTURE = os.environ.get("VIBEVOICE_TEST_DENSE_CHECKPOINT", "")

# 5,408,306,126 bytes, as reported by the filesystem.
EXPECTED_BYTES = 5_408_306_126


def _require_fixture():
    if not FIXTURE:
        pytest.skip("set VIBEVOICE_TEST_DENSE_CHECKPOINT to run this")
    if not os.path.isfile(FIXTURE):
        pytest.skip(f"fixture not present: {FIXTURE}")
    return FIXTURE


class TestDenseTTSFixtureStructure:
    """Header-only facts about the real 5.4 GB dense TTS checkpoint."""

    def test_file_size_matches_the_recorded_checkpoint(self):
        """A re-export that changes the size means different assumptions."""
        path = _require_fixture()
        assert os.path.getsize(path) == EXPECTED_BYTES

    def test_every_tensor_is_bf16(self):
        """Dense BF16: no .comfy_quant metadata, so the batch route is taken.

        This is the property that makes this file exercise the external DENSE
        branch (the plan's S4 site) rather than the quantized streaming one.
        """
        from safetensors import safe_open

        path = _require_fixture()
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            assert keys, "checkpoint has no tensors"
            for key in keys:
                # get_dtype() reads the header entry; no tensor data is read.
                assert str(f.get_slice(key).get_dtype()) == "BF16", key

    def test_carries_no_comfy_quant_metadata(self):
        """Zero .comfy_quant keys => convrot_quant_map == {} => dense route.

        This is why the file is the interesting one: it is the only external
        checkpoint that provably falls through to the un-streamed dense
        branch, since ``stream_quant_load`` requires a non-empty quant map.
        """
        from safetensors import safe_open

        path = _require_fixture()
        with safe_open(path, framework="pt", device="cpu") as f:
            quant_keys = [k for k in f.keys() if "comfy_quant" in k]
        assert quant_keys == []

    def test_all_tensors_live_under_the_model_group(self):
        """Single top-level 'model.' group, as an HF export has."""
        from safetensors import safe_open

        path = _require_fixture()
        with safe_open(path, framework="pt", device="cpu") as f:
            tops = {k.split(".")[0] for k in f.keys()}
        assert tops == {"model"}

    def test_embedding_key_is_the_tts_candidate(self):
        """The key auto-detect scans first is present, and is the 1.5B one."""
        from safetensors import safe_open

        path = _require_fixture()
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
        assert _EMBEDDING_KEY_CANDIDATES[0] in keys
        # tie_word_embeddings: the tied lm_head is legitimately absent.
        assert "lm_head.weight" not in keys

    def test_auto_detect_resolves_the_1p5b_tts_family(self):
        """End-to-end header-only: the fingerprint -> VibeVoice-1.5B.

        fingerprint_weights reads the header only (no tensor data), so this
        asserts the real routing decision the external loader would make for
        this file without loading it.
        """
        path = _require_fixture()
        fp = fingerprint_weights(path)
        assert fp.is_asr is False
        assert fp.source_key == _EMBEDDING_KEY_CANDIDATES[0]
        assert fp.config_name == "VibeVoice-1.5B"
        assert (fp.hidden_size, fp.vocab_size) == _FAMILY_SIGNATURES[
            "VibeVoice-1.5B"
        ]

    def test_is_not_misrouted_to_the_asr_family(self):
        """A dense TTS file must never classify as ASR.

        ASR and the 7B TTS family share the (3584, 152064) embedding shape, so
        the ASR gate keys on modules that exist only in an ASR state dict.
        Asserting the negative keeps that discrimination honest for this file.
        """
        from safetensors import safe_open

        path = _require_fixture()
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
        asr_hits = [
            k for k in keys
            if any(k.startswith(p) for p in _ASR_ONLY_KEY_PREFIXES)
        ]
        assert asr_hits == []

    def test_header_only_read_does_not_materialize_weights(self):
        """Non-vacuity guard on this file's whole premise.

        get_slice().get_shape() must not load tensor data. Reading one
        embedding shape and asserting the private-RSS delta stays far below
        the file size proves the test cannot silently become a 5.4 GB load.
        """
        from safetensors import safe_open

        path = _require_fixture()

        def rss():
            import ctypes

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            c = _PMC()
            c.cb = ctypes.sizeof(_PMC)
            if not psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            return c.PrivateUsage

        before = rss()
        with safe_open(path, framework="pt", device="cpu") as f:
            shape = f.get_slice(_EMBEDDING_KEY_CANDIDATES[0]).get_shape()
        after = rss()

        # The 1.5B signature: (hidden_size, vocab_size).
        assert tuple(shape) == (151936, 1536)
        # A real load would cost gigabytes; the header read must not.
        assert (after - before) < 64 * 1024 * 1024

    def test_dtype_guard_set_excludes_bf16(self):
        """The S2 guard set must not reject this file's only dtype.

        Ties the fixture to the quantized-storage contract: a BF16 checkpoint
        has to pass, or the dense route would reject a legitimate file.
        """
        from ComfyUI_VibeVoice.modules.loader import QUANT_STORAGE_DTYPES

        assert torch.bfloat16 not in QUANT_STORAGE_DTYPES
        assert torch.float32 not in QUANT_STORAGE_DTYPES

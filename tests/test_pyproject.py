"""Tests for pyproject.toml - V3 compliance."""

import os
import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _load_pyproject():
    """Load pyproject.toml."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject_path = os.path.join(root, "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)


class TestPyprojectCompliance:
    """Test pyproject.toml V3 compliance."""

    def test_requires_python(self):
        data = _load_pyproject()
        assert "requires-python" in data["project"]
        assert ">=3.10" in data["project"]["requires-python"]

    def test_requires_comfyui(self):
        data = _load_pyproject()
        assert "tool" in data
        assert "comfy" in data["tool"]
        assert "requires-comfyui" in data["tool"]["comfy"]
        assert ">=0.28.0" in data["tool"]["comfy"]["requires-comfyui"]

    def test_publisher_id(self):
        data = _load_pyproject()
        assert "PublisherId" in data["tool"]["comfy"]
        assert data["tool"]["comfy"]["PublisherId"]

    def test_display_name(self):
        data = _load_pyproject()
        assert "DisplayName" in data["tool"]["comfy"]

    def test_dependencies_present(self):
        data = _load_pyproject()
        deps = data["project"]["dependencies"]
        dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].strip() for d in deps]
        assert "torch" in dep_names
        assert "transformers" in dep_names
        assert "huggingface_hub" in dep_names

    def test_project_urls(self):
        data = _load_pyproject()
        assert "urls" in data["project"]
        urls = data["project"]["urls"]
        assert "Repository" in urls

    def test_classifiers(self):
        data = _load_pyproject()
        assert "classifiers" in data["project"]
        assert len(data["project"]["classifiers"]) > 0

    def test_version(self):
        data = _load_pyproject()
        assert "version" in data["project"]


class TestAudioDependencySurface:
    """Phase 4: librosa/scipy must be optional; torchaudio is the primary audio lib."""

    @staticmethod
    def _dep_names(data):
        deps = data["project"]["dependencies"]
        return [d.split(">=")[0].split("==")[0].split("[")[0].strip() for d in deps]

    def test_librosa_not_in_hard_dependencies(self):
        data = _load_pyproject()
        assert "librosa" not in self._dep_names(data), \
            "librosa must be optional, not a hard dependency"

    def test_scipy_not_in_hard_dependencies(self):
        data = _load_pyproject()
        assert "scipy" not in self._dep_names(data), \
            "scipy must be optional (torchaudio is the primary resampler)"

    def test_torchaudio_in_hard_dependencies(self):
        data = _load_pyproject()
        assert "torchaudio" in self._dep_names(data), \
            "torchaudio is the primary audio library and must stay a hard dependency"

    def test_soundfile_in_hard_dependencies(self):
        data = _load_pyproject()
        assert "soundfile" in self._dep_names(data), \
            "soundfile is the primary audio encoder and must stay a hard dependency"

    def test_audio_extra_optional_group_declared(self):
        data = _load_pyproject()
        extras = data["project"].get("optional-dependencies", {})
        assert "audio-extra" in extras, "optional-dependencies.audio-extra must exist"
        extra_names = [d.split(">=")[0].split("==")[0].strip() for d in extras["audio-extra"]]
        assert "scipy" in extra_names
        assert "librosa" in extra_names

    def test_requirements_txt_has_no_librosa(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "requirements.txt"), "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        names = [ln.split(">=")[0].split("==")[0].strip() for ln in lines]
        assert "librosa" not in names
        assert "scipy" not in names
        assert "torchaudio" in names
